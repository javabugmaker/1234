from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised through explicit dependency error
    torch = None
    nn = None
    DataLoader = TensorDataset = None

from .config import ModelConfig

TrainingProgress = Callable[[int, int, float, float], None]


if torch is not None:

    class _VectorizedTensorDataset(TensorDataset):
        """Fetch a complete batch with one tensor index operation.

        PyTorch's normal ``TensorDataset`` path calls ``__getitem__`` once per
        row before collating.  Implementing ``__getitems__`` keeps the same
        sampler and row order while removing millions of Python-level calls.
        """

        def __getitems__(self, indices: Sequence[int]) -> tuple[Any, ...]:
            return tuple(tensor[indices] for tensor in self.tensors)


else:  # pragma: no cover - dependency error is raised before construction

    class _VectorizedTensorDataset:  # type: ignore[no-redef]
        pass


def _identity_collate(batch: Any) -> Any:
    return batch


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required; install project dependencies with: pip install -e .")
    return torch


def select_device(preference: str = "auto", torch_module: Any | None = None) -> str:
    module = torch_module if torch_module is not None else require_torch()
    preference = preference.lower()
    cuda_available = bool(module.cuda.is_available())
    if preference == "auto":
        return "cuda" if cuda_available else "cpu"
    if preference.startswith("cuda"):
        return preference if cuda_available else "cpu"
    if preference != "cpu":
        raise ValueError(f"unsupported device preference: {preference}")
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class FeatureNormalizer:
    center: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "FeatureNormalizer":
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("normalizer input must be a 2D matrix")
        # Compute the identical robust statistics one column at a time.  The
        # previous whole-matrix float64 cast costs ~2.3 GiB at 2.5M x 122 even
        # when the source features are already float32.
        self.center = np.empty(array.shape[1], dtype=np.float64)
        self.scale = np.empty(array.shape[1], dtype=np.float64)
        for index in range(array.shape[1]):
            column = np.asarray(array[:, index], dtype=np.float64)
            self.center[index] = np.nanmedian(column)
            q25, q75 = np.nanquantile(column, (0.25, 0.75))
            self.scale[index] = q75 - q25
        self.center = np.where(np.isfinite(self.center), self.center, 0.0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-6), self.scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise ValueError("normalizer is not fitted")
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != len(self.center):
            raise ValueError("normalizer input shape does not match fitted features")
        output = np.empty(array.shape, dtype="float32")
        for start in range(0, len(array), 65_536):
            stop = min(start + 65_536, len(array))
            block = np.asarray(array[start:stop], dtype=np.float64).copy()
            np.copyto(block, self.center, where=~np.isfinite(block))
            block -= self.center
            block /= self.scale
            np.clip(block, -10.0, 10.0, out=block)
            output[start:stop] = block
        return output

    def state_dict(self) -> dict[str, list[float]]:
        if self.center is None or self.scale is None:
            raise ValueError("normalizer is not fitted")
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Sequence[float]]) -> "FeatureNormalizer":
        return cls(np.asarray(state["center"], dtype=float), np.asarray(state["scale"], dtype=float))


if nn is not None:

    class MultiTaskMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int] = (128, 64, 32),
            dropout: float = 0.15,
            horizons: Sequence[int] = (20, 40, 60),
        ) -> None:
            super().__init__()
            dims = [int(input_dim), *(int(value) for value in hidden_dims)]
            blocks: list[nn.Module] = [nn.LayerNorm(dims[0])]
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                blocks.extend(
                    [
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                    ]
                )
            self.trunk = nn.Sequential(*blocks)
            self.horizons = tuple(int(h) for h in horizons)
            self.heads = nn.ModuleDict({str(h): nn.Linear(dims[-1], 1) for h in self.horizons})
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                    nn.init.zeros_(module.bias)

        def forward(self, values: Any) -> Any:
            representation = self.trunk(values)
            return torch.cat([self.heads[str(h)](representation) for h in self.horizons], dim=1)

else:

    class MultiTaskMLP:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            require_torch()


def _masked_huber(prediction: Any, target: Any, delta: float) -> Any:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        raise ValueError("batch contains no mature labels")
    losses = torch.nn.functional.huber_loss(
        prediction[mask], target[mask], delta=float(delta), reduction="mean"
    )
    return losses


def _masked_huber_known_nonempty(prediction: Any, target: Any, delta: float) -> Any:
    """Huber loss for frames already filtered to at least one label per row.

    Avoiding ``bool(mask.any())`` removes a forced CUDA synchronization from
    every training batch.  ``_fit_one`` guarantees this precondition before
    tensors reach the trainer.
    """

    mask = torch.isfinite(target)
    return torch.nn.functional.huber_loss(
        prediction[mask], target[mask], delta=float(delta), reduction="mean"
    )


@dataclass(frozen=True)
class TrainingResult:
    best_validation_loss: float
    epochs_trained: int
    stopped_early: bool
    device: str
    amp_enabled: bool
    history: tuple[dict[str, float], ...]


class NeuralTrainer:
    def __init__(self, config: ModelConfig) -> None:
        require_torch()
        self.config = config

    def fit(
        self,
        model: MultiTaskMLP,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        progress: TrainingProgress | None = None,
    ) -> TrainingResult:
        set_seed(self.config.seed)
        device_name = select_device(self.config.device)
        device = torch.device(device_name)
        model.to(device)
        dataset = _VectorizedTensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.float32),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(int(self.config.batch_size), len(dataset)),
            shuffle=True,
            num_workers=int(self.config.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=_identity_collate,
        )
        validation_features = torch.as_tensor(validation_x, dtype=torch.float32, device=device)
        validation_targets = torch.as_tensor(validation_y, dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        amp_enabled = device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:  # PyTorch 2.3 compatibility
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience = 0
        history: list[dict[str, float]] = []
        stopped_early = False

        for epoch in range(1, int(self.config.epochs) + 1):
            model.train()
            running = torch.zeros((), dtype=torch.float32, device=device)
            batches = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    predictions = model(batch_x)
                    loss = _masked_huber_known_nonempty(
                        predictions, batch_y, self.config.huber_delta
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                running += loss.detach().float()
                batches += 1
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp_enabled):
                validation_predictions = model(validation_features)
                validation_loss = float(
                    _masked_huber_known_nonempty(
                        validation_predictions, validation_targets, self.config.huber_delta
                    ).detach().cpu()
                )
            train_loss = float((running / max(batches, 1)).cpu())
            history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss})
            if progress is not None:
                progress(
                    epoch,
                    int(self.config.epochs),
                    train_loss,
                    validation_loss,
                )
            if validation_loss < best_loss - float(self.config.min_delta):
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= int(self.config.patience):
                    stopped_early = True
                    break
        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.to("cpu")
        return TrainingResult(
            best_validation_loss=best_loss,
            epochs_trained=len(history),
            stopped_early=stopped_early,
            device=device_name,
            amp_enabled=amp_enabled,
            history=tuple(history),
        )


def predict_array(model: MultiTaskMLP, values: np.ndarray, device_preference: str = "auto") -> np.ndarray:
    require_torch()
    device = torch.device(select_device(device_preference))
    model.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 8192):
            tensor = torch.as_tensor(values[start : start + 8192], dtype=torch.float32, device=device)
            outputs.append(model(tensor).float().cpu().numpy())
    model.to("cpu")
    return np.concatenate(outputs) if outputs else np.empty((0, len(model.horizons)), dtype="float32")


def inference_frame(
    model: MultiTaskMLP,
    normalizer: FeatureNormalizer,
    frame: pd.DataFrame,
    names: Sequence[str],
    device_preference: str = "auto",
) -> pd.DataFrame:
    matrix = normalizer.transform(frame[list(names)].to_numpy())
    outputs = predict_array(model, matrix, device_preference)
    result = frame[["symbol", "trade_date"]].copy()
    for index, horizon in enumerate(model.horizons):
        result[f"Alpha{horizon}"] = outputs[:, index]
    alpha_columns = [f"Alpha{h}" for h in model.horizons]
    result["NeuralAlpha"] = result[alpha_columns].mean(axis=1)
    result["NeuralRank"] = result.groupby("trade_date", sort=False)["NeuralAlpha"].rank(
        ascending=False, method="first"
    ).astype(int)
    return result.sort_values(["trade_date", "NeuralRank"]).reset_index(drop=True)


@dataclass(frozen=True)
class CheckpointMetadata:
    model_version: str
    training_cutoff: str
    feature_names: tuple[str, ...]
    horizons: tuple[int, ...]
    hidden_dims: tuple[int, ...]
    dropout: float
    metrics: dict[str, float]
    survivorship_status: str = "PASS"


def make_model_version(training_cutoff: str | pd.Timestamp, feature_names: Iterable[str]) -> str:
    digest = hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"mlp-{pd.Timestamp(training_cutoff).date()}-{timestamp}-{digest}"


def save_checkpoint(
    path: str | Path,
    model: MultiTaskMLP,
    normalizer: FeatureNormalizer,
    metadata: CheckpointMetadata,
    training_result: TrainingResult | None = None,
) -> Path:
    require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "metadata": asdict(metadata),
        "training_result": asdict(training_result) if training_result else None,
    }
    fd, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[MultiTaskMLP, FeatureNormalizer, CheckpointMetadata]:
    require_torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    raw = payload["metadata"]
    metadata = CheckpointMetadata(
        model_version=raw["model_version"],
        training_cutoff=raw["training_cutoff"],
        feature_names=tuple(raw["feature_names"]),
        horizons=tuple(raw["horizons"]),
        hidden_dims=tuple(raw["hidden_dims"]),
        dropout=float(raw["dropout"]),
        metrics=dict(raw.get("metrics", {})),
        survivorship_status=str(raw.get("survivorship_status", "PASS")),
    )
    model = MultiTaskMLP(
        input_dim=len(metadata.feature_names),
        hidden_dims=metadata.hidden_dims,
        dropout=metadata.dropout,
        horizons=metadata.horizons,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, FeatureNormalizer.from_state_dict(payload["normalizer"]), metadata


class ModelRegistry:
    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.models_dir / "registry.json"

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"champion": None, "challengers": [], "models": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def register(self, metadata: CheckpointMetadata, checkpoint: Path, role: str = "challenger") -> None:
        registry = self.read()
        registry.setdefault("models", {})[metadata.model_version] = {
            "checkpoint": str(checkpoint),
            "training_cutoff": metadata.training_cutoff,
            "metrics": metadata.metrics,
            "survivorship_status": metadata.survivorship_status,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        if role == "champion":
            registry["champion"] = metadata.model_version
        else:
            challengers = set(registry.setdefault("challengers", []))
            challengers.add(metadata.model_version)
            registry["challengers"] = sorted(challengers)
        self._write(registry)

    def promote(self, model_version: str) -> None:
        registry = self.read()
        if model_version not in registry.get("models", {}):
            raise KeyError(model_version)
        previous = registry.get("champion")
        registry["champion"] = model_version
        challengers = set(registry.get("challengers", []))
        challengers.discard(model_version)
        if previous:
            challengers.add(previous)
        registry["challengers"] = sorted(challengers)
        self._write(registry)

    def champion_checkpoint(self) -> Path:
        registry = self.read()
        champion = registry.get("champion")
        if not champion:
            raise FileNotFoundError("no champion model is registered")
        return Path(registry["models"][champion]["checkpoint"])

    def _write(self, payload: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix="registry", suffix=".tmp", dir=self.models_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
