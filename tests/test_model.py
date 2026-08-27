from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from neural_a_share.config import ModelConfig
from neural_a_share.model import (
    CheckpointMetadata,
    FeatureNormalizer,
    MultiTaskMLP,
    NeuralTrainer,
    inference_frame,
    load_checkpoint,
    save_checkpoint,
    select_device,
    _VectorizedTensorDataset,
    _identity_collate,
)
from torch.utils.data import DataLoader


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)


def test_cuda_auto_and_cpu_fallback() -> None:
    assert select_device("auto", _FakeTorch(True)) == "cuda"
    assert select_device("auto", _FakeTorch(False)) == "cpu"
    assert select_device("cuda", _FakeTorch(False)) == "cpu"


def test_checkpoint_round_trip(tmp_path) -> None:
    names = ("f1", "f2", "f3")
    model = MultiTaskMLP(3, hidden_dims=(8, 4, 2), dropout=0.0, horizons=(20, 40, 60))
    normalizer = FeatureNormalizer().fit(np.arange(30, dtype=float).reshape(10, 3))
    metadata = CheckpointMetadata(
        "test-v1",
        "2024-01-31",
        names,
        (20, 40, 60),
        (8, 4, 2),
        0.0,
        {"rank_ic_20": 0.1},
        survivorship_status="DEGRADED",
    )
    path = save_checkpoint(tmp_path / "checkpoint.pt", model, normalizer, metadata)
    loaded, loaded_normalizer, loaded_meta = load_checkpoint(path)
    assert loaded_meta == metadata
    sample = np.ones((2, 3), dtype="float32")
    assert np.allclose(model(torch.tensor(sample)).detach().numpy(), loaded(torch.tensor(sample)).detach().numpy())
    assert np.allclose(normalizer.transform(sample), loaded_normalizer.transform(sample))


def test_feature_normalizer_matches_vectorized_reference_without_wide_float64_copy() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(2_000, 17)).astype("float32")
    values[::113, 3] = np.nan
    reference = values.astype("float64")
    center = np.nanmedian(reference, axis=0)
    scale = np.nanquantile(reference, 0.75, axis=0) - np.nanquantile(
        reference, 0.25, axis=0
    )
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    expected = np.where(np.isfinite(reference), reference, center)
    expected = np.clip((expected - center) / scale, -10.0, 10.0).astype("float32")

    normalizer = FeatureNormalizer().fit(values)
    actual = normalizer.transform(values)
    assert actual.dtype == np.dtype("float32")
    assert np.array_equal(normalizer.center, center)
    assert np.array_equal(normalizer.scale, scale)
    assert np.array_equal(actual, expected)


def test_mlp_trains_three_heads_on_cpu() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(160, 8)).astype("float32")
    y = np.column_stack([x[:, 0] * 0.01, x[:, 1] * 0.02, x[:, 2] * 0.03]).astype("float32")
    config = replace(
        ModelConfig(),
        hidden_dims=(16, 8, 4),
        epochs=3,
        patience=2,
        batch_size=64,
        device="cpu",
    )
    model = MultiTaskMLP(8, config.hidden_dims, 0.05)
    progress = []
    result = NeuralTrainer(config).fit(
        model,
        x[:120],
        y[:120],
        x[120:],
        y[120:],
        progress=lambda *values: progress.append(values),
    )
    assert result.device == "cpu"
    assert result.epochs_trained >= 1
    assert len(progress) == result.epochs_trained
    assert model(torch.tensor(x[:2])).shape == (2, 3)


def test_vectorized_tensor_dataset_preserves_batches_and_row_order() -> None:
    features = torch.arange(310, dtype=torch.float32).reshape(31, 10)
    targets = torch.arange(93, dtype=torch.float32).reshape(31, 3)
    loader = DataLoader(
        _VectorizedTensorDataset(features, targets),
        batch_size=7,
        shuffle=False,
        collate_fn=_identity_collate,
    )
    batches = list(loader)
    assert torch.equal(torch.cat([batch[0] for batch in batches]), features)
    assert torch.equal(torch.cat([batch[1] for batch in batches]), targets)


def test_inference_outputs_neural_rank() -> None:
    model = MultiTaskMLP(2, hidden_dims=(4, 3, 2), dropout=0.0)
    normalizer = FeatureNormalizer().fit(np.array([[0, 0], [1, 1], [2, 2]], dtype=float))
    frame = pd.DataFrame(
        {"symbol": ["A", "B", "C"], "trade_date": pd.Timestamp("2024-01-02"), "f1": [0.0, 1.0, 2.0], "f2": [0.0, 1.0, 2.0]}
    )
    result = inference_frame(model, normalizer, frame, ["f1", "f2"], "cpu")
    assert {"Alpha20", "Alpha40", "Alpha60", "NeuralAlpha", "NeuralRank"} <= set(result)
    assert sorted(result["NeuralRank"].tolist()) == [1, 2, 3]
