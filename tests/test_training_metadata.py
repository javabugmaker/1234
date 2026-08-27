from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import neural_a_share.pipeline as pipeline_module
from neural_a_share.config import AppConfig, PathsConfig, WalkForwardConfig
from neural_a_share.model import CheckpointMetadata
from neural_a_share.pipeline import NeuralAlphaPipeline


def _config(tmp_path: Path) -> AppConfig:
    paths = PathsConfig(
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        logs_dir=tmp_path / "logs",
        docs_dir=tmp_path / "docs",
    )
    walk_forward = WalkForwardConfig(
        initial_train_days=5,
        validation_days=3,
        test_days=2,
        step_days=2,
        purge_days=2,
        embargo_days=1,
    )
    return replace(AppConfig(), paths=paths, walk_forward=walk_forward)


def test_training_cutoff_is_last_train_date_not_latest_data_date(
    tmp_path, monkeypatch
) -> None:
    pipeline = NeuralAlphaPipeline(_config(tmp_path))
    dates = pd.bdate_range("2026-01-02", periods=12)
    monkeypatch.setattr(pipeline.store, "latest_bar_date", lambda: dates[-1])

    class _Loader:
        def scan(self, mature_cutoff=None):
            assert mature_cutoff == dates[-1]
            return SimpleNamespace(dates=dates)

        def load_splits(self, _index, specs, mature_cutoff=None):
            assert mature_cutoff == dates[-1]
            return {
                name: pd.DataFrame(
                    {"trade_date": spec.dates, "symbol": "A.SH"}
                )
                for name, spec in specs.items()
            }

    captured: dict[str, object] = {}

    def fake_fit(
        train,
        validation,
        model_version,
        training_cutoff,
        survivorship_status="PASS",
        training_progress=None,
        **kwargs,
    ):
        captured["training_cutoff"] = training_cutoff
        captured.update(kwargs)
        metadata = CheckpointMetadata(
            model_version=model_version,
            training_cutoff=str(pd.Timestamp(training_cutoff).date()),
            feature_names=("f1",),
            horizons=(20, 40, 60),
            hidden_dims=(4, 2, 1),
            dropout=0.0,
            metrics={"rank_ic_20": 0.01},
            survivorship_status=survivorship_status,
            data_cutoff=str(pd.Timestamp(kwargs["data_cutoff"]).date()),
            train_start=str(pd.Timestamp(kwargs["train_period"][0]).date()),
            train_end=str(pd.Timestamp(kwargs["train_period"][1]).date()),
            validation_start=str(
                pd.Timestamp(kwargs["validation_period"][0]).date()
            ),
            validation_end=str(
                pd.Timestamp(kwargs["validation_period"][1]).date()
            ),
        )
        result = SimpleNamespace(
            history=(
                {"epoch": 1.0, "train_loss": 0.2, "validation_loss": 0.1},
            ),
            epochs_trained=1,
            best_validation_loss=0.1,
            stopped_early=False,
            device="cpu",
            amp_enabled=False,
        )
        return object(), object(), metadata, result

    def fake_save(path, *_args, **_kwargs):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
        return destination

    monkeypatch.setattr(pipeline, "_research_loader", lambda _strict: _Loader())
    monkeypatch.setattr(pipeline, "_fit_one", fake_fit)
    monkeypatch.setattr(pipeline_module, "save_checkpoint", fake_save)
    monkeypatch.setattr(
        pipeline_module, "make_model_version", lambda cutoff, _features: f"mlp-{pd.Timestamp(cutoff).date()}"
    )

    metadata = pipeline.train()

    expected_cutoff = dates[6]
    assert captured["training_cutoff"] == expected_cutoff
    assert metadata.training_cutoff == str(expected_cutoff.date())
    manifest = pipeline.store.read_manifest("training")
    assert manifest["training_cutoff"].startswith(str(expected_cutoff.date()))
    assert manifest["data_cutoff"].startswith(str(dates[-1].date()))
    assert manifest["training_cutoff_semantics"] == "last_train_signal_date"
    registry_model = pipeline.registry.read()["models"][metadata.model_version]
    assert registry_model["training_cutoff"] == str(expected_cutoff.date())
    assert registry_model["data_cutoff"] == str(dates[-1].date())
    audit_path = Path(registry_model["training_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["training_cutoff"].startswith(str(expected_cutoff.date()))
    assert audit["data_cutoff"].startswith(str(dates[-1].date()))
