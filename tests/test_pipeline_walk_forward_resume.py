from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

import neural_a_share.pipeline as pipeline_module
from neural_a_share.config import AppConfig, PathsConfig
from neural_a_share.features import FEATURE_NAMES
from neural_a_share.pipeline import NeuralAlphaPipeline
from neural_a_share.walk_forward import WalkForwardFold


def _config(tmp_path) -> AppConfig:
    paths = PathsConfig(
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        logs_dir=tmp_path / "logs",
        docs_dir=tmp_path / "docs",
    )
    return replace(AppConfig(), paths=paths)


def _research_frame(dates: pd.DatetimeIndex, include_labels: bool) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [dates, ["A.SH", "B.SZ"]], names=["trade_date", "symbol"]
    ).to_frame(index=False)
    data = {
        "symbol": index["symbol"],
        "trade_date": index["trade_date"],
        **{
            feature: np.full(len(index), 0.1, dtype="float32")
            for feature in FEATURE_NAMES
        },
    }
    frame = pd.DataFrame(data)
    if include_labels:
        for horizon in (20, 40, 60):
            frame[f"label_{horizon}"] = pd.Series(
                0.001 * horizon, index=frame.index, dtype="float32"
            )
    return frame


def test_quick_walk_forward_cache_resumes_and_is_reused_by_full_run(
    tmp_path, monkeypatch
) -> None:
    pipeline = NeuralAlphaPipeline(_config(tmp_path))
    dates = pd.bdate_range("2024-01-02", periods=12)
    benchmark = pipeline.config.tickflow.benchmark
    bars = pd.DataFrame(
        {
            "symbol": benchmark,
            "trade_date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
            "amount": 100_000_000.0,
        }
    )
    pipeline.store.upsert_bars(bars)
    pipeline.store.write_derived_year(
        "features", 2024, pd.DataFrame({"symbol": ["A.SH"], "trade_date": [dates[0]]})
    )
    pipeline.store.write_derived_year(
        "labels", 2024, pd.DataFrame({"symbol": ["A.SH"], "trade_date": [dates[0]]})
    )
    folds = [
        WalkForwardFold(0, dates[:3], dates[4:5], dates[6:7], 60, 5),
        WalkForwardFold(1, dates[:5], dates[6:7], dates[8:9], 60, 5),
    ]

    class _Splitter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def split(self, _dates):
            return iter(folds)

    class _Loader:
        def scan(self):
            return object()

        def load_splits(self, _index, specs):
            return {
                name: _research_frame(spec.dates, include_labels=spec.include_labels)
                for name, spec in specs.items()
            }

    fit_calls: list[str] = []

    def fake_fit(
        train,
        validation,
        model_version,
        training_cutoff,
        survivorship_status="PASS",
        training_progress=None,
    ):
        fit_calls.append(model_version)
        if training_progress is not None:
            training_progress(1, 1, 0.1, 0.1)
        result = SimpleNamespace(epochs_trained=1, best_validation_loss=0.1)
        return object(), object(), object(), result

    def fake_inference(_model, _normalizer, frame, _features, _device):
        scored = frame[["symbol", "trade_date"]].copy()
        scored["Alpha20"] = pd.Series([0.02, 0.01], dtype="float32")
        scored["Alpha40"] = pd.Series([0.03, 0.02], dtype="float32")
        scored["Alpha60"] = pd.Series([0.04, 0.03], dtype="float32")
        scored["NeuralAlpha"] = pd.Series([0.03, 0.02], dtype="float32")
        scored["NeuralRank"] = [1, 2]
        return scored

    monkeypatch.setattr(pipeline_module, "PurgedWalkForward", _Splitter)
    monkeypatch.setattr(pipeline_module, "inference_frame", fake_inference)
    monkeypatch.setattr(pipeline, "_research_loader", lambda _strict: _Loader())
    monkeypatch.setattr(pipeline, "_fit_one", fake_fit)

    quick = pipeline.walk_forward(
        max_folds=1, allow_degraded_survivorship=True
    )
    repeated = pipeline.walk_forward(
        max_folds=1, allow_degraded_survivorship=True
    )
    full = pipeline.walk_forward(allow_degraded_survivorship=True)

    assert quick.coverage_status == "PARTIAL"
    assert repeated.cached_folds == 1
    assert full.coverage_status == "FULL"
    assert full.cached_folds == 1
    assert len(fit_calls) == 2
    published = pd.read_parquet(full.predictions_path)
    assert published["coverage_status"].eq("FULL").all()
    assert set(published["fold_id"]) == {0, 1}
