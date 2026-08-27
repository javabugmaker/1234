from __future__ import annotations

import pandas as pd
import pytest

from neural_a_share.config import AppConfig, PathsConfig
from neural_a_share.data.pit import (
    PITViolation,
    assert_no_future_information,
    filter_available_asof,
    reconstruct_pit_prices,
)
from neural_a_share.data.store import ParquetStore
from neural_a_share.pipeline import NeuralAlphaPipeline


def test_filter_available_asof_never_returns_future_rows() -> None:
    frame = pd.DataFrame(
        {"value": [1, 2, 3], "information_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])}
    )
    result = filter_available_asof(frame, "2024-01-03")
    assert result["value"].tolist() == [1, 2]


def test_future_feature_information_is_rejected() -> None:
    frame = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2024-01-02"]), "feature_information_date": pd.to_datetime(["2024-01-03"])}
    )
    with pytest.raises(PITViolation, match="future information"):
        assert_no_future_information(frame)


def test_pit_price_chain_uses_same_day_prev_close_only() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "trade_date": pd.date_range("2024-01-01", periods=3),
            "open": [100.0, 50.0, 51.0],
            "high": [101.0, 52.0, 53.0],
            "low": [99.0, 49.0, 50.0],
            "close": [100.0, 51.0, 52.0],
            "prev_close": [100.0, 50.0, 51.0],
        }
    )
    pit = reconstruct_pit_prices(bars)
    assert pit.loc[1, "daily_total_return"] == pytest.approx(0.02)
    assert pit.loc[1, "pit_close"] == pytest.approx(1.02)


def test_universe_asof_refuses_today_membership_for_past(tmp_path) -> None:
    store = ParquetStore(tmp_path)
    catalog = pd.DataFrame({"symbol": ["A", "B"]})
    store.write_universe_snapshot(catalog, "2024-06-01")
    with pytest.raises(ValueError, match="survivorship bias"):
        store.read_universe_asof("2024-05-31", strict=True)
    assert set(store.read_universe_asof("2024-06-02")["symbol"]) == {"A", "B"}


def test_explicit_degraded_membership_preserves_history_without_silent_strict_fallback(
    tmp_path,
) -> None:
    paths = PathsConfig(
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        logs_dir=tmp_path / "logs",
        docs_dir=tmp_path / "docs",
    )
    pipeline = NeuralAlphaPipeline(AppConfig(paths=paths))
    pipeline.store.write_universe_snapshot(
        pd.DataFrame(
            {"symbol": ["600000.SH"], "instrument_type": ["stock"]}
        ),
        "2024-06-01",
    )
    historical = pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ", "560990.SH"],
            "trade_date": pd.to_datetime(["2020-01-02"] * 3),
            "label_20": [0.1, 0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="--allow-degraded-survivorship"):
        pipeline._apply_observed_membership(historical, strict=True)

    degraded = pipeline._apply_observed_membership(historical, strict=False)
    assert degraded["symbol"].tolist() == ["600000.SH", "000001.SZ"]


def test_strict_membership_uses_only_observed_snapshot_members(tmp_path) -> None:
    paths = PathsConfig(
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        logs_dir=tmp_path / "logs",
        docs_dir=tmp_path / "docs",
    )
    pipeline = NeuralAlphaPipeline(AppConfig(paths=paths))
    pipeline.store.write_universe_snapshot(
        pd.DataFrame(
            {"symbol": ["600000.SH"], "instrument_type": ["stock"]}
        ),
        "2024-06-01",
    )
    observed = pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ"],
            "trade_date": pd.to_datetime(["2024-06-03", "2024-06-03"]),
        }
    )

    strict = pipeline._apply_observed_membership(observed, strict=True)
    assert strict["symbol"].tolist() == ["600000.SH"]
