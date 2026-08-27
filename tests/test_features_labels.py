from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

from neural_a_share.config import FeatureConfig
from neural_a_share.features import FEATURE_NAMES, build_features
from neural_a_share.labels import make_labels, mature_labels_only


def test_feature_count_and_causality(synthetic_bars: pd.DataFrame) -> None:
    result = build_features(synthetic_bars, FeatureConfig(min_cross_section=5))
    assert len(result.names) == 122
    assert 80 <= len(result.names) <= 160
    assert set(result.names) == set(FEATURE_NAMES)
    assert (result.frame["feature_information_date"] <= result.frame["trade_date"]).all()


def test_label_uses_next_open_and_horizon_close() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    rows = []
    for symbol, prices in {
        "000300.SH": [100, 100, 100, 100, 100, 100],
        "A.SH": [10, 11, 12, 13, 14, 15],
    }.items():
        previous = prices[0]
        for date, close in zip(dates, prices):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": float(close),
                    "high": float(close),
                    "low": float(close),
                    "close": float(close),
                    "prev_close": float(previous),
                    "volume": 1_000,
                    "amount": 100_000,
                }
            )
            previous = close
    labels = make_labels(pd.DataFrame(rows), horizons=[2], benchmark="000300.SH").frame
    first = labels[(labels["symbol"] == "A.SH") & (labels["trade_date"] == dates[0])].iloc[0]
    # Signal at day 0, entry at day 1 open=11, exit at day 2 close=12.
    assert first["raw_return_2"] == pytest.approx(12 / 11 - 1)
    assert first["label_2"] == pytest.approx(12 / 11 - 1)
    assert first["label_available_date_2"] == dates[2]


def test_unmature_labels_are_not_visible(synthetic_bars: pd.DataFrame) -> None:
    cutoff = pd.Timestamp(synthetic_bars["trade_date"].max()) - timedelta(days=40)
    labels = make_labels(synthetic_bars, horizons=[20, 40, 60], asof_date=cutoff).frame
    immature = labels[pd.to_datetime(labels["label_available_date_60"]) > cutoff]
    assert immature["label_60"].isna().all()
    mature = mature_labels_only(labels, cutoff, [20, 40, 60])
    assert (pd.to_datetime(mature["label_available_date_60"]) <= cutoff).all()


def test_suspended_entry_produces_no_label() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    benchmark = pd.DataFrame(
        {
            "symbol": "000300.SH",
            "trade_date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "prev_close": 100.0,
            "volume": 1000,
            "amount": 100000,
        }
    )
    stock = benchmark.copy()
    stock["symbol"] = "A.SH"
    stock = stock[stock["trade_date"] != dates[1]]
    labels = make_labels(pd.concat([benchmark, stock]), horizons=[2]).frame
    row = labels[(labels["symbol"] == "A.SH") & (labels["trade_date"] == dates[0])].iloc[0]
    assert np.isnan(row["label_2"])
