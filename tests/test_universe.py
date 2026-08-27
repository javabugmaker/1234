from __future__ import annotations

import pandas as pd

from neural_a_share.universe import (
    filter_catalog,
    filter_degraded_symbol_universe,
    filter_feature_coverage,
    is_a_share_stock_symbol,
)


def test_a_share_symbol_classifier_excludes_etfs_and_benchmark() -> None:
    for symbol in ("600000.SH", "688981.SH", "000001.SZ", "301578.SZ", "920871.BJ"):
        assert is_a_share_stock_symbol(symbol)
    for symbol in ("158010.SZ", "560990.SH", "000300.SH", "900901.SH", "200001.SZ"):
        assert not is_a_share_stock_symbol(symbol)


def test_observed_catalog_asset_type_wins_and_blank_type_has_stock_fallback() -> None:
    catalog = pd.DataFrame(
        {
            "symbol": [
                "600000.SH",
                "560990.SH",
                "000001.SZ",
                "301578.SZ",
                "688981.SH",
            ],
            "instrument_type": ["stock", "etf", "", "fund", pd.NA],
        }
    )

    filtered = filter_catalog(catalog, ["stock"])

    assert filtered["symbol"].tolist() == [
        "600000.SH",
        "000001.SZ",
        "688981.SH",
    ]


def test_degraded_asset_filter_does_not_apply_todays_catalog_membership() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ", "560990.SH"],
            "trade_date": pd.to_datetime(["2006-01-04"] * 3),
        }
    )

    filtered = filter_degraded_symbol_universe(frame, ["stock"])

    assert filtered["symbol"].tolist() == ["600000.SH", "000001.SZ"]


def test_feature_coverage_is_float32_and_filters_only_incomplete_rows() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ", "300001.SZ"],
            "f1": pd.Series([1.0, 1.0, None], dtype="float32"),
            "f2": pd.Series([2.0, None, None], dtype="float32"),
            "f3": pd.Series([3.0, 3.0, 3.0], dtype="float32"),
            "f4": pd.Series([4.0, 4.0, None], dtype="float32"),
        }
    )

    filtered = filter_feature_coverage(frame, ["f1", "f2", "f3", "f4"], 0.75)

    assert filtered["symbol"].tolist() == ["600000.SH", "000001.SZ"]
    assert filtered["FeatureCoverage"].dtype == "float32"
    assert filtered["FeatureCoverage"].tolist() == [1.0, 0.75]
