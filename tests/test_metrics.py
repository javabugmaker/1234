from __future__ import annotations

import numpy as np
import pandas as pd

from neural_a_share.metrics import newey_west_t, rank_ic_by_date, summarize_ic


def test_rank_ic_is_one_for_identical_cross_sectional_order() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 4 + ["2024-01-03"] * 4,
            "prediction": [1, 2, 3, 4, 4, 3, 2, 1],
            "label": [10, 20, 30, 40, 40, 30, 20, 10],
        }
    )
    ic = rank_ic_by_date(frame, "prediction", "label")
    assert np.allclose(ic, 1.0)
    assert summarize_ic(ic).mean == 1.0


def test_newey_west_t_is_finite_for_nonconstant_series() -> None:
    values = np.linspace(-0.02, 0.05, 100) + np.sin(np.arange(100)) * 0.01
    assert np.isfinite(newey_west_t(values))


def test_vectorized_rank_ic_matches_pairwise_spearman_with_ties_and_missing() -> None:
    dates = pd.to_datetime(
        ["2024-01-02"] * 6 + ["2024-01-03"] * 5 + ["2024-01-04"] * 2
    )
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "prediction": [1, 1, 2, 3, np.nan, 5, 5, 4, 4, 2, 1, 1, 2],
            "label": [2, 2, 1, 4, 5, np.nan, 1, 2, 2, np.nan, 5, 3, 4],
        }
    )
    expected = []
    for _, group in frame.groupby("trade_date", sort=True):
        valid = group["prediction"].notna() & group["label"].notna()
        if int(valid.sum()) < 3:
            expected.append(np.nan)
        else:
            expected.append(
                group.loc[valid, "prediction"]
                .rank(method="average")
                .corr(group.loc[valid, "label"].rank(method="average"))
            )

    actual = rank_ic_by_date(frame, "prediction", "label")
    assert actual.index.name == "trade_date"
    assert np.allclose(actual.to_numpy(), expected, equal_nan=True)


def test_long_oos_rank_ic_vectorized_path_preserves_daily_values() -> None:
    dates = pd.bdate_range("2010-01-04", periods=1_024)
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(dates.to_numpy(), 3),
            "prediction": np.tile([1.0, 2.0, 3.0], len(dates)),
            "label": np.tile([10.0, 20.0, 30.0], len(dates)),
        }
    )
    actual = rank_ic_by_date(frame, "prediction", "label")
    assert len(actual) == len(dates)
    assert np.allclose(actual, 1.0)
