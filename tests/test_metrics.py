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
