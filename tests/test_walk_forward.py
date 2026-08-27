from __future__ import annotations

import pandas as pd
import pytest

from neural_a_share.config import WalkForwardConfig
from neural_a_share.walk_forward import PurgedWalkForward


def test_expanding_purged_walk_forward_has_no_random_split() -> None:
    dates = pd.bdate_range("2015-01-01", periods=1500)
    config = WalkForwardConfig(
        initial_train_days=500,
        validation_days=100,
        test_days=60,
        step_days=60,
        purge_days=60,
        embargo_days=5,
    )
    folds = list(PurgedWalkForward(config, max_label_horizon=60).split(dates))
    assert len(folds) > 1
    assert len(folds[1].train_dates) > len(folds[0].train_dates)
    for fold in folds:
        assert fold.train_dates[-1] < fold.validation_dates[0] < fold.test_dates[0]
        assert not set(fold.train_dates) & set(fold.test_dates)
        assert dates.get_loc(fold.validation_dates[0]) - dates.get_loc(fold.train_dates[-1]) - 1 >= 60
        assert dates.get_loc(fold.test_dates[0]) - dates.get_loc(fold.validation_dates[-1]) - 1 >= 65


def test_purge_shorter_than_longest_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="purge_days"):
        PurgedWalkForward(WalkForwardConfig(purge_days=20), max_label_horizon=60)
