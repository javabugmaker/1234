from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import pandas as pd

from .config import WalkForwardConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purge_days: int
    embargo_days: int

    @property
    def training_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_dates[-1])

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_dates[0],
            "train_end": self.train_dates[-1],
            "validation_start": self.validation_dates[0],
            "validation_end": self.validation_dates[-1],
            "test_start": self.test_dates[0],
            "test_end": self.test_dates[-1],
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
        }


class PurgedWalkForward:
    def __init__(self, config: WalkForwardConfig, max_label_horizon: int = 60) -> None:
        self.config = config
        self.max_label_horizon = int(max_label_horizon)
        if config.purge_days < self.max_label_horizon:
            raise ValueError("purge_days must be at least the longest label horizon")
        if min(
            config.initial_train_days,
            config.validation_days,
            config.test_days,
            config.step_days,
        ) <= 0:
            raise ValueError("walk-forward window lengths must be positive")

    def split(self, dates: Iterable[pd.Timestamp]) -> Iterator[WalkForwardFold]:
        calendar = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().drop_duplicates().sort_values()
        train_end = self.config.initial_train_days - 1
        fold_id = 0
        while True:
            val_start = train_end + 1 + self.config.purge_days
            val_end = val_start + self.config.validation_days - 1
            test_start = val_end + 1 + self.config.purge_days + self.config.embargo_days
            test_end = test_start + self.config.test_days - 1
            if test_end >= len(calendar):
                break
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_dates=calendar[: train_end + 1],
                validation_dates=calendar[val_start : val_end + 1],
                test_dates=calendar[test_start : test_end + 1],
                purge_days=self.config.purge_days,
                embargo_days=self.config.embargo_days,
            )
            validate_fold(fold, self.max_label_horizon, calendar)
            yield fold
            fold_id += 1
            train_end += self.config.step_days


def validate_fold(
    fold: WalkForwardFold, max_label_horizon: int, calendar: pd.DatetimeIndex | None = None
) -> None:
    if len(set(fold.train_dates) & set(fold.validation_dates)):
        raise ValueError("train and validation overlap")
    if len(set(fold.validation_dates) & set(fold.test_dates)):
        raise ValueError("validation and test overlap")
    if not (fold.train_dates[-1] < fold.validation_dates[0] < fold.test_dates[0]):
        raise ValueError("walk-forward chronology violated")
    if fold.purge_days < max_label_horizon:
        raise ValueError("training labels may overlap the next evaluation window")
    if calendar is not None:
        positions = {date: i for i, date in enumerate(calendar)}
        train_gap = positions[fold.validation_dates[0]] - positions[fold.train_dates[-1]] - 1
        validation_gap = positions[fold.test_dates[0]] - positions[fold.validation_dates[-1]] - 1
        if train_gap < max_label_horizon:
            raise ValueError("train/validation purge is too short")
        if validation_gap < max_label_horizon + fold.embargo_days:
            raise ValueError("validation/test purge plus embargo is too short")
