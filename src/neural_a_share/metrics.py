from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    return float(left[valid].rank(method="average").corr(right[valid].rank(method="average")))


def rank_ic_by_date(
    frame: pd.DataFrame,
    prediction_col: str,
    label_col: str,
    date_col: str = "trade_date",
) -> pd.Series:
    return frame.groupby(date_col, sort=True).apply(
        lambda group: _spearman(group[prediction_col], group[label_col]),
        include_groups=False,
    ).rename(f"rank_ic_{label_col}")


def newey_west_t(values: Iterable[float], lags: int | None = None) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n < 3:
        return float("nan")
    centered = array - array.mean()
    if lags is None:
        lags = max(1, int(4 * (n / 100) ** (2 / 9)))
    lags = min(int(lags), n - 1)
    long_run = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    variance_mean = max(long_run / n, 0.0)
    return float(array.mean() / np.sqrt(variance_mean)) if variance_mean > 0 else float("nan")


@dataclass(frozen=True)
class ICSummary:
    mean: float
    standard_deviation: float
    icir: float
    newey_west_t: float
    observations: int


def summarize_ic(values: pd.Series, annualization: int = 252) -> ICSummary:
    clean = values.dropna().astype(float)
    mean = float(clean.mean()) if len(clean) else float("nan")
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    icir = mean / std * np.sqrt(annualization) if std and np.isfinite(std) else float("nan")
    return ICSummary(mean, std, float(icir), newey_west_t(clean), len(clean))


def rolling_rank_ic(
    frame: pd.DataFrame,
    prediction_col: str,
    label_col: str,
    window: int = 63,
) -> pd.DataFrame:
    daily = rank_ic_by_date(frame, prediction_col, label_col)
    return pd.DataFrame(
        {
            "trade_date": daily.index,
            "rank_ic": daily.values,
            "rolling_rank_ic": daily.rolling(window, min_periods=max(10, window // 3)).mean().values,
        }
    )


def ic_decay(
    frame: pd.DataFrame,
    prediction_col: str,
    label_columns: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for label in label_columns:
        daily = rank_ic_by_date(frame, prediction_col, label)
        summary = summarize_ic(daily)
        rows.append({"label": label, **summary.__dict__})
    return pd.DataFrame(rows)


def quantile_monotonicity(
    frame: pd.DataFrame,
    prediction_col: str,
    return_col: str,
    quantiles: int = 5,
) -> pd.DataFrame:
    work = frame[["trade_date", prediction_col, return_col]].dropna().copy()
    work["quantile"] = work.groupby("trade_date", sort=False)[prediction_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), quantiles, labels=False, duplicates="drop") + 1
    )
    return work.groupby("quantile")[return_col].agg(["mean", "count"]).reset_index()
