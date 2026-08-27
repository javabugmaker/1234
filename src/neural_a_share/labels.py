from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data.pit import reconstruct_pit_prices


@dataclass(frozen=True)
class LabelResult:
    frame: pd.DataFrame
    horizons: tuple[int, ...]


def _calendar_mapping(calendar: pd.DatetimeIndex, offset: int) -> pd.DataFrame:
    if offset < 1:
        raise ValueError("forward label offset must be positive")
    source = calendar[:-offset]
    target = calendar[offset:]
    return pd.DataFrame({"trade_date": source, f"date_plus_{offset}": target})


def make_labels(
    bars: pd.DataFrame,
    horizons: Iterable[int] = (20, 40, 60),
    benchmark: str = "000300.SH",
    asof_date: str | pd.Timestamp | None = None,
    cross_sectional_standardize: bool = False,
) -> LabelResult:
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if benchmark not in set(bars["symbol"]):
        raise ValueError(f"benchmark {benchmark} is required to define the trading calendar")
    pit = reconstruct_pit_prices(bars)
    pit["trade_date"] = pd.to_datetime(pit["trade_date"]).dt.normalize()
    calendar = pd.DatetimeIndex(
        pit.loc[pit["symbol"].eq(benchmark), "trade_date"].drop_duplicates().sort_values()
    )
    base = pit[["symbol", "trade_date"]].copy()
    for horizon in horizons:
        entry_date_col = f"entry_date_{horizon}"
        entry_map = _calendar_mapping(calendar, 1).rename(columns={"date_plus_1": entry_date_col})
        exit_map = _calendar_mapping(calendar, horizon).rename(
            columns={f"date_plus_{horizon}": f"label_available_date_{horizon}"}
        )
        dates = entry_map.merge(exit_map, on="trade_date", how="inner")
        base = base.merge(dates, on="trade_date", how="left")
        entry = pit[["symbol", "trade_date", "pit_open"]].rename(
            columns={"trade_date": entry_date_col, "pit_open": f"entry_index_{horizon}"}
        )
        exit_ = pit[["symbol", "trade_date", "pit_close"]].rename(
            columns={
                "trade_date": f"label_available_date_{horizon}",
                "pit_close": f"exit_index_{horizon}",
            }
        )
        base = base.merge(entry, on=["symbol", entry_date_col], how="left")
        base = base.merge(exit_, on=["symbol", f"label_available_date_{horizon}"], how="left")
        raw_name = f"raw_return_{horizon}"
        base[raw_name] = base[f"exit_index_{horizon}"].div(base[f"entry_index_{horizon}"]).sub(1.0)
        benchmark_return = (
            base.loc[base["symbol"].eq(benchmark), ["trade_date", raw_name]]
            .drop_duplicates("trade_date")
            .rename(columns={raw_name: f"benchmark_return_{horizon}"})
        )
        base = base.merge(benchmark_return, on="trade_date", how="left")
        label_name = f"label_{horizon}"
        base[label_name] = base[raw_name] - base[f"benchmark_return_{horizon}"]
        if cross_sectional_standardize:
            grouped = base.groupby("trade_date", sort=False)[label_name]
            mean = grouped.transform("mean")
            std = grouped.transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
            base[label_name] = (base[label_name] - mean) / std
        if asof_date is not None:
            mature = pd.to_datetime(base[f"label_available_date_{horizon}"]) <= pd.Timestamp(asof_date)
            base.loc[~mature, label_name] = np.nan
    keep = ["symbol", "trade_date"]
    for horizon in horizons:
        keep += [
            f"label_{horizon}",
            f"raw_return_{horizon}",
            f"benchmark_return_{horizon}",
            f"label_available_date_{horizon}",
        ]
    return LabelResult(base[keep].sort_values(["trade_date", "symbol"]).reset_index(drop=True), horizons)


def mature_labels_only(
    labels: pd.DataFrame, cutoff: str | pd.Timestamp, horizons: Iterable[int]
) -> pd.DataFrame:
    out = labels.copy()
    cutoff_ts = pd.Timestamp(cutoff)
    mask = pd.Series(True, index=out.index)
    for horizon in horizons:
        mask &= pd.to_datetime(out[f"label_available_date_{int(horizon)}"]) <= cutoff_ts
    return out[mask].copy()
