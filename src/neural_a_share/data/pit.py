from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class PITViolation(ValueError):
    pass


def filter_available_asof(
    frame: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    information_date_col: str = "information_date",
) -> pd.DataFrame:
    if information_date_col not in frame:
        raise PITViolation(f"missing {information_date_col}")
    result = frame.copy()
    result[information_date_col] = pd.to_datetime(result[information_date_col])
    cutoff_ts = pd.Timestamp(cutoff)
    return result[result[information_date_col] <= cutoff_ts].copy()


def assert_no_future_information(
    frame: pd.DataFrame,
    signal_col: str = "trade_date",
    information_col: str = "feature_information_date",
) -> None:
    if signal_col not in frame or information_col not in frame:
        raise PITViolation(f"required PIT columns not found: {signal_col}, {information_col}")
    signal = pd.to_datetime(frame[signal_col])
    information = pd.to_datetime(frame[information_col])
    bad = frame[information > signal]
    if not bad.empty:
        sample = bad[[signal_col, information_col]].head(3).to_dict("records")
        raise PITViolation(f"future information found in features: {sample}")


def reconstruct_pit_prices(bars: pd.DataFrame) -> pd.DataFrame:
    """Build a split/dividend-safe synthetic price chain using only each day's
    raw OHLC and provider-supplied previous close.

    This avoids today's forward-adjusted series, which may encode corporate
    actions that were unknown at an earlier research date.
    """
    required = {"symbol", "trade_date", "open", "high", "low", "close", "prev_close"}
    missing = required - set(bars)
    if missing:
        raise PITViolation(f"cannot reconstruct PIT prices; missing {sorted(missing)}")
    out = bars.sort_values(["symbol", "trade_date"]).copy()
    out["prev_close"] = out.groupby("symbol", sort=False)["prev_close"].transform(
        lambda s: s.ffill()
    )
    valid_prev = out["prev_close"].where(out["prev_close"] > 0)
    out["daily_total_return"] = out["close"].div(valid_prev).sub(1.0)
    first = out.groupby("symbol", sort=False).cumcount().eq(0)
    out.loc[first & out["daily_total_return"].isna(), "daily_total_return"] = 0.0
    out["pit_close"] = (
        out["daily_total_return"].fillna(0.0).add(1.0).groupby(out["symbol"], sort=False).cumprod()
    )
    previous_index = out.groupby("symbol", sort=False)["pit_close"].shift(1).fillna(1.0)
    for field in ("open", "high", "low"):
        out[f"pit_{field}"] = previous_index * out[field].div(valid_prev)
    out["feature_information_date"] = pd.to_datetime(out["trade_date"])
    return out


@dataclass(frozen=True)
class SurvivorshipAudit:
    status: str
    first_snapshot: pd.Timestamp | None
    evaluated_start: pd.Timestamp | None
    detail: str


def audit_survivorship(
    snapshot_dates: list[pd.Timestamp], evaluated_start: str | pd.Timestamp | None
) -> SurvivorshipAudit:
    start = pd.Timestamp(evaluated_start).normalize() if evaluated_start is not None else None
    first = min(snapshot_dates).normalize() if snapshot_dates else None
    if first is None:
        return SurvivorshipAudit("FAIL", None, start, "no historical universe snapshots")
    if start is not None and start < first:
        return SurvivorshipAudit(
            "DEGRADED",
            first,
            start,
            "evaluation begins before the first observed TickFlow universe snapshot",
        )
    return SurvivorshipAudit("PASS", first, start, "membership was known at each signal date")


def apply_listing_date_guard(samples: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    meta = catalog[["symbol", "listing_date"]].drop_duplicates("symbol")
    merged = samples.merge(meta, on="symbol", how="left")
    merged["listing_date"] = pd.to_datetime(merged["listing_date"], errors="coerce")
    dates = pd.to_datetime(merged["trade_date"])
    return merged[merged["listing_date"].isna() | (merged["listing_date"] <= dates)].copy()
