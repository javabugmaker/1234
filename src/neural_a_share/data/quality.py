from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    count: int = 0


@dataclass
class DataQualityReport:
    status: str
    latest_date: pd.Timestamp | None
    rows: int
    symbols: int
    calendar_days: int
    latest_coverage: float
    issues: list[QualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["latest_date"] = self.latest_date.isoformat() if self.latest_date else None
        return payload


REQUIRED_BAR_COLUMNS = {
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
}


def check_bars(
    bars: pd.DataFrame,
    benchmark: str = "000300.SH",
    minimum_latest_coverage: float = 0.75,
) -> DataQualityReport:
    issues: list[QualityIssue] = []
    if bars.empty:
        return DataQualityReport(
            "FAIL", None, 0, 0, 0, 0.0, [QualityIssue("EMPTY", "ERROR", "bar cache is empty")]
        )
    missing = REQUIRED_BAR_COLUMNS - set(bars)
    if missing:
        return DataQualityReport(
            "FAIL",
            None,
            len(bars),
            bars.get("symbol", pd.Series(dtype=str)).nunique(),
            0,
            0.0,
            [QualityIssue("SCHEMA", "ERROR", f"missing columns: {sorted(missing)}", len(missing))],
        )
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    duplicate_count = int(frame.duplicated(["symbol", "trade_date"]).sum())
    if duplicate_count:
        issues.append(QualityIssue("DUPLICATE", "ERROR", "duplicate symbol/date bars", duplicate_count))
    price_cols = ["open", "high", "low", "close"]
    non_positive = int((frame[price_cols] <= 0).any(axis=1).sum())
    if non_positive:
        issues.append(QualityIssue("NON_POSITIVE_PRICE", "ERROR", "non-positive OHLC", non_positive))
    invalid_ohlc = int(
        (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    if invalid_ohlc:
        issues.append(QualityIssue("OHLC", "ERROR", "OHLC invariant violated", invalid_ohlc))
    negative_flow = int(((frame["volume"] < 0) | (frame["amount"] < 0)).sum())
    if negative_flow:
        issues.append(QualityIssue("NEGATIVE_FLOW", "ERROR", "negative volume or amount", negative_flow))
    latest = pd.Timestamp(frame["trade_date"].max())
    counts = frame.groupby("trade_date")["symbol"].nunique().sort_index()
    baseline = float(counts.tail(21).iloc[:-1].median()) if len(counts) > 1 else float(counts.iloc[-1])
    coverage = float(counts.iloc[-1] / baseline) if baseline > 0 else 0.0
    if coverage < minimum_latest_coverage:
        issues.append(
            QualityIssue(
                "INCOMPLETE_LATEST",
                "ERROR",
                f"latest cross-section coverage {coverage:.1%} below {minimum_latest_coverage:.1%}",
                int(counts.iloc[-1]),
            )
        )
    benchmark_dates = frame.loc[frame["symbol"].eq(benchmark), "trade_date"].sort_values().unique()
    if not len(benchmark_dates):
        issues.append(QualityIssue("NO_CALENDAR", "ERROR", f"benchmark {benchmark} calendar missing"))
    future = int((frame["trade_date"] > pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()).sum())
    if future:
        issues.append(QualityIssue("FUTURE_DATE", "ERROR", "future-dated bars", future))
    status = "FAIL" if any(issue.severity == "ERROR" for issue in issues) else ("WARN" if issues else "PASS")
    return DataQualityReport(
        status=status,
        latest_date=latest,
        rows=len(frame),
        symbols=int(frame["symbol"].nunique()),
        calendar_days=int(len(benchmark_dates)),
        latest_coverage=coverage,
        issues=issues,
    )


def assert_complete(report: DataQualityReport) -> None:
    if report.status == "FAIL":
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
        raise ValueError(f"TickFlow data quality check failed: {details}")
