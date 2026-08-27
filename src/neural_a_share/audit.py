from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import summarize_ic


VALIDATION_SHORT_SESSIONS = 21
VALIDATION_LONG_SESSIONS = 63
VALIDATION_BLOCKS = 6
VALIDATION_ANCHORS = 9


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def prediction_fingerprint(
    frame: pd.DataFrame,
    *,
    model_version: str,
    top_k: int,
) -> str:
    """Return a stable fingerprint for the published neural ranking.

    The digest deliberately includes the model, candidate count and ranked
    values.  Re-running the same checkpoint on the same market close is
    expected to reproduce it; any model, membership or prediction change
    produces a different value.
    """

    columns = [
        "symbol",
        "trade_date",
        "Alpha20",
        "Alpha40",
        "Alpha60",
        "NeuralAlpha",
        "NeuralRank",
        "FeatureCoverage",
        "instrument_type",
    ]
    available = [column for column in columns if column in frame]
    order = [
        column
        for column in ("trade_date", "NeuralRank", "symbol")
        if column in frame
    ]
    ranked = frame.sort_values(order, kind="stable") if order else frame
    records: list[dict[str, Any]] = []
    for row in ranked[available].to_dict("records"):
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if key == "trade_date":
                normalized[key] = str(pd.Timestamp(value).date())
            elif key in {
                "Alpha20",
                "Alpha40",
                "Alpha60",
                "NeuralAlpha",
                "FeatureCoverage",
            }:
                normalized[key] = _number(value)
            elif key == "NeuralRank":
                normalized[key] = int(value)
            else:
                normalized[key] = None if pd.isna(value) else str(value)
        records.append(normalized)
    payload = {
        "model_version": str(model_version),
        "prediction_rows": int(len(frame)),
        "top_k": int(top_k),
        "records": records,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _window_summary(values: pd.Series) -> dict[str, Any]:
    summary = summarize_ic(values)
    return {
        "mean": _number(summary.mean),
        "newey_west_t": _number(summary.newey_west_t),
        "observations": int(summary.observations),
    }


def _positive_share(values: Sequence[float | None]) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None], dtype="float64"
    )
    if not len(finite):
        return None
    return float(np.mean(finite > 0.0))


def validation_stability_audit(
    daily_ic: Mapping[int, pd.Series],
    *,
    short_sessions: int = VALIDATION_SHORT_SESSIONS,
    long_sessions: int = VALIDATION_LONG_SESSIONS,
    blocks: int = VALIDATION_BLOCKS,
    anchors: int = VALIDATION_ANCHORS,
) -> dict[str, Any]:
    """Describe validation stability without changing training or split dates.

    Long/short windows, chronological blocks and rolling anchor windows are
    diagnostics over the one fixed validation set.  They are never additional
    Test folds and never feed model selection automatically.
    """

    if min(short_sessions, long_sessions, blocks, anchors) <= 0:
        raise ValueError("validation audit settings must be positive")
    output: dict[str, Any] = {
        "scope": "VALIDATION_ONLY",
        "selection_policy": "DIAGNOSTIC_NOT_AUTOMATIC_TUNING",
        "short_sessions": int(short_sessions),
        "long_sessions": int(long_sessions),
        "chronological_blocks": int(blocks),
        "rolling_anchors": int(anchors),
        "horizons": {},
    }
    for horizon, raw in daily_ic.items():
        values = raw.sort_index().astype("float64")
        date_blocks = [
            part
            for part in np.array_split(np.arange(len(values)), blocks)
            if len(part)
        ]
        block_rows: list[dict[str, Any]] = []
        for part in date_blocks:
            window = values.iloc[part]
            block_rows.append(
                {
                    "start": str(pd.Timestamp(window.index[0]).date()),
                    "end": str(pd.Timestamp(window.index[-1]).date()),
                    **_window_summary(window),
                }
            )

        minimum_anchor = min(int(short_sessions), len(values))
        anchor_rows: list[dict[str, Any]] = []
        if minimum_anchor:
            endpoints = np.linspace(
                minimum_anchor - 1,
                len(values) - 1,
                num=min(int(anchors), len(values) - minimum_anchor + 1),
                dtype=int,
            )
            for endpoint in np.unique(endpoints):
                start = max(0, int(endpoint) - int(short_sessions) + 1)
                window = values.iloc[start : int(endpoint) + 1]
                anchor_rows.append(
                    {
                        "start": str(pd.Timestamp(window.index[0]).date()),
                        "end": str(pd.Timestamp(window.index[-1]).date()),
                        **_window_summary(window),
                    }
                )

        block_means = [row["mean"] for row in block_rows]
        anchor_means = [row["mean"] for row in anchor_rows]
        output["horizons"][str(int(horizon))] = {
            "full": _window_summary(values),
            f"recent_{int(short_sessions)}": _window_summary(
                values.tail(int(short_sessions))
            ),
            f"recent_{int(long_sessions)}": _window_summary(
                values.tail(int(long_sessions))
            ),
            "block_positive_share": _positive_share(block_means),
            "anchor_positive_share": _positive_share(anchor_means),
            "blocks": block_rows,
            "anchors": anchor_rows,
        }
    return output
