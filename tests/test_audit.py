from __future__ import annotations

import numpy as np
import pandas as pd

from neural_a_share.audit import (
    prediction_fingerprint,
    validation_stability_audit,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A.SH", "B.SZ", "C.BJ"],
            "trade_date": pd.Timestamp("2026-08-27"),
            "Alpha20": np.asarray([0.03, 0.02, 0.01], dtype="float32"),
            "Alpha40": np.asarray([0.04, 0.03, 0.02], dtype="float32"),
            "Alpha60": np.asarray([0.05, 0.04, 0.03], dtype="float32"),
            "NeuralAlpha": np.asarray([0.04, 0.03, 0.02], dtype="float32"),
            "NeuralRank": [1, 2, 3],
            "FeatureCoverage": np.asarray([1.0, 1.0, 0.9], dtype="float32"),
            "instrument_type": "stock",
        }
    )


def test_prediction_fingerprint_is_stable_but_detects_rank_changes() -> None:
    frame = _predictions()
    expected = prediction_fingerprint(frame, model_version="mlp-v1", top_k=3)
    shuffled = frame.sample(frac=1.0, random_state=5)
    assert (
        prediction_fingerprint(shuffled, model_version="mlp-v1", top_k=3)
        == expected
    )
    changed = frame.copy()
    changed.loc[0, "NeuralAlpha"] += 0.001
    assert (
        prediction_fingerprint(changed, model_version="mlp-v1", top_k=3)
        != expected
    )
    assert (
        prediction_fingerprint(frame, model_version="mlp-v2", top_k=3)
        != expected
    )


def test_validation_stability_keeps_fixed_dates_and_records_six_blocks_nine_anchors() -> None:
    dates = pd.bdate_range("2025-11-24", periods=126)
    values = pd.Series(np.linspace(-0.02, 0.08, len(dates)), index=dates)
    audit = validation_stability_audit({20: values, 40: values * 0.5, 60: values * 0.25})

    assert audit["scope"] == "VALIDATION_ONLY"
    assert audit["selection_policy"] == "DIAGNOSTIC_NOT_AUTOMATIC_TUNING"
    for horizon in ("20", "40", "60"):
        result = audit["horizons"][horizon]
        assert result["full"]["observations"] == 126
        assert result["recent_21"]["observations"] == 21
        assert result["recent_63"]["observations"] == 63
        assert len(result["blocks"]) == 6
        assert len(result["anchors"]) == 9
        assert result["blocks"][0]["start"] == str(dates[0].date())
        assert result["blocks"][-1]["end"] == str(dates[-1].date())
