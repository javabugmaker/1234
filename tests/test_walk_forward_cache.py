from __future__ import annotations

import pandas as pd

from neural_a_share.walk_forward_cache import WalkForwardFoldCache


def _fold_predictions(fold_id: int, date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [f"{fold_id}A.SH", f"{fold_id}B.SZ"],
            "trade_date": pd.to_datetime([date, date]),
            "Alpha20": pd.Series([0.01, 0.02], dtype="float32"),
            "Alpha40": pd.Series([0.02, 0.03], dtype="float32"),
            "Alpha60": pd.Series([0.03, 0.04], dtype="float32"),
            "NeuralAlpha": pd.Series([0.02, 0.03], dtype="float32"),
            "NeuralRank": [2, 1],
            "fold_id": fold_id,
            "sample_zone": "HISTORICAL_OOS",
        }
    )


def test_fold_cache_resumes_only_an_identical_fold_signature(tmp_path) -> None:
    cache = WalkForwardFoldCache(tmp_path / "cache")
    frame = _fold_predictions(7, "2024-01-02")
    cache.save(7, "same-inputs", frame, {"fold_id": 7, "test_start": "2024-01-02"})

    assert cache.has(7, "same-inputs")
    assert not cache.has(7, "changed-inputs")
    assert cache.fold_row(7, "same-inputs") == {
        "fold_id": 7,
        "test_start": "2024-01-02",
    }
    loaded = cache.load(7, "same-inputs")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.predictions, frame)


def test_fold_cache_streams_publish_without_materializing_all_folds(tmp_path) -> None:
    cache = WalkForwardFoldCache(tmp_path / "cache")
    first = _fold_predictions(0, "2024-01-02")
    second = _fold_predictions(1, "2024-04-01")
    cache.save(0, "sig-0", first, {"fold_id": 0})
    cache.save(1, "sig-1", second, {"fold_id": 1})

    destination = tmp_path / "walk_forward_predictions.parquet"
    rows = cache.publish(
        [(0, "sig-0"), (1, "sig-1")], destination, coverage_status="PARTIAL"
    )
    published = pd.read_parquet(destination)

    assert rows == len(first) + len(second)
    assert "fold" not in published.columns
    assert published["coverage_status"].eq("PARTIAL").all()
    assert published.groupby("fold_id").size().to_dict() == {0: 2, 1: 2}
