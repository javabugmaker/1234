from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from neural_a_share.data.store import ParquetStore
from neural_a_share.features import FEATURE_NAMES
from neural_a_share.research import (
    PartitionedResearchLoader,
    ResearchSplitSpec,
)


def _write_partition(
    store: ParquetStore,
    year: int,
    days: int = 30,
    symbols: int = 120,
) -> None:
    dates = pd.bdate_range(f"{year}-01-02", periods=days)
    names = [f"{value:06d}.SZ" for value in range(symbols)]
    index = pd.MultiIndex.from_product(
        [dates, names], names=["trade_date", "symbol"]
    ).to_frame(index=False)
    index = index[["symbol", "trade_date"]]
    row = np.arange(len(index), dtype="float64")
    # Existing caches may contain float64 and must be narrowed on read.
    feature_values = {
        name: (row % (offset + 17)) / float(offset + 17)
        for offset, name in enumerate(FEATURE_NAMES)
    }
    features = pd.concat([index, pd.DataFrame(feature_values)], axis=1)
    labels = index.copy()
    for horizon in (20, 40, 60):
        labels[f"label_{horizon}"] = ((row + horizon) % 101 - 50) / 10_000.0
        labels[f"label_available_date_{horizon}"] = labels["trade_date"] + pd.offsets.BDay(
            horizon
        )
    # Exercise the legacy dropna-before-sample semantics.
    unavailable = (row.astype("int64") % 97) == 0
    labels.loc[unavailable, ["label_20", "label_40", "label_60"]] = np.nan
    store.write_derived_year("features", year, features)
    store.write_derived_year("labels", year, labels)


def test_large_partitioned_training_preparation_never_materializes_all_years(
    tmp_path, monkeypatch
) -> None:
    store = ParquetStore(tmp_path / "cache")
    years = (2020, 2021, 2022, 2023)
    for year in years:
        _write_partition(store, year)

    def forbidden_all_year_read(*args, **kwargs):
        raise AssertionError("training must not call read_derived_years")

    monkeypatch.setattr(store, "read_derived_years", forbidden_all_year_read)
    original_read_year = store.read_derived_year
    reads: list[tuple[str, int, tuple[str, ...] | None]] = []

    def tracked_read_year(name, year, columns=None, float32_columns=()):
        reads.append((name, int(year), tuple(columns) if columns is not None else None))
        return original_read_year(name, year, columns, float32_columns)

    monkeypatch.setattr(store, "read_derived_year", tracked_read_year)
    loader = PartitionedResearchLoader(store, FEATURE_NAMES, (20, 40, 60))
    cutoff = pd.Timestamp("2025-12-31")
    index = loader.scan(mature_cutoff=cutoff)
    train_dates = index.dates[:80]
    validation_dates = index.dates[-30:]
    prepared = loader.load_splits(
        index,
        {
            "train": ResearchSplitSpec(train_dates, 2_500, 1234),
            "validation": ResearchSplitSpec(
                validation_dates, 1_000, 1235, min_rows_per_date=3
            ),
        },
        mature_cutoff=cutoff,
    )

    assert len(prepared["train"]) == 2_500
    assert len(prepared["validation"]) == 1_000
    assert set(prepared["validation"]["trade_date"]) == set(validation_dates)
    assert prepared["validation"].groupby("trade_date").size().ge(3).all()
    for frame in prepared.values():
        assert all(frame[name].dtype == np.dtype("float32") for name in FEATURE_NAMES)
        assert all(frame[f"label_{h}"].dtype == np.dtype("float32") for h in (20, 40, 60))

    # Every access is one named annual partition with an explicit column list.
    assert all(columns is not None for _, _, columns in reads)
    assert set(year for _, year, _ in reads) == set(years)
    assert Counter(name for name, _, _ in reads)["features"] == 2 * len(years)
    assert Counter(name for name, _, _ in reads)["labels"] == 2 * len(years)
    scan_feature_reads = [
        columns for name, _, columns in reads[: 2 * len(years)] if name == "features"
    ]
    assert all(columns == ("symbol", "trade_date") for columns in scan_feature_reads)

    # The early global sample preserves the old DataFrame.sample row semantics.
    legacy_parts = []
    for year in years:
        features = original_read_year(
            "features", year, ["symbol", "trade_date", *FEATURE_NAMES]
        )
        labels = original_read_year(
            "labels", year, ["symbol", "trade_date", "label_20", "label_40", "label_60"]
        )
        legacy_parts.append(
            features.merge(
                labels,
                on=["symbol", "trade_date"],
                how="inner",
                validate="one_to_one",
            )
        )
    legacy = pd.concat(legacy_parts, ignore_index=True).sort_values(
        ["trade_date", "symbol"]
    )
    legacy = legacy[legacy["trade_date"].isin(train_dates)].dropna(
        subset=["label_20", "label_40", "label_60"], how="all"
    )
    expected = legacy.sample(n=2_500, random_state=1234).sort_values(
        ["trade_date", "symbol"]
    )
    assert list(
        prepared["train"][["trade_date", "symbol"]].itertuples(index=False, name=None)
    ) == list(expected[["trade_date", "symbol"]].itertuples(index=False, name=None))


def test_partition_one_to_one_validation_rejects_duplicate_keys(tmp_path) -> None:
    store = ParquetStore(tmp_path / "cache")
    _write_partition(store, 2024, days=3, symbols=4)
    path = store.derived_dir / "features" / "year=2024" / "features.parquet"
    features = pd.read_parquet(path)
    store.write_derived_year(
        "features", 2024, pd.concat([features, features.iloc[[0]]], ignore_index=True)
    )
    loader = PartitionedResearchLoader(store, FEATURE_NAMES, (20, 40, 60))
    with pytest.raises(ValueError, match="one-to-one"):
        loader.scan(mature_cutoff=pd.Timestamp("2025-12-31"))


def test_merged_annual_research_cache_avoids_repeated_feature_label_merge(
    tmp_path, monkeypatch
) -> None:
    store = ParquetStore(tmp_path / "cache")
    _write_partition(store, 2024, days=12, symbols=25)
    loader = PartitionedResearchLoader(store, FEATURE_NAMES, (20, 40, 60))
    labels = ("label_20", "label_40", "label_60")

    first = loader.read_partition(2024, FEATURE_NAMES, labels)
    cache_path = (
        store.derived_dir
        / "research_cache"
        / "year=2024"
        / "research.parquet"
    )
    assert cache_path.exists()

    def forbidden_source_read(*args, **kwargs):
        raise AssertionError("a valid merged annual cache must bypass both source files")

    monkeypatch.setattr(store, "read_derived_year", forbidden_source_read)
    second = loader.read_partition(2024, FEATURE_NAMES, labels)

    pd.testing.assert_frame_equal(first, second)
    assert all(second[name].dtype == np.dtype("float32") for name in FEATURE_NAMES)
    assert all(second[name].dtype == np.dtype("float32") for name in labels)
