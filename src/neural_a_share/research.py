from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data.store import ParquetStore

KEY_COLUMNS = ("symbol", "trade_date")
RowFilter = Callable[[pd.DataFrame], pd.DataFrame]
CacheProgress = Callable[[str], None]
RESEARCH_CACHE_VERSION = 1


@dataclass(frozen=True)
class ResearchIndex:
    """Lightweight, date-level inventory built without wide feature columns."""

    years: tuple[int, ...]
    dates: pd.DatetimeIndex
    rows_by_date: pd.Series
    labeled_rows_by_date: pd.Series


@dataclass(frozen=True)
class ResearchSplitSpec:
    dates: pd.DatetimeIndex
    limit: int | None
    seed: int
    require_any_label: bool = True
    include_labels: bool = True
    min_rows_per_date: int = 0


class PartitionedResearchLoader:
    """Prepare bounded research frames from existing annual parquet partitions.

    The loader never concatenates all feature or label years.  A narrow first
    pass inventories dates and row counts; a second pass reads, validates and
    merges one year at a time, retaining only rows selected for the final split.
    """

    def __init__(
        self,
        store: ParquetStore,
        feature_names: Sequence[str],
        horizons: Sequence[int],
        row_filter: RowFilter | None = None,
        cache_progress: CacheProgress | None = None,
    ) -> None:
        self.store = store
        self.feature_names = tuple(feature_names)
        self.horizons = tuple(int(value) for value in horizons)
        self.label_columns = tuple(f"label_{value}" for value in self.horizons)
        self.maturity_columns = tuple(
            f"label_available_date_{value}" for value in self.horizons
        )
        self.row_filter = row_filter
        self.cache_progress = cache_progress
        self.cache_dir = self.store.derived_dir / "research_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._validated_cache_signatures: set[tuple[int, str]] = set()

    def years(self) -> tuple[int, ...]:
        # An inner merge across the old monolithic frames implicitly ignored a
        # year missing on either side.  Intersecting partitions preserves that
        # behavior while making the boundary explicit.
        feature_years = set(self.store.derived_years("features"))
        label_years = set(self.store.derived_years("labels"))
        return tuple(sorted(feature_years & label_years))

    def read_partition(
        self,
        year: int,
        feature_columns: Sequence[str],
        label_columns: Sequence[str],
    ) -> pd.DataFrame:
        """Read and one-to-one merge a single year using only requested columns."""

        feature_columns = tuple(dict.fromkeys(feature_columns))
        label_columns = tuple(dict.fromkeys(label_columns))
        cached = self._read_cached_partition(year, feature_columns, label_columns)
        if cached is not None:
            return cached

        if feature_columns == self.feature_names:
            canonical_labels = tuple(
                dict.fromkeys((*self.label_columns, *self.maturity_columns))
            )
            if self.cache_progress is not None:
                self.cache_progress(f"Building merged research cache year={year}")
            merged = self._read_source_partition(
                year, self.feature_names, canonical_labels
            )
            if not merged.empty:
                self._write_cached_partition(year, merged)
            requested = [*KEY_COLUMNS, *feature_columns, *label_columns]
            return merged[requested].copy() if not merged.empty else merged[requested]

        return self._read_source_partition(year, feature_columns, label_columns)

    def _read_source_partition(
        self,
        year: int,
        feature_columns: Sequence[str],
        label_columns: Sequence[str],
    ) -> pd.DataFrame:
        feature_request = [*KEY_COLUMNS, *feature_columns]
        label_request = [*KEY_COLUMNS, *label_columns]
        features = self.store.read_derived_year(
            "features",
            year,
            columns=feature_request,
            float32_columns=feature_columns,
        )
        labels = self.store.read_derived_year(
            "labels",
            year,
            columns=label_request,
            float32_columns=[
                column for column in label_columns if column not in self.maturity_columns
            ],
        )
        if features.empty or labels.empty:
            return pd.DataFrame(columns=[*feature_request, *label_columns])
        for frame in (features, labels):
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        if features.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"features year={year} violates symbol + trade_date one-to-one")
        if labels.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"labels year={year} violates symbol + trade_date one-to-one")
        merged = features.merge(
            labels,
            on=list(KEY_COLUMNS),
            how="inner",
            sort=False,
            copy=False,
            validate="one_to_one",
        )
        # Arrow already narrowed these values.  The assignments are a defensive
        # guard for older parquet schemas and avoid a wide float64 consolidation.
        for column in (*feature_columns, *label_columns):
            if column in merged and column not in self.maturity_columns:
                merged[column] = merged[column].astype("float32", copy=False)
        return merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def _cache_paths(self, year: int) -> tuple[Path, Path]:
        directory = self.cache_dir / f"year={int(year)}"
        return directory / "research.parquet", directory / "manifest.json"

    def _source_signature(self, year: int) -> str:
        sources = []
        for name in ("features", "labels"):
            path = (
                self.store.derived_dir
                / name
                / f"year={int(year)}"
                / f"{name}.parquet"
            )
            stat = path.stat()
            sources.append(
                {
                    "name": name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        payload = {
            "version": RESEARCH_CACHE_VERSION,
            "year": int(year),
            "feature_names": self.feature_names,
            "label_columns": self.label_columns,
            "maturity_columns": self.maturity_columns,
            "sources": sources,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read_cached_partition(
        self,
        year: int,
        feature_columns: Sequence[str],
        label_columns: Sequence[str],
    ) -> pd.DataFrame | None:
        data_path, manifest_path = self._cache_paths(year)
        if not data_path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_signature = self._source_signature(year)
            if manifest.get("source_signature") != source_signature:
                return None
            requested = [*KEY_COLUMNS, *feature_columns, *label_columns]
            if not set(requested).issubset(set(manifest.get("columns", []))):
                return None
            numeric = [
                column
                for column in (*feature_columns, *label_columns)
                if column not in self.maturity_columns
            ]
            frame = self.store.read_parquet(
                data_path,
                columns=requested,
                float32_columns=numeric,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        validation_key = (int(year), source_signature)
        if not frame.empty:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
            if (
                validation_key not in self._validated_cache_signatures
                and frame.duplicated(list(KEY_COLUMNS)).any()
            ):
                raise ValueError(
                    f"research cache year={year} violates symbol + trade_date one-to-one"
                )
        self._validated_cache_signatures.add(validation_key)
        # Cache files are written in source merge order (trade_date, symbol).
        # Avoid re-sorting the same annual partition for every expanding fold.
        return frame.reset_index(drop=True)

    def _write_cached_partition(self, year: int, frame: pd.DataFrame) -> None:
        data_path, manifest_path = self._cache_paths(year)
        columns = [
            *KEY_COLUMNS,
            *self.feature_names,
            *self.label_columns,
            *self.maturity_columns,
        ]
        cached = frame if list(frame.columns) == columns else frame[columns]
        for column in (*self.feature_names, *self.label_columns):
            if cached[column].dtype != np.dtype("float32"):
                cached[column] = cached[column].astype("float32", copy=False)
        self.store._atomic_parquet(cached, data_path)
        self.store._atomic_json(
            {
                "version": RESEARCH_CACHE_VERSION,
                "source_signature": self._source_signature(year),
                "rows": len(cached),
                "columns": columns,
            },
            manifest_path,
        )

    def scan(self, mature_cutoff: pd.Timestamp | None = None) -> ResearchIndex:
        years = self.years()
        if not years:
            raise FileNotFoundError("feature/label cache missing; run feature build first")
        label_request = [*self.label_columns]
        if mature_cutoff is not None:
            label_request.extend(self.maturity_columns)
        all_counts: list[pd.Series] = []
        labeled_counts: list[pd.Series] = []
        for year in years:
            frame = self.read_partition(year, (), label_request)
            frame = self._filter_rows(frame, mature_cutoff)
            if self.row_filter is not None and not frame.empty:
                frame = self.row_filter(frame)
            if frame.empty:
                continue
            all_counts.append(frame.groupby("trade_date", sort=True).size())
            has_label = frame[list(self.label_columns)].notna().any(axis=1)
            labeled_counts.append(
                frame.loc[has_label].groupby("trade_date", sort=True).size()
            )
            del frame
            gc.collect()
        if not all_counts:
            empty = pd.Series(dtype="int64")
            return ResearchIndex(years, pd.DatetimeIndex([]), empty, empty.copy())
        rows_by_date = pd.concat(all_counts).groupby(level=0).sum().sort_index().astype("int64")
        if labeled_counts:
            labeled = pd.concat(labeled_counts).groupby(level=0).sum().sort_index()
            labeled = labeled.reindex(rows_by_date.index, fill_value=0).astype("int64")
        else:
            labeled = pd.Series(0, index=rows_by_date.index, dtype="int64")
        return ResearchIndex(
            years=years,
            dates=pd.DatetimeIndex(rows_by_date.index),
            rows_by_date=rows_by_date,
            labeled_rows_by_date=labeled,
        )

    def load_splits(
        self,
        index: ResearchIndex,
        specs: Mapping[str, ResearchSplitSpec],
        mature_cutoff: pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        if not specs:
            return {}
        normalized_specs = {
            name: ResearchSplitSpec(
                dates=pd.DatetimeIndex(pd.to_datetime(spec.dates)).normalize().sort_values(),
                limit=spec.limit,
                seed=spec.seed,
                require_any_label=spec.require_any_label,
                include_labels=spec.include_labels,
                min_rows_per_date=spec.min_rows_per_date,
            )
            for name, spec in specs.items()
        }
        positions: dict[str, np.ndarray | None] = {}
        expected: dict[str, int] = {}
        observed = {name: 0 for name in specs}
        pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in specs}
        for name, spec in normalized_specs.items():
            counts_source = (
                index.labeled_rows_by_date if spec.require_any_label else index.rows_by_date
            )
            counts = counts_source.reindex(spec.dates, fill_value=0).astype("int64")
            total = int(counts.sum())
            expected[name] = total
            positions[name] = _sample_positions(
                counts,
                limit=spec.limit,
                seed=spec.seed,
                min_rows_per_date=spec.min_rows_per_date,
            )

        needed_dates = pd.DatetimeIndex(
            sorted({date for spec in normalized_specs.values() for date in spec.dates})
        )
        needed_years = set(int(year) for year in needed_dates.year)
        label_request = [*self.label_columns]
        if mature_cutoff is not None:
            label_request.extend(self.maturity_columns)
        output_base = [*KEY_COLUMNS, *self.feature_names]
        for year in index.years:
            if year not in needed_years:
                continue
            frame = self.read_partition(year, self.feature_names, label_request)
            frame = self._filter_rows(frame, mature_cutoff)
            if self.row_filter is not None and not frame.empty:
                frame = self.row_filter(frame)
            if frame.empty:
                continue
            for name, spec in normalized_specs.items():
                chunk = frame[frame["trade_date"].isin(spec.dates)]
                if spec.require_any_label:
                    chunk = chunk.dropna(subset=list(self.label_columns), how="all")
                chunk_size = len(chunk)
                start = observed[name]
                stop = start + chunk_size
                selected = positions[name]
                if selected is None:
                    retained = chunk
                else:
                    left = int(np.searchsorted(selected, start, side="left"))
                    right = int(np.searchsorted(selected, stop, side="left"))
                    local = selected[left:right] - start
                    retained = chunk.iloc[local]
                if not retained.empty:
                    columns = [*output_base]
                    if spec.include_labels:
                        columns.extend(self.label_columns)
                    pieces[name].append(retained[columns].copy())
                observed[name] = stop
            del frame
            gc.collect()

        results: dict[str, pd.DataFrame] = {}
        for name, spec in normalized_specs.items():
            if observed[name] != expected[name]:
                raise RuntimeError(
                    f"derived partitions changed while preparing {name}: "
                    f"scanned {expected[name]:,}, loaded {observed[name]:,}"
                )
            columns = [*output_base]
            if spec.include_labels:
                columns.extend(self.label_columns)
            result = (
                pd.concat(pieces[name], ignore_index=True, copy=False)
                if pieces[name]
                else pd.DataFrame(columns=columns)
            )
            for column in (*self.feature_names, *self.label_columns):
                if column in result:
                    result[column] = result[column].astype("float32", copy=False)
            results[name] = result.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        return results

    def _filter_rows(
        self, frame: pd.DataFrame, mature_cutoff: pd.Timestamp | None
    ) -> pd.DataFrame:
        if frame.empty or mature_cutoff is None:
            return frame
        cutoff = pd.Timestamp(mature_cutoff).normalize()
        mature = pd.Series(True, index=frame.index)
        for column in self.maturity_columns:
            mature &= pd.to_datetime(frame[column]).le(cutoff)
        return frame.loc[mature]


def _sample_positions(
    counts: pd.Series,
    limit: int | None,
    seed: int,
    min_rows_per_date: int = 0,
) -> np.ndarray | None:
    """Return global row positions using the legacy uniform sample semantics.

    ``RandomState.choice`` is what ``DataFrame.sample(random_state=int)`` uses.
    Sorting the positions only changes loading order; callers restore the same
    date/symbol order as the old ``_downsample`` helper afterwards.
    """

    values = counts.to_numpy(dtype="int64", copy=False)
    total = int(values.sum())
    if limit is None or total <= int(limit):
        return None
    size = int(limit)
    if size <= 0:
        raise ValueError("row limit must be positive")
    selected = np.random.RandomState(int(seed)).choice(
        total, size=size, replace=False
    ).astype(np.int64, copy=False)
    if min_rows_per_date > 0:
        selected = _repair_date_coverage(
            selected, values, int(min_rows_per_date), int(seed)
        )
    return np.sort(selected)


def _repair_date_coverage(
    selected: np.ndarray,
    counts: np.ndarray,
    minimum: int,
    seed: int,
) -> np.ndarray:
    required = np.minimum(counts, int(minimum)).astype("int64", copy=False)
    if len(selected) < int(required.sum()):
        raise ValueError(
            "max_validation_rows is too small to retain the complete validation date structure"
        )
    cumulative = np.cumsum(counts)
    date_ids = np.searchsorted(cumulative, selected, side="right")
    selected_counts = np.bincount(date_ids, minlength=len(counts)).astype("int64")
    deficits = np.maximum(required - selected_counts, 0)
    if not deficits.any():
        return selected

    rng = np.random.RandomState(int(seed) ^ 0x5EED5EED)
    selected_set = set(int(value) for value in selected)
    additions: list[int] = []
    starts = np.concatenate(([0], cumulative[:-1]))
    for date_id in np.flatnonzero(deficits):
        candidates = np.arange(starts[date_id], cumulative[date_id], dtype="int64")
        rng.shuffle(candidates)
        needed = int(deficits[date_id])
        for candidate in candidates:
            value = int(candidate)
            if value not in selected_set:
                selected_set.add(value)
                additions.append(value)
                selected_counts[date_id] += 1
                needed -= 1
                if needed == 0:
                    break
        if needed:
            raise RuntimeError("unable to preserve validation date coverage")

    donor_order = selected.copy()
    rng.shuffle(donor_order)
    removals: list[int] = []
    for candidate in donor_order:
        date_id = int(np.searchsorted(cumulative, candidate, side="right"))
        if selected_counts[date_id] > required[date_id]:
            removals.append(int(candidate))
            selected_counts[date_id] -= 1
            if len(removals) == len(additions):
                break
    if len(removals) != len(additions):
        raise RuntimeError("unable to rebalance validation sample coverage")
    selected_set.difference_update(removals)
    return np.fromiter(selected_set, dtype="int64", count=len(selected_set))
