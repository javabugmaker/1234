from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .audit import prediction_fingerprint, validation_stability_audit
from .backtest import BacktestResult, PortfolioBacktester, prepare_signal_targets
from .config import AppConfig
from .data.pit import audit_survivorship, reconstruct_pit_prices
from .data.quality import DataQualityReport, assert_complete, check_bars
from .data.store import ParquetStore
from .data.tickflow import TickFlowFreeClient
from .features import FEATURE_NAMES, build_features
from .labels import make_labels
from .metrics import quantile_monotonicity, rank_ic_by_date, summarize_ic
from .model import (
    CheckpointMetadata,
    FeatureNormalizer,
    ModelRegistry,
    MultiTaskMLP,
    NeuralTrainer,
    inference_frame,
    load_checkpoint,
    make_model_version,
    predict_array,
    save_checkpoint,
)
from .pages_publish import PagesPublishResult, publish_pages_to_git
from .reports import ReportContext, StaticReportPublisher, write_status_json
from .research import PartitionedResearchLoader, ResearchSplitSpec
from .universe import (
    STOCK_CLASSIFIER_VERSION,
    filter_catalog,
    filter_degraded_symbol_universe,
    filter_feature_coverage,
)
from .walk_forward import PurgedWalkForward, WalkForwardFold
from .walk_forward_cache import (
    WALK_FORWARD_CACHE_VERSION,
    WalkForwardFoldCache,
    WalkForwardRunResult,
    file_inventory,
    stable_digest,
)

LOGGER = logging.getLogger("neural_a_share")
ProgressCallback = Callable[[str, float | None], None]


def _no_progress(_: str, __: float | None = None) -> None:
    return None


def _downsample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    # Sampling occurs only inside an already time-separated train/validation
    # window. It never determines the chronological split.
    return frame.sample(n=int(limit), random_state=seed).sort_values(["trade_date", "symbol"])


class NeuralAlphaPipeline:
    def __init__(
        self,
        config: AppConfig,
        progress: ProgressCallback | None = None,
        tickflow_client: TickFlowFreeClient | None = None,
    ) -> None:
        self.config = config
        self.config.ensure_directories()
        self.store = ParquetStore(config.paths.cache_dir)
        self.registry = ModelRegistry(config.paths.models_dir)
        self.publisher = StaticReportPublisher(config.paths.docs_dir, config.reports.title)
        self.progress = progress or _no_progress
        self._tickflow_client = tickflow_client

    def emit(self, message: str, progress: float | None = None) -> None:
        LOGGER.info(message)
        self.progress(message, progress)

    def update_tickflow(self, full: bool = False) -> DataQualityReport:
        latest = self.store.latest_bar_date()
        start = self.config.tickflow.history_start if full or latest is None else latest - pd.Timedelta(days=14)
        self.emit(f"TickFlow.free() update from {pd.Timestamp(start).date()}", 0.02)
        client = self._tickflow_client or TickFlowFreeClient(
            self.config.tickflow, cache_dir=self.config.paths.cache_dir / "tickflow_sdk"
        )
        should_close = self._tickflow_client is None
        try:
            result = client.update(start_date=start)
        finally:
            if should_close:
                client.close()
        if result.bars.empty:
            raise RuntimeError("TickFlow.free() returned no daily bars")
        self.emit(f"Caching {len(result.bars):,} raw rows", 0.68)
        self.store.upsert_bars(result.bars)
        observed_date = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
        self.store.write_universe_snapshot(result.catalog, observed_date)
        bars = self.store.read_bars()
        quality = check_bars(bars, benchmark=self.config.tickflow.benchmark)
        self.store.write_manifest(
            "tickflow",
            {
                "service": "TickFlow.free()",
                "updated_at": pd.Timestamp.now(tz="Asia/Shanghai"),
                "latest_date": quality.latest_date,
                "rows": quality.rows,
                "symbols": quality.symbols,
                "quality": quality.to_dict(),
                "adjustment": "none",
            },
        )
        assert_complete(quality)
        self.emit(f"TickFlow PASS · latest {quality.latest_date.date()}", 1.0)
        return quality

    def build_derived(self, years: Iterable[int] | None = None) -> dict[str, int]:
        available_years = self.store.bar_years()
        if not available_years:
            raise FileNotFoundError("raw bar cache is empty; run update first")
        targets = [int(year) for year in years] if years is not None else available_years
        rows = {"features": 0, "labels": 0}
        for number, year in enumerate(targets, start=1):
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31)
            context = self.store.read_bars(
                start_date=start - pd.Timedelta(days=400),
                end_date=end + pd.Timedelta(days=120),
            )
            # Keep the established feature definition and the existing
            # checkpoint's input distribution intact. Asset eligibility is
            # applied to research rows and inference candidates after causal
            # features have been built, never as an extra ranking score.
            feature_context = context[pd.to_datetime(context["trade_date"]) <= end]
            self.emit(f"Features {year} ({number}/{len(targets)})", (number - 1) / max(len(targets), 1))
            feature_result = build_features(feature_context, self.config.features)
            features = feature_result.frame[pd.to_datetime(feature_result.frame["trade_date"]).dt.year.eq(year)]
            self.store.write_derived_year("features", year, features)
            label_result = make_labels(
                context,
                horizons=self.config.labels.horizons,
                benchmark=self.config.labels.benchmark,
                cross_sectional_standardize=self.config.labels.cross_sectional_standardize,
            )
            labels = label_result.frame[pd.to_datetime(label_result.frame["trade_date"]).dt.year.eq(year)]
            self.store.write_derived_year("labels", year, labels)
            rows["features"] += len(features)
            rows["labels"] += len(labels)
        self.store.write_manifest(
            "derived",
            {
                "feature_count": len(FEATURE_NAMES),
                "feature_names": FEATURE_NAMES,
                "years": targets,
                "rows": rows,
                "selection_instrument_types": list(
                    self.config.portfolio.selection_instrument_types
                ),
                "stock_classifier_version": STOCK_CLASSIFIER_VERSION,
                "built_at": pd.Timestamp.now(tz="Asia/Shanghai"),
            },
        )
        self.emit(f"Derived cache ready · {len(FEATURE_NAMES)} features", 1.0)
        return rows

    def _research_loader(self, strict_membership: bool) -> PartitionedResearchLoader:
        return PartitionedResearchLoader(
            self.store,
            FEATURE_NAMES,
            self.config.labels.horizons,
            row_filter=lambda frame: self._apply_observed_membership(
                frame, strict=strict_membership
            ),
            cache_progress=lambda message: self.emit(message, None),
        )

    def _apply_observed_membership(self, frame: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        allowed_types = self.config.portfolio.selection_instrument_types
        # The override is deliberately explicit. Applying today's catalog to
        # older rows would create survivorship bias. Degraded research therefore
        # uses only deterministic symbol-level asset classification and remains
        # marked DEGRADED; strict membership stays fail-closed.
        if not strict:
            return filter_degraded_symbol_universe(frame, allowed_types)
        snapshots = self.store.universe_snapshot_dates()
        audit = audit_survivorship(snapshots, frame["trade_date"].min() if not frame.empty else None)
        if audit.status != "PASS":
            raise ValueError(
                "strict survivorship audit failed: " + audit.detail + ". "
                "Do not backfill old membership from today's TickFlow catalog. "
                "For an explicitly DEGRADED research run, pass "
                "--allow-degraded-survivorship."
            )
        if not snapshots:
            return frame.iloc[0:0].copy()
        pieces = []
        ordered = sorted(snapshots)
        for index, snapshot_date in enumerate(ordered):
            end = ordered[index + 1] - pd.Timedelta(days=1) if index + 1 < len(ordered) else frame["trade_date"].max()
            observed_catalog = self.store.read_universe_asof(
                snapshot_date, strict=True
            )
            eligible_catalog = filter_catalog(observed_catalog, allowed_types)
            members = set(eligible_catalog["symbol"])
            eligible = frame[
                frame["trade_date"].between(snapshot_date, end) & frame["symbol"].isin(members)
            ]
            pieces.append(eligible)
        return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()

    def _fit_one(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        model_version: str,
        training_cutoff: pd.Timestamp,
        survivorship_status: str = "PASS",
        training_progress: Callable[[int, int, float, float], None] | None = None,
        *,
        data_cutoff: pd.Timestamp | None = None,
        train_period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
        validation_period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> tuple[MultiTaskMLP, FeatureNormalizer, CheckpointMetadata, Any]:
        label_columns = [f"label_{h}" for h in self.config.labels.horizons]
        train = train.dropna(subset=label_columns, how="all")
        validation = validation.dropna(subset=label_columns, how="all")
        validation_dates = set(pd.to_datetime(validation["trade_date"]).dt.normalize())
        train_before = len(train)
        validation_before = len(validation)
        train = filter_feature_coverage(
            train,
            FEATURE_NAMES,
            self.config.data.min_feature_coverage,
        )
        validation = filter_feature_coverage(
            validation,
            FEATURE_NAMES,
            self.config.data.min_feature_coverage,
        )
        retained_validation_dates = set(
            pd.to_datetime(validation["trade_date"]).dt.normalize()
        )
        if missing_dates := validation_dates - retained_validation_dates:
            raise ValueError(
                "feature completeness removed entire validation dates: "
                + ", ".join(str(pd.Timestamp(value).date()) for value in sorted(missing_dates)[:5])
            )
        self.emit(
            f"Feature coverage >= {self.config.data.min_feature_coverage:.0%} · "
            f"train {len(train):,}/{train_before:,} · "
            f"validation {len(validation):,}/{validation_before:,}",
            None,
        )
        if train.empty or validation.empty:
            raise ValueError("train or validation has no mature labels")
        train = _downsample(train, self.config.model.max_train_rows, self.config.model.seed)
        validation = _downsample(
            validation, self.config.model.max_validation_rows, self.config.model.seed + 1
        )
        train_values = train[FEATURE_NAMES].to_numpy(dtype="float32", copy=False)
        validation_values = validation[FEATURE_NAMES].to_numpy(dtype="float32", copy=False)
        normalizer = FeatureNormalizer().fit(train_values)
        train_x = normalizer.transform(train_values)
        validation_x = normalizer.transform(validation_values)
        train_y = train[label_columns].to_numpy(dtype="float32")
        validation_y = validation[label_columns].to_numpy(dtype="float32")
        del train_values, validation_values
        model = MultiTaskMLP(
            input_dim=len(FEATURE_NAMES),
            hidden_dims=self.config.model.hidden_dims,
            dropout=self.config.model.dropout,
            horizons=self.config.labels.horizons,
        )
        training_result = NeuralTrainer(self.config.model).fit(
            model,
            train_x,
            train_y,
            validation_x,
            validation_y,
            progress=training_progress,
        )
        del train_x, train_y, validation_y
        output = predict_array(model, validation_x, self.config.model.device)
        scored = validation[["trade_date", "symbol", *label_columns]].copy()
        metrics: dict[str, float] = {"validation_loss": training_result.best_validation_loss}
        daily_ic: dict[int, pd.Series] = {}
        for index, horizon in enumerate(self.config.labels.horizons):
            column = f"Alpha{horizon}"
            scored[column] = output[:, index]
            daily = rank_ic_by_date(scored, column, f"label_{horizon}")
            daily_ic[int(horizon)] = daily
            summary = summarize_ic(daily)
            metrics[f"rank_ic_{horizon}"] = summary.mean
            metrics[f"icir_{horizon}"] = summary.icir
            metrics[f"newey_west_t_{horizon}"] = summary.newey_west_t
        validation_audit = validation_stability_audit(daily_ic)
        actual_train_period = train_period or (
            pd.Timestamp(train["trade_date"].min()),
            pd.Timestamp(train["trade_date"].max()),
        )
        actual_validation_period = validation_period or (
            pd.Timestamp(validation["trade_date"].min()),
            pd.Timestamp(validation["trade_date"].max()),
        )
        metadata = CheckpointMetadata(
            model_version=model_version,
            training_cutoff=str(pd.Timestamp(training_cutoff).date()),
            feature_names=tuple(FEATURE_NAMES),
            horizons=tuple(self.config.labels.horizons),
            hidden_dims=tuple(self.config.model.hidden_dims),
            dropout=self.config.model.dropout,
            metrics=metrics,
            survivorship_status=survivorship_status,
            training_cutoff_semantics="last_train_signal_date",
            data_cutoff=(
                str(pd.Timestamp(data_cutoff).date())
                if data_cutoff is not None
                else None
            ),
            train_start=str(pd.Timestamp(actual_train_period[0]).date()),
            train_end=str(pd.Timestamp(actual_train_period[1]).date()),
            validation_start=str(
                pd.Timestamp(actual_validation_period[0]).date()
            ),
            validation_end=str(pd.Timestamp(actual_validation_period[1]).date()),
            training_seed=int(self.config.model.seed),
            validation_audit=validation_audit,
        )
        return model, normalizer, metadata, training_result

    def train(self, allow_degraded_survivorship: bool = False) -> CheckpointMetadata:
        latest = self.store.latest_bar_date()
        if latest is None:
            raise FileNotFoundError("no TickFlow data")
        strict_membership = (
            self.config.data.strict_survivorship and not allow_degraded_survivorship
        )
        survivorship_status = "PASS" if strict_membership else "DEGRADED"
        if survivorship_status == "DEGRADED":
            self.emit(
                "DEGRADED survivorship mode: preserving the cached historical bar universe; "
                "results are not strict Historical OOS",
                0.005,
            )
        loader = self._research_loader(strict_membership)
        self.emit("Scanning narrow annual research partitions", 0.01)
        index = loader.scan(mature_cutoff=latest)
        dates = index.dates
        validation_days = self.config.walk_forward.validation_days
        purge = self.config.walk_forward.purge_days
        if len(dates) < self.config.walk_forward.initial_train_days + validation_days + purge:
            raise ValueError("not enough mature PIT sessions for train/validation")
        validation_dates = dates[-validation_days:]
        train_end_position = len(dates) - validation_days - purge - 1
        train_dates = dates[: train_end_position + 1]
        prepared = loader.load_splits(
            index,
            {
                "train": ResearchSplitSpec(
                    train_dates,
                    self.config.model.max_train_rows,
                    self.config.model.seed,
                ),
                "validation": ResearchSplitSpec(
                    validation_dates,
                    self.config.model.max_validation_rows,
                    self.config.model.seed + 1,
                    min_rows_per_date=3,
                ),
            },
            mature_cutoff=latest,
        )
        train = prepared["train"]
        validation = prepared["validation"]
        training_cutoff = pd.Timestamp(train_dates[-1])
        version = make_model_version(training_cutoff, FEATURE_NAMES)
        if survivorship_status == "DEGRADED":
            version = f"{version}-degraded"
        self.emit(
            f"Training {version} on {len(train):,} rows; validation {len(validation):,}",
            0.05,
        )

        def training_progress(
            epoch: int,
            total: int,
            train_loss: float,
            validation_loss: float,
        ) -> None:
            if epoch == 1 or epoch % 5 == 0 or epoch == total:
                self.emit(
                    f"Epoch {epoch}/{total} · train {train_loss:.6f} · "
                    f"validation {validation_loss:.6f}",
                    0.05 + 0.90 * epoch / max(total, 1),
                )

        model, normalizer, metadata, result = self._fit_one(
            train,
            validation,
            version,
            training_cutoff,
            survivorship_status=survivorship_status,
            training_progress=training_progress,
            data_cutoff=latest,
            train_period=(train_dates[0], train_dates[-1]),
            validation_period=(validation_dates[0], validation_dates[-1]),
        )
        checkpoint = self.config.paths.models_dir / version / "checkpoint.pt"
        save_checkpoint(checkpoint, model, normalizer, metadata, result)
        role = "champion" if self.registry.read().get("champion") is None else "challenger"
        history = list(result.history)
        best_epoch = (
            int(min(history, key=lambda row: row["validation_loss"])["epoch"])
            if history
            else None
        )
        training_audit = {
            "model_version": version,
            "role": role,
            "training_cutoff": training_cutoff,
            "training_cutoff_semantics": "last_train_signal_date",
            "data_cutoff": latest,
            "label_maturity_cutoff": latest,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "validation_start": validation_dates[0],
            "validation_end": validation_dates[-1],
            "purge_days": purge,
            "epochs_trained": int(result.epochs_trained),
            "best_epoch": best_epoch,
            "best_validation_loss": float(result.best_validation_loss),
            "stopped_early": bool(result.stopped_early),
            "device": str(result.device),
            "amp_enabled": bool(result.amp_enabled),
            "training_seed": int(self.config.model.seed),
            "history": history,
            "metrics": metadata.metrics,
            "validation_audit": metadata.validation_audit,
            "survivorship_status": survivorship_status,
        }
        write_status_json(checkpoint.with_name("training_audit.json"), training_audit)
        self.registry.register(metadata, checkpoint, role=role)
        self.store.write_manifest(
            "training",
            {
                "model_version": version,
                "training_cutoff": training_cutoff,
                "training_cutoff_semantics": "last_train_signal_date",
                "data_cutoff": latest,
                "label_maturity_cutoff": latest,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "validation_start": validation_dates[0],
                "validation_end": validation_dates[-1],
                "purge_days": purge,
                "selection_instrument_types": list(
                    self.config.portfolio.selection_instrument_types
                ),
                "min_feature_coverage": self.config.data.min_feature_coverage,
                "survivorship_status": survivorship_status,
                "survivorship_detail": (
                    "membership was known at each signal date"
                    if strict_membership
                    else "explicit override kept the cached historical bar universe; "
                    "current catalog membership was not represented as historical PIT membership"
                ),
                "metrics": metadata.metrics,
                "validation_audit": metadata.validation_audit,
                "epochs_trained": int(result.epochs_trained),
                "best_epoch": best_epoch,
                "stopped_early": bool(result.stopped_early),
                "device": str(result.device),
                "amp_enabled": bool(result.amp_enabled),
                "role": role,
            },
        )
        self.emit(f"Model saved as {role}: {version}", 1.0)
        return metadata

    def _walk_forward_fold_signature(
        self,
        fold: WalkForwardFold,
        survivorship_status: str,
    ) -> str:
        dates = fold.train_dates.append(fold.validation_dates).append(fold.test_dates)
        years = sorted(set(int(year) for year in dates.year))
        source_files = []
        for name in ("features", "labels"):
            for year in years:
                path = (
                    self.store.derived_dir
                    / name
                    / f"year={year}"
                    / f"{name}.parquet"
                )
                if path.exists():
                    source_files.append(path)
        if survivorship_status == "PASS":
            for path in self.store.universe_dir.glob("asof=*.parquet"):
                snapshot_date = pd.Timestamp(path.stem.split("=", 1)[1])
                if snapshot_date <= fold.test_dates[-1]:
                    source_files.append(path)
        return stable_digest(
            {
                "cache_version": WALK_FORWARD_CACHE_VERSION,
                "fold": fold.as_dict(),
                "calendar": [int(value) for value in dates.asi8],
                "feature_names": FEATURE_NAMES,
                "horizons": self.config.labels.horizons,
                "model": asdict(self.config.model),
                "data": asdict(self.config.data),
                "selection_instrument_types": list(
                    self.config.portfolio.selection_instrument_types
                ),
                "stock_classifier_version": STOCK_CLASSIFIER_VERSION,
                "walk_forward": asdict(self.config.walk_forward),
                "survivorship_status": survivorship_status,
                "sources": file_inventory(
                    source_files, relative_to=self.config.paths.cache_dir
                ),
            }
        )

    def walk_forward(
        self,
        max_folds: int | None = None,
        allow_degraded_survivorship: bool = False,
        resume: bool = True,
    ) -> WalkForwardRunResult:
        if max_folds is not None and int(max_folds) <= 0:
            raise ValueError("max_folds must be positive")
        strict_membership = (
            self.config.data.strict_survivorship and not allow_degraded_survivorship
        )
        survivorship_status = "PASS" if strict_membership else "DEGRADED"
        sample_zone = (
            "HISTORICAL_OOS"
            if strict_membership
            else "HISTORICAL_OOS_DEGRADED"
        )
        if survivorship_status == "DEGRADED":
            self.emit(
                "DEGRADED survivorship mode: walk-forward dates and purge/embargo are unchanged, "
                "but results are not strict Historical OOS",
                0.005,
            )
        loader = self._research_loader(strict_membership)
        self.emit("Scanning narrow annual research partitions", 0.01)
        index = loader.scan()
        benchmark_bars = self.store.read_bars(symbols=[self.config.tickflow.benchmark])
        dates = pd.DatetimeIndex(
            pd.to_datetime(benchmark_bars["trade_date"]).drop_duplicates().sort_values()
        )
        splitter = PurgedWalkForward(
            self.config.walk_forward, max_label_horizon=max(self.config.labels.horizons)
        )
        all_folds = list(splitter.split(dates))
        if not all_folds:
            raise ValueError("not enough benchmark sessions for walk-forward")
        folds = all_folds
        if max_folds is not None:
            folds = folds[-int(max_folds) :]
        coverage_status = "FULL" if len(folds) == len(all_folds) else "PARTIAL"
        cache = WalkForwardFoldCache(
            self.config.paths.backtests_dir / "walk_forward_cache"
        )
        artifacts: list[tuple[int, str]] = []
        fold_rows: list[dict[str, Any]] = []
        cached_folds = 0
        run_started = time.perf_counter()
        self.emit(
            f"Walk Forward {coverage_status} · selected {len(folds)}/{len(all_folds)} folds · "
            f"resume {'ON' if resume else 'OFF'}",
            0.02,
        )
        for number, fold in enumerate(folds, start=1):
            signature = self._walk_forward_fold_signature(
                fold, survivorship_status
            )
            artifacts.append((fold.fold_id, signature))
            fold_started = time.perf_counter()
            if resume:
                cached_row = cache.fold_row(fold.fold_id, signature)
                if cached_row is not None:
                    cached_folds += 1
                    cached_row["cache_status"] = "REUSED"
                    fold_rows.append(cached_row)
                    self.emit(
                        f"Fold {fold.fold_id} ({number}/{len(folds)}) reused from cache",
                        0.02 + 0.94 * number / len(folds),
                    )
                    continue

            self.emit(
                f"Fold {fold.fold_id} ({number}/{len(folds)}) preparing bounded partitions",
                0.02 + 0.94 * (number - 1) / len(folds),
            )
            prepared = loader.load_splits(
                index,
                {
                    "train": ResearchSplitSpec(
                        fold.train_dates,
                        self.config.model.max_train_rows,
                        self.config.model.seed,
                    ),
                    "validation": ResearchSplitSpec(
                        fold.validation_dates,
                        self.config.model.max_validation_rows,
                        self.config.model.seed + 1,
                        min_rows_per_date=3,
                    ),
                    "test": ResearchSplitSpec(
                        fold.test_dates,
                        limit=None,
                        seed=self.config.model.seed,
                        require_any_label=False,
                        include_labels=False,
                    ),
                },
            )
            train = prepared["train"]
            validation = prepared["validation"]
            test = prepared["test"]
            version = f"wf-{fold.fold_id}-{fold.test_dates[0].date()}"
            if survivorship_status == "DEGRADED":
                version = f"{version}-degraded"

            def training_progress(
                epoch: int,
                total: int,
                train_loss: float,
                validation_loss: float,
            ) -> None:
                fold_fraction = min(epoch / max(total, 1), 1.0)
                if epoch == 1 or epoch % 5 == 0 or epoch == total:
                    elapsed = time.perf_counter() - fold_started
                    self.emit(
                        f"Fold {fold.fold_id} epoch {epoch}/{total} · "
                        f"train {train_loss:.6f} · validation {validation_loss:.6f} · "
                        f"elapsed {elapsed / 60:.1f}m",
                        0.02
                        + 0.94
                        * ((number - 1) + 0.12 + 0.76 * fold_fraction)
                        / len(folds),
                    )

            model, normalizer, _, training_result = self._fit_one(
                train,
                validation,
                version,
                fold.training_cutoff,
                survivorship_status=survivorship_status,
                training_progress=training_progress,
            )
            test_features = filter_feature_coverage(
                test,
                FEATURE_NAMES,
                self.config.data.min_feature_coverage,
            )
            scored = inference_frame(
                model,
                normalizer,
                test_features,
                FEATURE_NAMES,
                self.config.model.device,
            )
            scored = scored.merge(
                test_features[["symbol", "trade_date", "FeatureCoverage"]],
                on=["symbol", "trade_date"],
                how="left",
                validate="one_to_one",
            )
            scored["fold_id"] = fold.fold_id
            scored["sample_zone"] = sample_zone
            fold_row = fold.as_dict()
            fold_row["survivorship_status"] = survivorship_status
            fold_row["sample_zone"] = sample_zone
            fold_row["cache_status"] = "COMPUTED"
            fold_row["prediction_rows"] = len(scored)
            fold_row["epochs_trained"] = training_result.epochs_trained
            fold_row["best_validation_loss"] = (
                training_result.best_validation_loss
            )
            fold_row["duration_seconds"] = time.perf_counter() - fold_started
            cache.save(fold.fold_id, signature, scored, fold_row)
            fold_rows.append(fold_row)
            del prepared, train, validation, test, test_features
            del model, normalizer, scored, training_result
            gc.collect()
            elapsed = time.perf_counter() - run_started
            average = elapsed / max(number - cached_folds, 1)
            remaining_uncached = max(len(folds) - number, 0)
            self.emit(
                f"Fold {fold.fold_id} persisted · estimated remaining "
                f"{average * remaining_uncached / 60:.1f}m",
                0.02 + 0.94 * number / len(folds),
            )

        destination = self.config.paths.backtests_dir / "walk_forward_predictions.parquet"
        prediction_rows = cache.publish(
            artifacts, destination, coverage_status=coverage_status
        )
        ParquetStore._atomic_csv(
            pd.DataFrame(fold_rows),
            self.config.paths.backtests_dir / "walk_forward_folds.csv",
        )
        self.store.write_manifest(
            "walk_forward",
            {
                "survivorship_status": survivorship_status,
                "sample_zone": sample_zone,
                "coverage_status": coverage_status,
                "folds": len(folds),
                "selected_folds": len(folds),
                "total_folds": len(all_folds),
                "cached_folds": cached_folds,
                "resume_enabled": bool(resume),
                "cache_version": WALK_FORWARD_CACHE_VERSION,
                "selection_instrument_types": list(
                    self.config.portfolio.selection_instrument_types
                ),
                "min_feature_coverage": self.config.data.min_feature_coverage,
                "predictions": prediction_rows,
            },
        )
        self.emit(
            f"{sample_zone} {coverage_status} predictions: {prediction_rows:,} · "
            f"cache hits {cached_folds}/{len(folds)}",
            1.0,
        )
        return WalkForwardRunResult(
            predictions_path=destination,
            selected_folds=len(folds),
            total_folds=len(all_folds),
            predictions=prediction_rows,
            cached_folds=cached_folds,
            coverage_status=coverage_status,
            sample_zone=sample_zone,
        )

    def run_backtest(self) -> BacktestResult:
        prediction_path = self.config.paths.backtests_dir / "walk_forward_predictions.parquet"
        if not prediction_path.exists():
            raise FileNotFoundError("walk-forward predictions missing")
        predictions = pd.read_parquet(
            prediction_path,
            columns=["symbol", "trade_date", "NeuralRank", "NeuralAlpha"],
        )
        start = pd.to_datetime(predictions["trade_date"]).min() - pd.Timedelta(days=7)
        benchmark_bars = self.store.read_bars(
            start_date=start, symbols=[self.config.tickflow.benchmark]
        )
        calendar = pd.DatetimeIndex(
            pd.to_datetime(benchmark_bars["trade_date"])
            .drop_duplicates()
            .sort_values()
        )
        targets = prepare_signal_targets(
            predictions, calendar, self.config.portfolio
        )
        needed_symbols = {self.config.tickflow.benchmark}
        needed_symbols.update(
            symbol for symbols in targets.values() for symbol in symbols
        )
        bars = self.store.read_bars(
            start_date=start, symbols=sorted(needed_symbols)
        )
        snapshot_dates = self.store.universe_snapshot_dates()
        metadata = (
            self.store.read_universe_asof(snapshot_dates[-1], strict=True) if snapshot_dates else pd.DataFrame()
        )
        result = PortfolioBacktester(self.config.portfolio).run(
            bars,
            predictions,
            metadata,
            benchmark=self.config.tickflow.benchmark,
            prepared_targets=targets,
        )
        for name, frame in {
            "nav": result.nav,
            "trades": result.trades,
            "position_ledger": result.position_ledger,
            "exit_events": result.exit_events,
        }.items():
            ParquetStore._atomic_parquet(frame, self.config.paths.backtests_dir / f"{name}.parquet")
        write_status_json(self.config.paths.backtests_dir / "metrics.json", result.metrics)
        self.emit(f"Backtest complete · NAV {result.nav['nav'].iloc[-1]:,.2f}", 1.0)
        return result

    def _current_catalog(
        self, asof_date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        dates = self.store.universe_snapshot_dates()
        if asof_date is not None:
            target = pd.Timestamp(asof_date).normalize()
            dates = [date for date in dates if date <= target]
        if not dates:
            return pd.DataFrame()
        return self.store.read_universe_asof(dates[-1], strict=True)

    def _infer_latest(self) -> tuple[pd.DataFrame, Any]:
        latest = self.store.latest_bar_date()
        if latest is None:
            raise FileNotFoundError("no TickFlow daily data")
        context = self.store.read_bars(start_date=latest - pd.Timedelta(days=420), end_date=latest)
        feature_result = build_features(context, self.config.features)
        latest_features = feature_result.frame[feature_result.frame["trade_date"].eq(latest)].copy()
        observed_catalog = self._current_catalog(latest)
        catalog = filter_catalog(
            observed_catalog,
            self.config.portfolio.selection_instrument_types,
        )
        if not observed_catalog.empty:
            if catalog.empty:
                raise ValueError(
                    "observed TickFlow catalog has no eligible stock candidates"
                )
            latest_features = latest_features[latest_features["symbol"].isin(set(catalog["symbol"]))]
        else:
            self.emit(
                "No universe snapshot was observable by the signal date; "
                "using identifier-only stock classification",
                None,
            )
            latest_features = filter_degraded_symbol_universe(
                latest_features,
                self.config.portfolio.selection_instrument_types,
            )
        latest_features = filter_feature_coverage(
            latest_features,
            FEATURE_NAMES,
            self.config.data.min_feature_coverage,
        )
        if latest_features.empty:
            raise ValueError(
                "no stock has enough point-in-time feature coverage for daily inference"
            )
        model, normalizer, metadata = load_checkpoint(self.registry.champion_checkpoint())
        if tuple(metadata.feature_names) != tuple(FEATURE_NAMES):
            raise ValueError("champion feature schema does not match current code")
        predictions = inference_frame(
            model,
            normalizer,
            latest_features,
            FEATURE_NAMES,
            self.config.model.device,
        )
        predictions = predictions.merge(
            latest_features[["symbol", "trade_date", "FeatureCoverage"]],
            on=["symbol", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        if not catalog.empty:
            catalog = catalog.copy()
            for column in ("name", "instrument_type"):
                if column not in catalog:
                    catalog[column] = ""
            predictions = predictions.merge(
                catalog[["symbol", "name", "instrument_type"]].drop_duplicates("symbol"),
                on="symbol",
                how="left",
            )
        return predictions, metadata

    def publish_pages(self) -> PagesPublishResult:
        """Validate and push only docs/ without touching the caller's Git state."""

        self.emit("Validating complete Pages site", 0.91)
        result = publish_pages_to_git(
            self.config.paths.docs_dir,
            remote=self.config.reports.pages_remote,
            branch=self.config.reports.pages_branch,
            message=f"reports: publish TickFlow Pages {self.store.latest_bar_date() or ''}".rstrip(),
            progress=lambda message: self.emit(message, None),
        )
        write_status_json(
            self.config.paths.docs_dir / "pages-publish-status.json",
            {
                "status": "PUSHED" if result.pushed else "UNCHANGED",
                "commit_sha": result.commit_sha,
                "remote": result.remote,
                "branch": result.branch,
                "pages": result.pages,
                "updated_at": pd.Timestamp.now(tz="Asia/Shanghai"),
            },
        )
        self.emit(
            (
                f"Pages pushed · {result.commit_sha[:12]}"
                if result.pushed
                else f"Pages already current · {result.commit_sha[:12]}"
            ),
            0.99,
        )
        return result

    def _auto_publish_pages(self) -> PagesPublishResult | None:
        if not self.config.reports.auto_push_pages:
            self.emit("Pages auto-push disabled; local site is ready", None)
            return None
        try:
            return self.publish_pages()
        except Exception as exc:
            # A credential, network, branch-protection, CI or deployment
            # failure must never invalidate the already healthy local/remote
            # site. The manual GUI/CLI command remains available for retry.
            LOGGER.exception("Pages auto-push failed")
            try:
                write_status_json(
                    self.config.paths.docs_dir / "pages-publish-status.json",
                    {
                        "status": "FAILED",
                        "detail": (
                            "See the local GUI/log for the Git authentication, "
                            "network, branch protection, CI or deployment error."
                        ),
                        "error_type": type(exc).__name__,
                        "remote": self.config.reports.pages_remote,
                        "branch": self.config.reports.pages_branch,
                        "updated_at": pd.Timestamp.now(tz="Asia/Shanghai"),
                    },
                )
            except Exception:
                LOGGER.exception("Unable to persist Pages failure status")
            self.emit(
                "Pages auto-push failed; previous healthy site is unchanged · "
                f"{type(exc).__name__}: {exc}",
                None,
            )
            return None

    def daily(self, skip_update: bool = False) -> pd.DataFrame:
        quality = (
            check_bars(self.store.read_bars(), benchmark=self.config.tickflow.benchmark)
            if skip_update
            else self.update_tickflow(full=False)
        )
        latest = quality.latest_date
        if latest is None:
            raise RuntimeError("latest TickFlow date unavailable")
        self.build_derived(years=[latest.year])
        predictions, metadata = self._infer_latest()
        destination = self.config.paths.predictions_dir / f"{latest.date()}.parquet"
        ParquetStore._atomic_parquet(predictions, destination)
        predictions.to_csv(self.config.paths.predictions_dir / f"{latest.date()}.csv", index=False)
        context = self._report_context(predictions, metadata.model_version, metadata.training_cutoff, quality)
        self.publisher.publish(context, include_weekly=False)
        write_status_json(
            self.config.paths.docs_dir / "status.json",
            {
                "latest_date": latest,
                "tickflow_status": quality.status,
                "model_version": metadata.model_version,
                "training_cutoff": metadata.training_cutoff,
                "training_cutoff_semantics": metadata.training_cutoff_semantics,
                "data_cutoff": metadata.data_cutoff,
                "signal_source": "champion_mlp",
                "generated_at": context.generated_at,
                "prediction_date": context.data_date,
                "prediction_rows": context.prediction_rows,
                "prediction_fingerprint": context.prediction_fingerprint,
                "selection_instrument_types": list(
                    self.config.portfolio.selection_instrument_types
                ),
                "min_feature_coverage": self.config.data.min_feature_coverage,
                "top_k": predictions.nsmallest(self.config.portfolio.top_k, "NeuralRank")[
                    [
                        "symbol",
                        "Alpha20",
                        "Alpha40",
                        "Alpha60",
                        "NeuralAlpha",
                        "NeuralRank",
                        "FeatureCoverage",
                    ]
                ].to_dict("records"),
            },
        )
        self._auto_publish_pages()
        self.emit(f"DAILY published · {len(predictions)} neural predictions", 1.0)
        return predictions

    def weekly(self) -> ReportContext:
        prediction_files = sorted(self.config.paths.predictions_dir.glob("*.parquet"))
        if prediction_files:
            predictions = pd.read_parquet(prediction_files[-1])
        else:
            predictions = pd.DataFrame(
                columns=["symbol", "name", "Alpha20", "Alpha40", "Alpha60", "NeuralAlpha", "NeuralRank"]
            )
        registry = self.registry.read()
        champion = registry.get("champion") or "UNTRAINED"
        model_info = registry.get("models", {}).get(champion, {})
        quality_manifest = self.store.read_manifest("tickflow")
        quality_payload = quality_manifest.get("quality", {"status": "PENDING"})
        quality = DataQualityReport(
            status=quality_payload.get("status", "PENDING"),
            latest_date=pd.Timestamp(quality_payload["latest_date"]) if quality_payload.get("latest_date") else self.store.latest_bar_date(),
            rows=int(quality_payload.get("rows", 0)),
            symbols=int(quality_payload.get("symbols", 0)),
            calendar_days=int(quality_payload.get("calendar_days", 0)),
            latest_coverage=float(quality_payload.get("latest_coverage", 0.0)),
            issues=[],
        )
        context = self._report_context(
            predictions,
            champion,
            model_info.get("training_cutoff", "N/A"),
            quality,
        )
        self.publisher.publish(context, include_weekly=True)
        self._auto_publish_pages()
        self.emit("WEEKLY published", 1.0)
        return context

    def _report_context(
        self,
        predictions: pd.DataFrame,
        model_version: str,
        training_cutoff: str,
        quality: DataQualityReport,
    ) -> ReportContext:
        latest = quality.latest_date or self.store.latest_bar_date() or pd.Timestamp.today()
        nav_path = self.config.paths.backtests_dir / "nav.parquet"
        metrics_path = self.config.paths.backtests_dir / "metrics.json"
        oos_path = self.config.paths.backtests_dir / "walk_forward_predictions.parquet"
        nav = pd.read_parquet(nav_path) if nav_path.exists() else pd.DataFrame()
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rolling = pd.DataFrame()
        quantiles = pd.DataFrame()
        research: dict[str, Any] = {}
        walk_forward = self.store.read_manifest("walk_forward")
        coverage_status = str(
            walk_forward.get("coverage_status", "FULL")
        ).upper()
        if oos_path.exists() and self.store.derived_years("labels"):
            oos = pd.read_parquet(oos_path)
            oos["trade_date"] = pd.to_datetime(oos["trade_date"]).dt.normalize()
            label_columns = [f"label_{h}" for h in self.config.labels.horizons]
            report_columns = [*label_columns, "raw_return_20"]
            joined_parts: list[pd.DataFrame] = []
            for year in sorted(set(oos["trade_date"].dt.year)):
                oos_year = oos[oos["trade_date"].dt.year.eq(year)]
                labels = self.store.read_derived_year(
                    "labels",
                    int(year),
                    columns=["symbol", "trade_date", *report_columns],
                    float32_columns=report_columns,
                )
                if labels.empty:
                    joined_parts.append(oos_year.copy())
                    continue
                labels["trade_date"] = pd.to_datetime(labels["trade_date"]).dt.normalize()
                labels = labels[labels["trade_date"].isin(oos_year["trade_date"].unique())]
                if labels.duplicated(["symbol", "trade_date"]).any():
                    raise ValueError(
                        f"labels year={year} violates symbol + trade_date one-to-one"
                    )
                joined_parts.append(
                    oos_year.merge(
                        labels,
                        on=["symbol", "trade_date"],
                        how="left",
                        validate="one_to_one",
                    )
                )
            joined = pd.concat(joined_parts, ignore_index=True) if joined_parts else oos.copy()
            dates: pd.Series | None = None
            for horizon in self.config.labels.horizons:
                ic = rank_ic_by_date(joined, f"Alpha{horizon}", f"label_{horizon}")
                values = ic.rolling(self.config.reports.rolling_ic_window, min_periods=20).mean()
                if rolling.empty:
                    rolling = pd.DataFrame({"trade_date": ic.index})
                rolling[f"rolling_ic_{horizon}"] = values.reindex(rolling["trade_date"]).to_numpy()
                summary = summarize_ic(ic)
                research[f"icir_{horizon}"] = summary.icir
                research[f"nw_t_{horizon}"] = summary.newey_west_t
            quantiles = quantile_monotonicity(joined, "NeuralAlpha", "raw_return_20")
            research.update(
                {
                    "historical_oos_status": (
                        "DEGRADED"
                        if "sample_zone" in oos
                        and oos["sample_zone"].astype(str).str.contains("DEGRADED").any()
                        else "PASS"
                    ),
                    "icir": " / ".join(_display(research.get(f"icir_{h}")) for h in self.config.labels.horizons),
                    "newey_west_t": " / ".join(_display(research.get(f"nw_t_{h}")) for h in self.config.labels.horizons),
                    "mature_labels": f"{joined[[f'label_{h}' for h in self.config.labels.horizons]].notna().all(axis=1).sum():,}",
                    "ic_decay": "20D / 40D / 60D shown in rolling IC",
                }
            )
        training = self.store.read_manifest("training")
        training_status = training.get("survivorship_status", "PASS") if training else "PENDING"
        fold_path = self.config.paths.backtests_dir / "walk_forward_folds.csv"
        if fold_path.exists():
            folds = pd.read_csv(fold_path)
            research["test_period"] = f"{folds['test_start'].min()} — {folds['test_end'].max()}"
            selected = int(walk_forward.get("selected_folds", len(folds)))
            total = int(walk_forward.get("total_folds", selected))
            research["split_summary"] = (
                f"{selected}/{total} expanding folds · {coverage_status}"
            )
        research.update(
            {
                "in_sample_status": (
                    "DEGRADED" if training_status == "DEGRADED" else "RECORDED"
                ) if training else "PENDING",
                "validation_status": (
                    "DEGRADED" if training_status == "DEGRADED" else "RECORDED"
                ) if training else "PENDING",
                "train_period": f"{training.get('train_start','N/A')} — {training.get('train_end','N/A')}",
                "validation_period": f"{training.get('validation_start','N/A')} — {training.get('validation_end','N/A')}",
                "shadow_status": "PENDING",
                "shadow_period": "accumulates after champion deployment",
                "purge_embargo": f"{self.config.walk_forward.purge_days} / {self.config.walk_forward.embargo_days} sessions",
                "champion_challenger": f"{self.registry.read().get('champion') or 'N/A'} / {len(self.registry.read().get('challengers', []))}",
                "topk_performance": f"Top {self.config.portfolio.top_k}, rebalance every {self.config.portfolio.rebalance_every} sessions",
                "selection_universe": " / ".join(
                    str(value).upper()
                    for value in self.config.portfolio.selection_instrument_types
                ),
                "min_feature_coverage": f"{self.config.data.min_feature_coverage:.0%}",
                "pages_auto_push": bool(self.config.reports.auto_push_pages),
                "pages_target": (
                    f"{self.config.reports.pages_remote}/"
                    f"{self.config.reports.pages_branch}"
                ),
                "yearly_regime_ic": "computed only after each bucket has mature OOS labels",
            }
        )
        generated_at = pd.Timestamp.now(
            tz=self.config.reports.timezone
        ).floor("s")
        fingerprint = prediction_fingerprint(
            predictions,
            model_version=model_version,
            top_k=self.config.portfolio.top_k,
        )
        registry_model = self.registry.read().get("models", {}).get(
            model_version, {}
        )
        cutoff_semantics = str(
            registry_model.get(
                "training_cutoff_semantics", "legacy_data_cutoff"
            )
        )
        research.update(
            {
                "generated_at": str(generated_at),
                "prediction_rows": int(len(predictions)),
                "prediction_fingerprint": fingerprint,
                "training_cutoff_semantics": cutoff_semantics,
                "training_data_cutoff": registry_model.get("data_cutoff", "N/A"),
            }
        )
        bars = self.store.read_bars(symbols=[self.config.tickflow.benchmark])
        benchmark_nav = pd.DataFrame()
        if not bars.empty:
            pit = reconstruct_pit_prices(bars)
            benchmark_nav = pit[["trade_date", "pit_close"]].rename(columns={"pit_close": "nav"})
        audit = audit_survivorship(self.store.universe_snapshot_dates(), nav["trade_date"].min() if not nav.empty else None)
        declared_statuses = {
            audit.status,
            training.get("survivorship_status"),
            walk_forward.get("survivorship_status"),
        }
        if "DEGRADED" in declared_statuses:
            survivorship_status = "DEGRADED"
        elif "FAIL" in declared_statuses:
            survivorship_status = "FAIL"
        else:
            survivorship_status = "PASS"
        historical_status = str(
            research.get("historical_oos_status", survivorship_status)
        )
        if survivorship_status != "PASS" and survivorship_status not in historical_status:
            historical_status = survivorship_status
        if coverage_status != "FULL" and "PARTIAL" not in historical_status:
            historical_status = f"{historical_status} / PARTIAL"
        research["historical_oos_status"] = historical_status
        quality_dict = quality.to_dict()
        quality_dict.update(
            {"pit_status": "PASS", "survivorship_status": survivorship_status}
        )
        return ReportContext(
            data_date=str(pd.Timestamp(latest).date()),
            tickflow_status=quality.status,
            model_version=model_version,
            training_cutoff=str(training_cutoff),
            predictions=predictions,
            generated_at=str(generated_at),
            prediction_fingerprint=fingerprint,
            prediction_rows=int(len(predictions)),
            training_cutoff_semantics=cutoff_semantics,
            rolling_ic=rolling,
            nav=nav,
            benchmark_nav=benchmark_nav,
            quantiles=quantiles,
            metrics=metrics,
            research=research,
            quality=quality_dict,
        )


def _display(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"
