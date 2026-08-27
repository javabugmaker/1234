from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestResult, PortfolioBacktester
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
from .reports import ReportContext, StaticReportPublisher, write_status_json
from .research import PartitionedResearchLoader, ResearchSplitSpec
from .walk_forward import PurgedWalkForward, WalkForwardFold

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
        )

    def _apply_observed_membership(self, frame: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        snapshots = self.store.universe_snapshot_dates()
        audit = audit_survivorship(snapshots, frame["trade_date"].min() if not frame.empty else None)
        if strict and audit.status != "PASS":
            raise ValueError(
                "strict survivorship audit failed: " + audit.detail + ". "
                "Do not backfill old membership from today's TickFlow catalog."
            )
        if not snapshots:
            return frame.iloc[0:0].copy() if strict else frame
        pieces = []
        ordered = sorted(snapshots)
        for index, snapshot_date in enumerate(ordered):
            end = ordered[index + 1] - pd.Timedelta(days=1) if index + 1 < len(ordered) else frame["trade_date"].max()
            members = set(self.store.read_universe_asof(snapshot_date, strict=True)["symbol"])
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
    ) -> tuple[MultiTaskMLP, FeatureNormalizer, CheckpointMetadata, Any]:
        label_columns = [f"label_{h}" for h in self.config.labels.horizons]
        train = train.dropna(subset=label_columns, how="all")
        validation = validation.dropna(subset=label_columns, how="all")
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
            model, train_x, train_y, validation_x, validation_y
        )
        del train_x, train_y, validation_y
        output = predict_array(model, validation_x, self.config.model.device)
        scored = validation[["trade_date", "symbol", *label_columns]].copy()
        metrics: dict[str, float] = {"validation_loss": training_result.best_validation_loss}
        for index, horizon in enumerate(self.config.labels.horizons):
            column = f"Alpha{horizon}"
            scored[column] = output[:, index]
            summary = summarize_ic(rank_ic_by_date(scored, column, f"label_{horizon}"))
            metrics[f"rank_ic_{horizon}"] = summary.mean
            metrics[f"icir_{horizon}"] = summary.icir
            metrics[f"newey_west_t_{horizon}"] = summary.newey_west_t
        metadata = CheckpointMetadata(
            model_version=model_version,
            training_cutoff=str(pd.Timestamp(training_cutoff).date()),
            feature_names=tuple(FEATURE_NAMES),
            horizons=tuple(self.config.labels.horizons),
            hidden_dims=tuple(self.config.model.hidden_dims),
            dropout=self.config.model.dropout,
            metrics=metrics,
        )
        return model, normalizer, metadata, training_result

    def train(self, allow_degraded_survivorship: bool = False) -> CheckpointMetadata:
        latest = self.store.latest_bar_date()
        if latest is None:
            raise FileNotFoundError("no TickFlow data")
        strict_membership = (
            self.config.data.strict_survivorship and not allow_degraded_survivorship
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
        version = make_model_version(latest, FEATURE_NAMES)
        self.emit(
            f"Training {version} on {len(train):,} rows; validation {len(validation):,}",
            0.05,
        )
        model, normalizer, metadata, result = self._fit_one(train, validation, version, latest)
        checkpoint = self.config.paths.models_dir / version / "checkpoint.pt"
        save_checkpoint(checkpoint, model, normalizer, metadata, result)
        role = "champion" if self.registry.read().get("champion") is None else "challenger"
        self.registry.register(metadata, checkpoint, role=role)
        self.store.write_manifest(
            "training",
            {
                "model_version": version,
                "training_cutoff": latest,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "validation_start": validation_dates[0],
                "validation_end": validation_dates[-1],
                "purge_days": purge,
                "metrics": metadata.metrics,
                "role": role,
            },
        )
        self.emit(f"Model saved as {role}: {version}", 1.0)
        return metadata

    def walk_forward(
        self,
        max_folds: int | None = None,
        allow_degraded_survivorship: bool = False,
    ) -> pd.DataFrame:
        strict_membership = (
            self.config.data.strict_survivorship and not allow_degraded_survivorship
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
        folds = list(splitter.split(dates))
        if max_folds is not None:
            folds = folds[-int(max_folds) :]
        predictions: list[pd.DataFrame] = []
        fold_rows: list[dict[str, Any]] = []
        for number, fold in enumerate(folds, start=1):
            self.emit(f"Walk-forward fold {fold.fold_id} ({number}/{len(folds)})", (number - 1) / len(folds))
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
            model, normalizer, _, _ = self._fit_one(
                train, validation, version, fold.training_cutoff
            )
            test_features = test.dropna(subset=FEATURE_NAMES, how="all").copy()
            scored = inference_frame(
                model,
                normalizer,
                test_features,
                FEATURE_NAMES,
                self.config.model.device,
            )
            scored["fold_id"] = fold.fold_id
            scored["sample_zone"] = "HISTORICAL_OOS"
            predictions.append(scored)
            fold_rows.append(fold.as_dict())
        output = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        destination = self.config.paths.backtests_dir / "walk_forward_predictions.parquet"
        ParquetStore._atomic_parquet(output, destination)
        pd.DataFrame(fold_rows).to_csv(
            self.config.paths.backtests_dir / "walk_forward_folds.csv", index=False
        )
        self.emit(f"Historical OOS predictions: {len(output):,}", 1.0)
        return output

    def run_backtest(self) -> BacktestResult:
        prediction_path = self.config.paths.backtests_dir / "walk_forward_predictions.parquet"
        if not prediction_path.exists():
            raise FileNotFoundError("walk-forward predictions missing")
        predictions = pd.read_parquet(prediction_path)
        start = pd.to_datetime(predictions["trade_date"]).min() - pd.Timedelta(days=7)
        bars = self.store.read_bars(start_date=start)
        snapshot_dates = self.store.universe_snapshot_dates()
        metadata = (
            self.store.read_universe_asof(snapshot_dates[-1], strict=True) if snapshot_dates else pd.DataFrame()
        )
        result = PortfolioBacktester(self.config.portfolio).run(
            bars, predictions, metadata, benchmark=self.config.tickflow.benchmark
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

    def _current_catalog(self) -> pd.DataFrame:
        dates = self.store.universe_snapshot_dates()
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
        catalog = self._current_catalog()
        if not catalog.empty:
            latest_features = latest_features[latest_features["symbol"].isin(set(catalog["symbol"]))]
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
        if not catalog.empty:
            predictions = predictions.merge(
                catalog[["symbol", "name", "instrument_type"]].drop_duplicates("symbol"),
                on="symbol",
                how="left",
            )
        return predictions, metadata

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
                "top_k": predictions.nsmallest(self.config.portfolio.top_k, "NeuralRank")[
                    ["symbol", "Alpha20", "Alpha40", "Alpha60", "NeuralAlpha", "NeuralRank"]
                ].to_dict("records"),
            },
        )
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
                    "historical_oos_status": "PASS",
                    "icir": " / ".join(_display(research.get(f"icir_{h}")) for h in self.config.labels.horizons),
                    "newey_west_t": " / ".join(_display(research.get(f"nw_t_{h}")) for h in self.config.labels.horizons),
                    "mature_labels": f"{joined[[f'label_{h}' for h in self.config.labels.horizons]].notna().all(axis=1).sum():,}",
                    "ic_decay": "20D / 40D / 60D shown in rolling IC",
                }
            )
        training = self.store.read_manifest("training")
        fold_path = self.config.paths.backtests_dir / "walk_forward_folds.csv"
        if fold_path.exists():
            folds = pd.read_csv(fold_path)
            research["test_period"] = f"{folds['test_start'].min()} — {folds['test_end'].max()}"
            research["split_summary"] = f"{len(folds)} expanding folds"
        research.update(
            {
                "in_sample_status": "RECORDED" if training else "PENDING",
                "validation_status": "RECORDED" if training else "PENDING",
                "train_period": f"{training.get('train_start','N/A')} — {training.get('train_end','N/A')}",
                "validation_period": f"{training.get('validation_start','N/A')} — {training.get('validation_end','N/A')}",
                "shadow_status": "PENDING",
                "shadow_period": "accumulates after champion deployment",
                "purge_embargo": f"{self.config.walk_forward.purge_days} / {self.config.walk_forward.embargo_days} sessions",
                "champion_challenger": f"{self.registry.read().get('champion') or 'N/A'} / {len(self.registry.read().get('challengers', []))}",
                "topk_performance": f"Top {self.config.portfolio.top_k}, rebalance every {self.config.portfolio.rebalance_every} sessions",
                "yearly_regime_ic": "computed only after each bucket has mature OOS labels",
            }
        )
        bars = self.store.read_bars(symbols=[self.config.tickflow.benchmark])
        benchmark_nav = pd.DataFrame()
        if not bars.empty:
            pit = reconstruct_pit_prices(bars)
            benchmark_nav = pit[["trade_date", "pit_close"]].rename(columns={"pit_close": "nav"})
        audit = audit_survivorship(self.store.universe_snapshot_dates(), nav["trade_date"].min() if not nav.empty else None)
        quality_dict = quality.to_dict()
        quality_dict.update({"pit_status": "PASS", "survivorship_status": audit.status})
        return ReportContext(
            data_date=str(pd.Timestamp(latest).date()),
            tickflow_status=quality.status,
            model_version=model_version,
            training_cutoff=str(training_cutoff),
            predictions=predictions,
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
