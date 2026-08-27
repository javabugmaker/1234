from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    models_dir: Path = Path("data/models")
    predictions_dir: Path = Path("data/predictions")
    backtests_dir: Path = Path("data/backtests")
    logs_dir: Path = Path("data/logs")
    docs_dir: Path = Path("docs")


@dataclass(frozen=True)
class TickFlowConfig:
    service: str = "free"
    exchanges: tuple[str, ...] = ("SH", "SZ", "BJ")
    instrument_types: tuple[str, ...] = ("stock", "etf", "fund")
    benchmark: str = "000300.SH"
    batch_size: int = 100
    max_workers: int = 2
    timeout_seconds: int = 45
    max_retries: int = 4
    history_start: str = "2005-01-01"
    period: str = "1d"
    adjust: str = "none"


@dataclass(frozen=True)
class DataConfig:
    strict_pit: bool = True
    strict_survivorship: bool = True
    max_staleness_trading_days: int = 1
    min_history_days: int = 260
    universe_snapshot_hour_cst: int = 15


@dataclass(frozen=True)
class FeatureConfig:
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    min_cross_section: int = 20
    max_lookback: int = 240


@dataclass(frozen=True)
class LabelConfig:
    horizons: tuple[int, ...] = (20, 40, 60)
    benchmark: str = "000300.SH"
    cross_sectional_standardize: bool = False


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_days: int = 756
    validation_days: int = 126
    test_days: int = 63
    step_days: int = 63
    purge_days: int = 60
    embargo_days: int = 5


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.15
    batch_size: int = 4096
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 0.02
    patience: int = 10
    min_delta: float = 1e-6
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    num_workers: int = 0
    seed: int = 20260827
    max_train_rows: int = 2_500_000
    max_validation_rows: int = 500_000


@dataclass(frozen=True)
class PortfolioConfig:
    top_k: int = 30
    rebalance_every: int = 5
    initial_cash: float = 1_000_000.0
    max_holding_days: int = 60
    lot_size: int = 100
    max_weight: float = 0.05
    max_participation: float = 0.05
    fixed_slippage_bps: float = 5.0
    impact_coefficient: float = 0.10
    impact_exponent: float = 0.5
    max_impact_bps: float = 30.0
    stock_commission_buy: float = 0.00008499999
    stock_commission_sell: float = 0.00008499999
    fund_commission_buy: float = 0.00005000001
    fund_commission_sell: float = 0.00005000001
    stock_stamp_duty_sell: float = 0.0005
    minimum_commission: float = 0.0
    max_exit_deferral_days: int = 10


@dataclass(frozen=True)
class ReportConfig:
    title: str = "TickFlow Neural Alpha"
    timezone: str = "Asia/Shanghai"
    rolling_ic_window: int = 63


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    tickflow: TickFlowConfig = field(default_factory=TickFlowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)

    def ensure_directories(self) -> None:
        for path in vars(self.paths).values():
            Path(path).mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _paths_from(mapping: Mapping[str, Any], root: Path) -> PathsConfig:
    values: dict[str, Path] = {}
    for key, raw in mapping.items():
        path = Path(raw)
        values[key] = path if path.is_absolute() else (root / path).resolve()
    return PathsConfig(**values)


def load_config(path: str | Path = "config/default.yaml", overrides: Mapping[str, Any] | None = None) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _deep_merge(raw, overrides or {})
    project_root = config_path.parent.parent
    return AppConfig(
        paths=_paths_from(raw.get("paths", {}), project_root),
        tickflow=TickFlowConfig(**raw.get("tickflow", {})),
        data=DataConfig(**raw.get("data", {})),
        features=FeatureConfig(**raw.get("features", {})),
        labels=LabelConfig(**raw.get("labels", {})),
        walk_forward=WalkForwardConfig(**raw.get("walk_forward", {})),
        model=ModelConfig(**raw.get("model", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        reports=ReportConfig(**raw.get("reports", {})),
    )
