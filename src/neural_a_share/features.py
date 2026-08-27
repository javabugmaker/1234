from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data.pit import assert_no_future_information, reconstruct_pit_prices


RET_WINDOWS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 240)
VOL_WINDOWS = (5, 10, 15, 20, 30, 40, 60, 90, 120)
MA_WINDOWS = (3, 5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 240)
POSITION_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
ATR_WINDOWS = (5, 10, 14, 20, 40, 60)
FLOW_WINDOWS = (3, 5, 10, 20, 40, 60, 120)
ZSCORE_WINDOWS = (5, 10, 20, 40, 60, 120)
RSI_WINDOWS = (6, 12, 14, 24, 48)
CMF_WINDOWS = (5, 10, 20, 40, 60)
OBV_WINDOWS = (5, 10, 20, 40, 60, 120)
ACCEL_WINDOWS = (5, 10, 20, 40, 60)
MACD_SPECS = ((6, 19), (8, 24), (12, 26), (20, 50))
SHAPE_FEATURES = (
    "gap_1d",
    "intraday_return",
    "range_pct",
    "upper_shadow",
    "lower_shadow",
    "close_location",
    "log_amount",
)


def feature_names() -> list[str]:
    names = [f"ret_{w}" for w in RET_WINDOWS]
    names += [f"vol_{w}" for w in VOL_WINDOWS]
    names += [f"downvol_{w}" for w in VOL_WINDOWS]
    names += [f"ma_ratio_{w}" for w in MA_WINDOWS]
    names += [f"price_z_{w}" for w in POSITION_WINDOWS]
    names += [f"range_position_{w}" for w in POSITION_WINDOWS]
    names += [f"atr_{w}" for w in ATR_WINDOWS]
    names += [f"volume_ratio_{w}" for w in FLOW_WINDOWS]
    names += [f"volume_z_{w}" for w in ZSCORE_WINDOWS]
    names += [f"amount_ratio_{w}" for w in FLOW_WINDOWS]
    names += [f"amount_z_{w}" for w in ZSCORE_WINDOWS]
    names += [f"rsi_{w}" for w in RSI_WINDOWS]
    names += [f"cmf_{w}" for w in CMF_WINDOWS]
    names += [f"obv_trend_{w}" for w in OBV_WINDOWS]
    names += [f"momentum_accel_{w}" for w in ACCEL_WINDOWS]
    names += [f"macd_{fast}_{slow}" for fast, slow in MACD_SPECS]
    names += list(SHAPE_FEATURES)
    return names


FEATURE_NAMES = feature_names()
assert 80 <= len(FEATURE_NAMES) <= 160


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.pct_change()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = _safe_div(gain, loss)
    return (100 - 100 / (1 + rs)).div(100).sub(0.5).mul(2)


def _symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    out = group.sort_values("trade_date").copy()
    computed: dict[str, pd.Series] = {}
    close = out["pit_close"]
    high = out["pit_high"]
    low = out["pit_low"]
    open_ = out["pit_open"]
    daily = out["daily_total_return"]
    volume = out["volume"].astype(float)
    amount = out["amount"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)

    for window in RET_WINDOWS:
        computed[f"ret_{window}"] = close.pct_change(window)
    for window in VOL_WINDOWS:
        computed[f"vol_{window}"] = daily.rolling(window, min_periods=window).std(ddof=0)
        downside = daily.where(daily < 0, 0.0)
        computed[f"downvol_{window}"] = downside.rolling(window, min_periods=window).std(ddof=0)
    for window in MA_WINDOWS:
        mean = close.rolling(window, min_periods=window).mean()
        computed[f"ma_ratio_{window}"] = _safe_div(close, mean).sub(1.0)
    for window in POSITION_WINDOWS:
        mean = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=0)
        computed[f"price_z_{window}"] = _safe_div(close - mean, std)
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        computed[f"range_position_{window}"] = _safe_div(close - rolling_low, rolling_high - rolling_low).sub(0.5).mul(2)
    for window in ATR_WINDOWS:
        atr = true_range.rolling(window, min_periods=window).mean()
        computed[f"atr_{window}"] = _safe_div(atr, close)
    for window in FLOW_WINDOWS:
        computed[f"volume_ratio_{window}"] = _safe_div(
            volume, volume.rolling(window, min_periods=window).mean()
        ).sub(1.0)
        computed[f"amount_ratio_{window}"] = _safe_div(
            amount, amount.rolling(window, min_periods=window).mean()
        ).sub(1.0)
    for window in ZSCORE_WINDOWS:
        volume_mean = volume.rolling(window, min_periods=window).mean()
        volume_std = volume.rolling(window, min_periods=window).std(ddof=0)
        amount_mean = amount.rolling(window, min_periods=window).mean()
        amount_std = amount.rolling(window, min_periods=window).std(ddof=0)
        computed[f"volume_z_{window}"] = _safe_div(volume - volume_mean, volume_std)
        computed[f"amount_z_{window}"] = _safe_div(amount - amount_mean, amount_std)
    for window in RSI_WINDOWS:
        computed[f"rsi_{window}"] = _rsi(close, window)
    money_flow_multiplier = _safe_div((close - low) - (high - close), high - low).fillna(0.0)
    money_flow_volume = money_flow_multiplier * volume
    for window in CMF_WINDOWS:
        computed[f"cmf_{window}"] = _safe_div(
            money_flow_volume.rolling(window, min_periods=window).sum(),
            volume.rolling(window, min_periods=window).sum(),
        )
    signed_volume = np.sign(daily.fillna(0.0)) * volume
    obv = signed_volume.cumsum()
    for window in OBV_WINDOWS:
        computed[f"obv_trend_{window}"] = _safe_div(obv.diff(window), volume.rolling(window, min_periods=window).sum())
    for window in ACCEL_WINDOWS:
        momentum = close.pct_change(window)
        computed[f"momentum_accel_{window}"] = momentum - momentum.shift(window)
    for fast, slow in MACD_SPECS:
        fast_ema = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
        slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        computed[f"macd_{fast}_{slow}"] = _safe_div(fast_ema - slow_ema, close)

    computed["gap_1d"] = _safe_div(open_, previous_close).sub(1.0)
    computed["intraday_return"] = _safe_div(close, open_).sub(1.0)
    computed["range_pct"] = _safe_div(high - low, previous_close)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    computed["upper_shadow"] = _safe_div(high - body_high, previous_close)
    computed["lower_shadow"] = _safe_div(body_low - low, previous_close)
    computed["close_location"] = _safe_div(close - low, high - low).sub(0.5).mul(2)
    computed["log_amount"] = np.log1p(amount.clip(lower=0))
    return pd.concat([out, pd.DataFrame(computed, index=out.index)], axis=1)


def _cross_sectional_normalize(
    frame: pd.DataFrame,
    names: Iterable[str],
    lower: float,
    upper: float,
    minimum: int,
) -> pd.DataFrame:
    out = frame.copy()
    columns = list(names)
    normalized = np.full((len(out), len(columns)), np.nan, dtype="float32")
    for indices in out.groupby("trade_date", sort=False).indices.values():
        positions = np.asarray(indices, dtype=int)
        matrix = out.iloc[positions][columns].to_numpy(dtype=float)
        counts = np.isfinite(matrix).sum(axis=0)
        valid_columns = counts >= minimum
        if not valid_columns.any():
            continue
        selected = matrix[:, valid_columns]
        lo = np.nanquantile(selected, lower, axis=0)
        hi = np.nanquantile(selected, upper, axis=0)
        selected = np.clip(selected, lo, hi)
        mean = np.nanmean(selected, axis=0)
        std = np.nanstd(selected, axis=0)
        std = np.where(std > 1e-12, std, np.nan)
        normalized[np.ix_(positions, np.flatnonzero(valid_columns))] = ((selected - mean) / std).astype("float32")
    out[columns] = normalized
    return out


def _rolling(series: pd.Series, keys: pd.Series, window: int, operation: str) -> pd.Series:
    grouped = series.groupby(keys, sort=False).rolling(window, min_periods=window)
    result = grouped.std(ddof=0) if operation == "std" else getattr(grouped, operation)()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def _ewm(series: pd.Series, keys: pd.Series, *, alpha: float | None = None, span: int | None = None, min_periods: int) -> pd.Series:
    kwargs: dict[str, float | int | bool] = {"adjust": False, "min_periods": min_periods}
    if alpha is not None:
        kwargs["alpha"] = alpha
    if span is not None:
        kwargs["span"] = span
    result = series.groupby(keys, sort=False).ewm(**kwargs).mean()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def _vectorized_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True).copy()
    keys = out["symbol"]
    close = out["pit_close"]
    high = out["pit_high"]
    low = out["pit_low"]
    open_ = out["pit_open"]
    daily = out["daily_total_return"]
    volume = out["volume"].astype(float)
    amount = out["amount"].astype(float)
    previous_close = close.groupby(keys, sort=False).shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    computed: dict[str, pd.Series] = {}
    for window in RET_WINDOWS:
        computed[f"ret_{window}"] = close.groupby(keys, sort=False).pct_change(window)
    downside = daily.where(daily < 0, 0.0)
    for window in VOL_WINDOWS:
        computed[f"vol_{window}"] = _rolling(daily, keys, window, "std").fillna(np.nan)
        computed[f"downvol_{window}"] = _rolling(downside, keys, window, "std").fillna(np.nan)
    for window in MA_WINDOWS:
        mean = _rolling(close, keys, window, "mean")
        computed[f"ma_ratio_{window}"] = _safe_div(close, mean).sub(1.0)
    for window in POSITION_WINDOWS:
        mean = _rolling(close, keys, window, "mean")
        std = _rolling(close, keys, window, "std")
        computed[f"price_z_{window}"] = _safe_div(close - mean, std)
        rolling_high = _rolling(high, keys, window, "max")
        rolling_low = _rolling(low, keys, window, "min")
        computed[f"range_position_{window}"] = _safe_div(close - rolling_low, rolling_high - rolling_low).sub(0.5).mul(2)
    for window in ATR_WINDOWS:
        computed[f"atr_{window}"] = _safe_div(_rolling(true_range, keys, window, "mean"), close)
    for window in FLOW_WINDOWS:
        computed[f"volume_ratio_{window}"] = _safe_div(volume, _rolling(volume, keys, window, "mean")).sub(1.0)
        computed[f"amount_ratio_{window}"] = _safe_div(amount, _rolling(amount, keys, window, "mean")).sub(1.0)
    for window in ZSCORE_WINDOWS:
        volume_mean = _rolling(volume, keys, window, "mean")
        amount_mean = _rolling(amount, keys, window, "mean")
        computed[f"volume_z_{window}"] = _safe_div(volume - volume_mean, _rolling(volume, keys, window, "std"))
        computed[f"amount_z_{window}"] = _safe_div(amount - amount_mean, _rolling(amount, keys, window, "std"))
    delta = close.groupby(keys, sort=False).pct_change()
    for window in RSI_WINDOWS:
        gain = _ewm(delta.clip(lower=0), keys, alpha=1 / window, min_periods=window)
        loss = _ewm(-delta.clip(upper=0), keys, alpha=1 / window, min_periods=window)
        rs = _safe_div(gain, loss)
        computed[f"rsi_{window}"] = (100 - 100 / (1 + rs)).div(100).sub(0.5).mul(2)
    money_flow_multiplier = _safe_div((close - low) - (high - close), high - low).fillna(0.0)
    money_flow_volume = money_flow_multiplier * volume
    for window in CMF_WINDOWS:
        computed[f"cmf_{window}"] = _safe_div(
            _rolling(money_flow_volume, keys, window, "sum"),
            _rolling(volume, keys, window, "sum"),
        )
    obv = (np.sign(daily.fillna(0.0)) * volume).groupby(keys, sort=False).cumsum()
    for window in OBV_WINDOWS:
        computed[f"obv_trend_{window}"] = _safe_div(
            obv.groupby(keys, sort=False).diff(window),
            _rolling(volume, keys, window, "sum"),
        )
    for window in ACCEL_WINDOWS:
        momentum = close.groupby(keys, sort=False).pct_change(window)
        computed[f"momentum_accel_{window}"] = momentum - momentum.groupby(keys, sort=False).shift(window)
    for fast, slow in MACD_SPECS:
        fast_ema = _ewm(close, keys, span=fast, min_periods=slow)
        slow_ema = _ewm(close, keys, span=slow, min_periods=slow)
        computed[f"macd_{fast}_{slow}"] = _safe_div(fast_ema - slow_ema, close)
    computed["gap_1d"] = _safe_div(open_, previous_close).sub(1.0)
    computed["intraday_return"] = _safe_div(close, open_).sub(1.0)
    computed["range_pct"] = _safe_div(high - low, previous_close)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    computed["upper_shadow"] = _safe_div(high - body_high, previous_close)
    computed["lower_shadow"] = _safe_div(body_low - low, previous_close)
    computed["close_location"] = _safe_div(close - low, high - low).sub(0.5).mul(2)
    computed["log_amount"] = np.log1p(amount.clip(lower=0))
    return pd.concat([out, pd.DataFrame(computed, index=out.index)], axis=1)


@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame
    names: tuple[str, ...]
    lookback: int


def build_features(bars: pd.DataFrame, config: FeatureConfig | None = None) -> FeatureResult:
    config = config or FeatureConfig()
    pit = reconstruct_pit_prices(bars)
    features = _vectorized_features(pit)
    features = _cross_sectional_normalize(
        features,
        FEATURE_NAMES,
        config.winsor_lower,
        config.winsor_upper,
        config.min_cross_section,
    )
    features["feature_information_date"] = pd.to_datetime(features["trade_date"])
    assert_no_future_information(features)
    columns = ["symbol", "trade_date", "feature_information_date", *FEATURE_NAMES]
    result = features[columns].replace([np.inf, -np.inf], np.nan)
    return FeatureResult(result, tuple(FEATURE_NAMES), max(RET_WINDOWS))
