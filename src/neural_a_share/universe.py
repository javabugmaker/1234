from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import pandas as pd


STOCK_CLASSIFIER_VERSION = 1
_STOCK_PATTERNS = (
    re.compile(r"^(?:600|601|603|605|688|689)\d{3}\.SH$"),
    re.compile(r"^(?:000|001|002|003|300|301)\d{3}\.SZ$"),
    re.compile(r"^(?:4|8)\d{5}\.BJ$"),
    re.compile(r"^920\d{3}\.BJ$"),
)
_STOCK_TYPE_ALIASES = {"stock", "equity", "a_share", "a-share", "ashare"}


def normalize_instrument_type(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = str(value).strip().lower()
    return "stock" if normalized in _STOCK_TYPE_ALIASES else normalized


def normalized_allowed_types(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalize_instrument_type(value) for value in values))


def is_a_share_stock_symbol(symbol: object) -> bool:
    if symbol is None or pd.isna(symbol):
        return False
    value = str(symbol).strip().upper()
    return any(pattern.fullmatch(value) is not None for pattern in _STOCK_PATTERNS)


def stock_symbol_mask(symbols: pd.Series) -> pd.Series:
    values = symbols.astype("string").str.upper()
    mask = pd.Series(False, index=symbols.index)
    for pattern in _STOCK_PATTERNS:
        mask |= values.str.fullmatch(pattern.pattern, na=False)
    return mask


def filter_catalog(
    catalog: pd.DataFrame,
    allowed_types: Sequence[str],
) -> pd.DataFrame:
    """Filter an observed catalog without inventing past membership."""

    if catalog.empty:
        return catalog.copy()
    allowed = set(normalized_allowed_types(allowed_types))
    if not allowed:
        return catalog.iloc[0:0].copy()
    if "instrument_type" in catalog:
        types = catalog["instrument_type"].map(normalize_instrument_type)
        mask = types.isin(allowed)
        # Some free-service catalog rows can omit the type. Symbol fallback is
        # used only for those missing rows, never to override an observed ETF or
        # fund classification.
        if "stock" in allowed:
            mask |= types.eq("") & stock_symbol_mask(catalog["symbol"])
        return catalog.loc[mask].copy()
    if allowed == {"stock"}:
        return catalog.loc[stock_symbol_mask(catalog["symbol"])].copy()
    return catalog.copy()


def filter_degraded_symbol_universe(
    frame: pd.DataFrame,
    allowed_types: Sequence[str],
) -> pd.DataFrame:
    """Use only identifier-level asset classification in degraded history.

    This intentionally does not use today's TickFlow catalog, which would drop
    historical delistings and introduce survivorship bias.  The deterministic
    stock-code classification only decides asset class; the run remains marked
    DEGRADED because historical membership itself is still unknown.
    """

    if frame.empty:
        return frame.copy()
    allowed = set(normalized_allowed_types(allowed_types))
    if allowed == {"stock"}:
        return frame.loc[stock_symbol_mask(frame["symbol"])].copy()
    return frame.copy()


def feature_coverage(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float32", index=frame.index)
    return frame[list(feature_names)].notna().mean(axis=1).astype("float32")


def filter_feature_coverage(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    minimum: float,
) -> pd.DataFrame:
    threshold = float(minimum)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_feature_coverage must be between 0 and 1")
    if frame.empty:
        result = frame.copy()
        result["FeatureCoverage"] = pd.Series(dtype="float32")
        return result
    coverage = feature_coverage(frame, feature_names)
    result = frame.loc[coverage.ge(threshold)].copy()
    result["FeatureCoverage"] = coverage.loc[result.index].to_numpy()
    return result
