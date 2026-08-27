from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..config import TickFlowConfig

LOGGER = logging.getLogger("neural_a_share")
BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
]


def _millis(value: str | pd.Timestamp, end_of_day: bool = False) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Shanghai")
    if end_of_day:
        ts = ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return int(ts.tz_convert("UTC").timestamp() * 1000)


def _columnar_to_frame(symbol: str, payload: Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    timestamps = list(payload.get("timestamp", []))
    size = len(timestamps)
    if not size:
        return pd.DataFrame(columns=BAR_COLUMNS)

    def values(name: str, default: float = float("nan")) -> list[Any]:
        raw = payload.get(name)
        return list(raw) if raw is not None else [default] * size

    utc = pd.to_datetime(timestamps, unit="ms", utc=True)
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": utc.tz_convert("Asia/Shanghai").normalize().tz_localize(None),
            "timestamp": timestamps,
            "open": values("open"),
            "high": values("high"),
            "low": values("low"),
            "close": values("close"),
            "prev_close": values("prev_close"),
            "volume": values("volume", 0),
            "amount": values("amount", 0.0),
        }
    )
    numeric = ["open", "high", "low", "close", "prev_close", "volume", "amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    # Some free-tier rows omit prev_close. A one-day shift is safe within a raw
    # series; the first observation remains unknown instead of being invented.
    frame["prev_close"] = frame["prev_close"].fillna(frame["close"].shift(1))
    return frame[BAR_COLUMNS].sort_values("trade_date").reset_index(drop=True)


@dataclass(frozen=True)
class TickFlowUpdateResult:
    symbols_requested: int
    symbols_received: int
    rows_received: int
    latest_date: pd.Timestamp | None
    catalog: pd.DataFrame
    bars: pd.DataFrame


class TickFlowFreeClient:
    """Thin, injectable adapter around the official ``TickFlow.free()`` SDK.

    No API key and no direct HTTP endpoint are used. Keeping this boundary small
    makes provider responses testable and prevents accidental fallback to another
    data vendor.
    """

    def __init__(
        self,
        config: TickFlowConfig,
        cache_dir: Path | None = None,
        sdk: Any | None = None,
    ) -> None:
        self.config = config
        self._owns_sdk = sdk is None
        if sdk is None:
            try:
                from tickflow import TickFlow
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError("tickflow is required; run: pip install -e .") from exc
            sdk = TickFlow.free(
                timeout=float(config.timeout_seconds),
                max_retries=int(config.max_retries),
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        self.sdk = sdk

    def close(self) -> None:
        if self._owns_sdk and hasattr(self.sdk, "close"):
            self.sdk.close()

    def __enter__(self) -> "TickFlowFreeClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def fetch_catalog(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for exchange in self.config.exchanges:
            for instrument_type in self.config.instrument_types:
                instruments = self.sdk.exchanges.get_instruments(
                    exchange, instrument_type=instrument_type
                )
                for instrument in instruments:
                    ext = instrument.get("ext") or {}
                    rows.append(
                        {
                            "symbol": instrument.get("symbol"),
                            "code": instrument.get("code"),
                            "name": instrument.get("name"),
                            "exchange": instrument.get("exchange", exchange),
                            "region": instrument.get("region", "CN"),
                            "instrument_type": instrument.get("type", instrument_type),
                            "listing_date": ext.get("listing_date"),
                            "float_shares": ext.get("float_shares"),
                            "total_shares": ext.get("total_shares"),
                        }
                    )
        catalog = pd.DataFrame(rows)
        if catalog.empty:
            raise RuntimeError("TickFlow.free() returned an empty A-share catalog")
        catalog = catalog.dropna(subset=["symbol"]).drop_duplicates("symbol", keep="last")
        catalog["listing_date"] = pd.to_datetime(catalog["listing_date"], errors="coerce")
        catalog["observed_at"] = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
        return catalog.sort_values("symbol").reset_index(drop=True)

    def fetch_bars(
        self,
        symbols: Sequence[str],
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        count: int = 10_000,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=BAR_COLUMNS)
        kwargs: dict[str, Any] = {
            "period": self.config.period,
            "count": min(int(count), 10_000),
            "adjust": "none",  # required for point-in-time-safe reconstruction
            "as_dataframe": False,
            "show_progress": False,
            "max_workers": int(self.config.max_workers),
            "batch_size": min(int(self.config.batch_size), 100),
        }
        if start_date is not None:
            kwargs["start_time"] = _millis(start_date)
        if end_date is not None:
            kwargs["end_time"] = _millis(end_date, end_of_day=True)
        payload = self.sdk.klines.batch(list(symbols), **kwargs)
        frames = [_columnar_to_frame(symbol, data) for symbol, data in payload.items()]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS)
        bars = pd.concat(frames, ignore_index=True)
        bars["ingested_at"] = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
        return bars.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def update(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        symbols: Iterable[str] | None = None,
    ) -> TickFlowUpdateResult:
        catalog = self.fetch_catalog()
        requested = list(symbols) if symbols is not None else catalog["symbol"].tolist()
        if self.config.benchmark not in requested:
            requested.append(self.config.benchmark)
        bars = self.fetch_bars(requested, start_date=start_date, end_date=end_date)
        received = int(bars["symbol"].nunique()) if not bars.empty else 0
        latest = pd.Timestamp(bars["trade_date"].max()) if not bars.empty else None
        LOGGER.info(
            "TickFlow.free update: requested=%s received=%s rows=%s latest=%s",
            len(requested), received, len(bars), latest.date() if latest is not None else None,
        )
        return TickFlowUpdateResult(len(requested), received, len(bars), latest, catalog, bars)
