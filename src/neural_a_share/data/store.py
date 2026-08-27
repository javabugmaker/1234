from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


class ParquetStore:
    """Atomic, year-partitioned local cache with immutable universe snapshots."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.universe_dir = self.root / "universe"
        self.derived_dir = self.root / "derived"
        self.manifests_dir = self.root / "manifests"
        for path in (self.bars_dir, self.universe_dir, self.derived_dir, self.manifests_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            frame.to_parquet(temporary, index=False, engine="pyarrow")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def upsert_bars(self, bars: pd.DataFrame) -> int:
        if bars.empty:
            return 0
        required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
        missing = required - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing columns: {sorted(missing)}")
        incoming = bars.copy()
        incoming["trade_date"] = pd.to_datetime(incoming["trade_date"]).dt.normalize()
        incoming["year"] = incoming["trade_date"].dt.year
        written = 0
        for year, chunk in incoming.groupby("year", sort=True):
            path = self.bars_dir / f"year={int(year)}" / "bars.parquet"
            chunk = chunk.drop(columns="year")
            if path.exists():
                existing = pd.read_parquet(path)
                chunk = pd.concat([existing, chunk], ignore_index=True)
            chunk = (
                chunk.sort_values(["symbol", "trade_date", "ingested_at"] if "ingested_at" in chunk else ["symbol", "trade_date"])
                .drop_duplicates(["symbol", "trade_date"], keep="last")
                .sort_values(["trade_date", "symbol"])
                .reset_index(drop=True)
            )
            self._atomic_parquet(chunk, path)
            written += len(chunk)
        return written

    def read_bars(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        symbols: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        files = sorted(self.bars_dir.glob("year=*/bars.parquet"))
        if start_date is not None:
            start_year = pd.Timestamp(start_date).year
            files = [p for p in files if int(p.parent.name.split("=")[1]) >= start_year]
        if end_date is not None:
            end_year = pd.Timestamp(end_date).year
            files = [p for p in files if int(p.parent.name.split("=")[1]) <= end_year]
        if not files:
            return pd.DataFrame()
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        if start_date is not None:
            frame = frame[frame["trade_date"] >= pd.Timestamp(start_date).normalize()]
        if end_date is not None:
            frame = frame[frame["trade_date"] <= pd.Timestamp(end_date).normalize()]
        if symbols is not None:
            wanted = set(symbols)
            frame = frame[frame["symbol"].isin(wanted)]
        return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def latest_bar_date(self) -> pd.Timestamp | None:
        files = sorted(self.bars_dir.glob("year=*/bars.parquet"))
        if not files:
            return None
        latest = pd.read_parquet(files[-1], columns=["trade_date"])["trade_date"].max()
        return pd.Timestamp(latest).normalize() if pd.notna(latest) else None

    def bar_years(self) -> list[int]:
        return sorted(
            int(path.parent.name.split("=")[1])
            for path in self.bars_dir.glob("year=*/bars.parquet")
        )

    def write_universe_snapshot(self, catalog: pd.DataFrame, asof_date: str | pd.Timestamp) -> Path:
        date = pd.Timestamp(asof_date).normalize()
        snapshot = catalog.copy()
        snapshot["snapshot_date"] = date
        snapshot["information_date"] = date
        path = self.universe_dir / f"asof={date.date().isoformat()}.parquet"
        self._atomic_parquet(snapshot.sort_values("symbol").reset_index(drop=True), path)
        return path

    def universe_snapshot_dates(self) -> list[pd.Timestamp]:
        dates = []
        for path in self.universe_dir.glob("asof=*.parquet"):
            dates.append(pd.Timestamp(path.stem.split("=", 1)[1]).normalize())
        return sorted(dates)

    def read_universe_asof(self, asof_date: str | pd.Timestamp, strict: bool = True) -> pd.DataFrame:
        target = pd.Timestamp(asof_date).normalize()
        eligible = [date for date in self.universe_snapshot_dates() if date <= target]
        if not eligible:
            if strict:
                raise ValueError(
                    f"no TickFlow universe snapshot existed by {target.date()}; "
                    "using today's membership would create survivorship bias"
                )
            return pd.DataFrame()
        chosen = eligible[-1]
        return pd.read_parquet(self.universe_dir / f"asof={chosen.date().isoformat()}.parquet")

    def write_derived(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.derived_dir / f"{name}.parquet"
        self._atomic_parquet(frame, path)
        return path

    def write_derived_year(self, name: str, year: int, frame: pd.DataFrame) -> Path:
        path = self.derived_dir / name / f"year={int(year)}" / f"{name}.parquet"
        self._atomic_parquet(frame, path)
        return path

    def read_derived_years(
        self,
        name: str,
        years: Iterable[int] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        files = sorted((self.derived_dir / name).glob(f"year=*/{name}.parquet"))
        if years is not None:
            wanted = {int(year) for year in years}
            files = [path for path in files if int(path.parent.name.split("=")[1]) in wanted]
        if not files:
            return pd.DataFrame(columns=columns)
        return pd.concat((pd.read_parquet(path, columns=columns) for path in files), ignore_index=True)

    def derived_years(self, name: str) -> list[int]:
        return sorted(
            int(path.parent.name.split("=")[1])
            for path in (self.derived_dir / name).glob(f"year=*/{name}.parquet")
        )

    def read_derived(self, name: str) -> pd.DataFrame:
        path = self.derived_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def write_manifest(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.manifests_dir / f"{name}.json"
        self._atomic_json(payload, path)
        return path

    def read_manifest(self, name: str) -> dict[str, Any]:
        path = self.manifests_dir / f"{name}.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
