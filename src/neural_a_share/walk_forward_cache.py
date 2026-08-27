from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data.store import ParquetStore

WALK_FORWARD_CACHE_VERSION = 1


def stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_inventory(
    paths: Iterable[Path], relative_to: Path | None = None
) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(set(Path(value) for value in paths)):
        stat = path.stat()
        try:
            name = str(path.relative_to(relative_to)) if relative_to else str(path)
        except ValueError:
            name = str(path)
        inventory.append(
            {"path": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    return inventory


@dataclass(frozen=True)
class CachedFold:
    predictions: pd.DataFrame
    fold_row: dict[str, Any]


@dataclass(frozen=True)
class WalkForwardRunResult:
    predictions_path: Path
    selected_folds: int
    total_folds: int
    predictions: int
    cached_folds: int
    coverage_status: str
    sample_zone: str


class WalkForwardFoldCache:
    """Atomic per-fold predictions enabling safe resume after interruption."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, fold_id: int) -> tuple[Path, Path]:
        directory = self.root / f"fold={int(fold_id):03d}"
        return directory / "predictions.parquet", directory / "manifest.json"

    def has(self, fold_id: int, signature: str) -> bool:
        prediction_path, manifest_path = self._paths(fold_id)
        if not prediction_path.exists() or not manifest_path.exists():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            payload.get("version") == WALK_FORWARD_CACHE_VERSION
            and payload.get("signature") == signature
            and payload.get("complete") is True
        )

    def load(self, fold_id: int, signature: str) -> CachedFold | None:
        if not self.has(fold_id, signature):
            return None
        prediction_path, manifest_path = self._paths(fold_id)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            predictions = pd.read_parquet(prediction_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if "fold_id" not in predictions or (
            not predictions.empty
            and not predictions["fold_id"].eq(int(fold_id)).all()
        ):
            return None
        return CachedFold(predictions, dict(payload.get("fold_row", {})))

    def fold_row(self, fold_id: int, signature: str) -> dict[str, Any] | None:
        """Read only the small fold manifest when predictions need not be loaded."""

        if not self.has(fold_id, signature):
            return None
        prediction_path, manifest_path = self._paths(fold_id)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            parquet = pq.ParquetFile(prediction_path)
            if parquet.metadata.num_rows != int(payload.get("rows", -1)):
                return None
            if "fold_id" not in parquet.schema_arrow.names:
                return None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        row = payload.get("fold_row")
        return dict(row) if isinstance(row, dict) else None

    def save(
        self,
        fold_id: int,
        signature: str,
        predictions: pd.DataFrame,
        fold_row: Mapping[str, Any],
    ) -> None:
        prediction_path, manifest_path = self._paths(fold_id)
        ParquetStore._atomic_parquet(predictions, prediction_path)
        ParquetStore._atomic_json(
            {
                "version": WALK_FORWARD_CACHE_VERSION,
                "signature": signature,
                "complete": True,
                "rows": len(predictions),
                "fold_row": dict(fold_row),
            },
            manifest_path,
        )

    def publish(
        self,
        folds: Sequence[tuple[int, str]],
        destination: str | Path,
        coverage_status: str,
    ) -> int:
        """Stream cached fold files into one atomic parquet without a giant concat."""

        if not folds:
            raise ValueError("cannot publish an empty walk-forward run")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=target.name, suffix=".tmp", dir=target.parent
        )
        os.close(fd)
        writer: pq.ParquetWriter | None = None
        rows = 0
        try:
            for fold_id, signature in folds:
                if not self.has(fold_id, signature):
                    raise RuntimeError(f"walk-forward fold {fold_id} cache is incomplete")
                prediction_path, _ = self._paths(fold_id)
                # ``read_table`` treats parent directories named ``fold=...`` as
                # a Hive dataset and may inject a synthetic partition column.
                # Reading the physical file keeps the published schema exact.
                table = pq.ParquetFile(prediction_path).read()
                table = table.append_column(
                    "coverage_status",
                    pa.array([coverage_status] * table.num_rows, type=pa.string()),
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="snappy"
                    )
                writer.write_table(table)
                rows += table.num_rows
            if writer is not None:
                writer.close()
                writer = None
            os.replace(temporary, target)
        finally:
            if writer is not None:
                writer.close()
            if os.path.exists(temporary):
                os.unlink(temporary)
        return rows
