from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from neural_a_share.config import AppConfig, PathsConfig
from neural_a_share.pages_publish import PagesPublishError, publish_pages_to_git
from neural_a_share.pipeline import NeuralAlphaPipeline


CORE_PAGES = ("index.html", "daily.html", "weekly.html", "publish.html")


def _git(directory: Path, *arguments: str, identity: bool = False) -> str:
    command = ["git", "-C", str(directory)]
    if identity:
        command.extend(["-c", "user.name=Test", "-c", "user.email=test@example.com"])
    command.extend(arguments)
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _write_pages(docs: Path, marker: str) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    padding = marker * 360
    for filename in CORE_PAGES:
        (docs / filename).write_text(
            f"<!doctype html><html><head><title>{filename}</title></head>"
            f"<body>{padding}</body></html>",
            encoding="utf-8",
        )


def _repository(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    work.mkdir()
    _git(work, "init", "-b", "main")
    _git(work, "remote", "add", "origin", str(remote))
    (work / "source.txt").write_text("remote source\n", encoding="utf-8")
    _write_pages(work / "docs", "initial")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "initial", identity=True)
    _git(work, "push", "-u", "origin", "main")
    return work, remote, _git(work, "rev-parse", "HEAD")


def test_pages_push_uses_temporary_index_and_preserves_source_changes(tmp_path) -> None:
    work, _remote, initial_head = _repository(tmp_path)
    (work / "source.txt").write_text("local staged source\n", encoding="utf-8")
    _git(work, "add", "source.txt")
    staged_before = _git(work, "diff", "--cached", "--name-only")
    _write_pages(work / "docs", "updated")

    result = publish_pages_to_git(work / "docs")

    assert result.pushed is True
    assert result.changed is True
    assert _git(work, "rev-parse", "HEAD") == initial_head
    assert _git(work, "diff", "--cached", "--name-only") == staged_before == "source.txt"
    assert _git(work, "show", f"{result.commit_sha}:source.txt") == "remote source"
    assert "updated" in _git(work, "show", f"{result.commit_sha}:docs/daily.html")


def test_invalid_site_is_rejected_before_remote_changes(tmp_path) -> None:
    work, remote, initial_head = _repository(tmp_path)
    (work / "docs" / "weekly.html").write_text("<html>broken</html>", encoding="utf-8")

    with pytest.raises(PagesPublishError, match="invalid weekly.html"):
        publish_pages_to_git(work / "docs")

    remote_head = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=os.environ.copy(),
    ).stdout.strip()
    assert remote_head == initial_head


def test_automatic_pages_failure_does_not_fail_the_pipeline(tmp_path, monkeypatch) -> None:
    paths = PathsConfig(
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        logs_dir=tmp_path / "logs",
        docs_dir=tmp_path / "docs",
    )
    base = AppConfig()
    config = replace(
        base,
        paths=paths,
        reports=replace(base.reports, auto_push_pages=True),
    )
    pipeline = NeuralAlphaPipeline(config)

    def fail_publish():
        raise PagesPublishError("credential unavailable")

    monkeypatch.setattr(pipeline, "publish_pages", fail_publish)

    assert pipeline._auto_publish_pages() is None
    status = json.loads(
        (paths.docs_dir / "pages-publish-status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "FAILED"
    assert status["error_type"] == "PagesPublishError"
    assert "credential unavailable" not in status["detail"]
