from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PublishProgress = Callable[[str], None]
_SAFE_GIT_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")
_CORE_PAGES = ("index.html", "daily.html", "weekly.html", "publish.html")
_LOCAL_ASSET = re.compile(r'''(?:src|href)=["']([^"']+)["']''')
_URL_CREDENTIAL = re.compile(r"(?i)(https?://)[^/@\s]+@")
_GITHUB_TOKEN = re.compile(r"(?i)(?:gh[opsu]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)")


class PagesPublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class PagesPublishResult:
    commit_sha: str
    changed: bool
    pushed: bool
    remote: str
    branch: str
    pages: int


def _safe_error_detail(value: str) -> str:
    redacted = _URL_CREDENTIAL.sub(r"\1***@", value)
    redacted = _GITHUB_TOKEN.sub("***", redacted)
    return redacted[:2_000]


def validate_pages(docs_dir: str | Path) -> tuple[Path, ...]:
    docs = Path(docs_dir).resolve()
    failures: list[str] = []
    paths = tuple(docs / name for name in _CORE_PAGES)
    for path in paths:
        if not path.exists():
            failures.append(f"missing {path.name}")
            continue
        content = path.read_text(encoding="utf-8")
        if len(content) < 300 or "</html>" not in content.lower():
            failures.append(f"invalid {path.name}")
            continue
        if "http://" in content or "https://cdn" in content:
            failures.append(f"external CDN/reference in {path.name}")
        for reference in _LOCAL_ASSET.findall(content):
            if reference.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = reference.split("?", 1)[0].split("#", 1)[0]
            if not clean or not clean.startswith("assets/"):
                continue
            target = (docs / clean).resolve()
            try:
                target.relative_to(docs)
            except ValueError:
                failures.append(f"unsafe asset path in {path.name}: {reference}")
                continue
            if not target.exists():
                failures.append(f"missing asset for {path.name}: {clean}")
    if failures:
        raise PagesPublishError("Pages validation failed: " + "; ".join(failures))
    return paths


def _git(
    repo: Path,
    arguments: list[str],
    env: dict[str, str] | None = None,
    timeout: int = 180,
    allow_failure: bool = False,
) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
    )
    if completed.returncode and not allow_failure:
        detail = _safe_error_detail(
            completed.stderr.strip() or completed.stdout.strip()
        )
        raise PagesPublishError(
            f"git {' '.join(arguments[:2])} failed: {detail}"
        )
    if completed.returncode:
        return ""
    return completed.stdout.strip()


def publish_pages_to_git(
    docs_dir: str | Path,
    remote: str = "origin",
    branch: str = "main",
    message: str | None = None,
    progress: PublishProgress | None = None,
) -> PagesPublishResult:
    """Commit only local docs/ over the latest remote tree and fast-forward push.

    A temporary Git index is used, so staged/unstaged source changes and the
    caller's current branch are never modified.  The commit is parented to the
    fetched remote head; a concurrent remote update makes the non-force push
    fail safely instead of overwriting it.
    """

    if not _SAFE_GIT_NAME.fullmatch(remote) or remote.startswith("-"):
        raise PagesPublishError("unsafe Git remote name")
    if not _SAFE_GIT_NAME.fullmatch(branch) or branch.startswith("-"):
        raise PagesPublishError("unsafe Git branch name")
    docs = Path(docs_dir).resolve()
    validated = validate_pages(docs)
    notify = progress or (lambda _message: None)
    repo_text = _git(docs.parent, ["rev-parse", "--show-toplevel"])
    repo = Path(repo_text).resolve()
    try:
        docs_relative = docs.relative_to(repo)
    except ValueError as exc:
        raise PagesPublishError("docs directory is outside the Git repository") from exc
    if docs_relative == Path(".") or ".git" in docs_relative.parts:
        raise PagesPublishError("unsafe docs directory")

    notify(f"Pages validation PASS · {len(validated)} core pages")
    notify(f"Fetching {remote}/{branch}")
    _git(repo, ["fetch", remote, branch])
    remote_ref = f"refs/remotes/{remote}/{branch}"
    parent = _git(repo, ["rev-parse", "--verify", remote_ref])

    descriptor, index_name = tempfile.mkstemp(prefix="tickflow-pages-index-")
    os.close(descriptor)
    os.unlink(index_name)
    index_path = Path(index_name)
    index_env = os.environ.copy()
    index_env["GIT_INDEX_FILE"] = str(index_path)
    try:
        _git(repo, ["read-tree", parent], env=index_env)
        _git(
            repo,
            ["add", "-A", "-f", "--", str(docs_relative)],
            env=index_env,
        )
        tree = _git(repo, ["write-tree"], env=index_env)
        parent_tree = _git(repo, ["rev-parse", f"{parent}^{{tree}}"])
        if tree == parent_tree:
            notify("Pages already match remote main; nothing to push")
            return PagesPublishResult(
                commit_sha=parent,
                changed=False,
                pushed=False,
                remote=remote,
                branch=branch,
                pages=len(validated),
            )

        identity_env = index_env.copy()
        configured_name = _git(
            repo,
            ["config", "--get", "user.name"],
            allow_failure=True,
        )
        configured_email = _git(
            repo,
            ["config", "--get", "user.email"],
            allow_failure=True,
        )
        identity_env.setdefault(
            "GIT_AUTHOR_NAME",
            configured_name or "TickFlow Pages Bot",
        )
        identity_env.setdefault(
            "GIT_COMMITTER_NAME", identity_env["GIT_AUTHOR_NAME"]
        )
        identity_env.setdefault(
            "GIT_AUTHOR_EMAIL",
            configured_email or "tickflow-pages@users.noreply.github.com",
        )
        identity_env.setdefault(
            "GIT_COMMITTER_EMAIL", identity_env["GIT_AUTHOR_EMAIL"]
        )
        commit_message = message or "reports: publish TickFlow Pages"
        commit = _git(
            repo,
            ["commit-tree", tree, "-p", parent, "-m", commit_message],
            env=identity_env,
        )
        notify(f"Pushing Pages commit {commit[:12]} to {remote}/{branch}")
        _git(
            repo,
            ["push", remote, f"{commit}:refs/heads/{branch}"],
            timeout=300,
        )
        return PagesPublishResult(
            commit_sha=commit,
            changed=True,
            pushed=True,
            remote=remote,
            branch=branch,
            pages=len(validated),
        )
    finally:
        index_path.unlink(missing_ok=True)
