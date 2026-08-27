from __future__ import annotations

import html
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSS = """
:root{--bg:#07111f;--panel:#0d1b2d;--panel2:#11243a;--text:#edf5ff;--muted:#8fa5bd;--cyan:#46d9ff;--green:#4ce0a2;--red:#ff6f87;--amber:#ffc857;--line:#20364f}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#07111f,#091827 55%,#07111f);color:var(--text);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
a{color:var(--cyan);text-decoration:none}.wrap{max-width:1320px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-bottom:22px}.hero h1{font-size:30px;margin:0}.kicker{color:var(--cyan);letter-spacing:.12em;text-transform:uppercase;font-size:12px}.muted{color:var(--muted)}
.grid{display:grid;gap:14px}.metrics{grid-template-columns:repeat(auto-fit,minmax(155px,1fr))}.card,.panel{background:rgba(13,27,45,.92);border:1px solid var(--line);border-radius:13px;box-shadow:0 12px 32px rgba(0,0,0,.18)}.card{padding:16px}.panel{padding:20px;margin-top:15px}.label{color:var(--muted);font-size:12px;text-transform:uppercase}.value{font-size:24px;font-weight:720;margin-top:5px}.small{font-size:12px;color:var(--muted)}
.badge{display:inline-flex;border-radius:999px;padding:4px 9px;font-size:11px;font-weight:700;border:1px solid}.pass{color:var(--green);border-color:#286d58;background:#11362e}.warn{color:var(--amber);border-color:#745b26;background:#382d15}.fail{color:var(--red);border-color:#783647;background:#391925}.info{color:var(--cyan);border-color:#27617a;background:#102f41}
table{width:100%;border-collapse:collapse;margin-top:8px}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--muted);font-size:11px;text-transform:uppercase;position:sticky;top:0;background:var(--panel)}.table-wrap{overflow:auto;max-height:680px}.two{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}.chart{width:100%;min-height:280px;border-radius:8px;background:#081522}.nav{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0 0}.section-title{font-size:18px;margin:0 0 12px}.zones{grid-template-columns:repeat(4,1fr)}.zone{border-left:3px solid var(--cyan)}.footer{color:var(--muted);font-size:12px;padding:24px 0;text-align:center}@media(max-width:800px){.wrap{padding:16px}.hero{display:block}.two,.zones{grid-template-columns:1fr}.chart{min-height:220px}}
"""
LOCAL_ASSET_REFERENCE = re.compile(r'''(?:src|href)=["'](assets/[^"']+)["']''')


def _fmt(value: Any, percent: bool = False, digits: int = 2) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "N/A"
    if percent:
        return f"{float(value):.{digits}%}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def _badge(value: str) -> str:
    lowered = value.lower()
    style = "pass" if lowered in {"pass", "healthy", "ok"} else "fail" if lowered in {"fail", "error"} else "warn"
    return f'<span class="badge {style}">{html.escape(value)}</span>'


def _page(title: str, data_date: str, body: str, active: str) -> str:
    links = [
        ("首页", "index.html", "index"),
        ("日报", "daily.html", "daily"),
        ("周报", "weekly.html", "weekly"),
        ("发布状态", "publish.html", "publish"),
    ]
    navigation = "".join(
        f'<a class="badge {"info" if key == active else "warn"}" href="{href}">{label}</a>'
        for label, href, key in links
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body><main class="wrap"><header class="hero"><div><div class="kicker">TickFlow · Pure Neural Alpha</div><h1>{html.escape(title)}</h1><div class="muted">数据严格截止：{html.escape(data_date)}</div></div><nav class="nav">{navigation}</nav></header>{body}<footer class="footer">研究用途 · 信号在 t 日收盘后产生，最早 t+1 成交 · 页面不依赖外部 CDN</footer></main></body></html>"""


def _metrics_cards(items: Iterable[tuple[str, Any, bool]]) -> str:
    cards = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{_fmt(value, percent)}</div></div>'
        for label, value, percent in items
    )
    return f'<section class="grid metrics">{cards}</section>'


def _table(frame: pd.DataFrame, columns: list[tuple[str, str]], limit: int | None = None) -> str:
    if frame is None or frame.empty:
        return '<div class="muted">暂无成熟数据</div>'
    shown = frame.head(limit) if limit else frame
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows = []
    for _, row in shown.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                rendered = _fmt(
                    value,
                    percent=(
                        key.lower().endswith("return")
                        or key.startswith("Alpha")
                        or key in {"NeuralAlpha", "FeatureCoverage"}
                    ),
                )
            else:
                rendered = html.escape(str(value))
            cells.append(f"<td>{rendered}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _save_chart(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="#081522")
    plt.close(fig)


def chart_nav(nav: pd.DataFrame, benchmark_nav: pd.DataFrame | None, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4.2), dpi=130)
    fig.patch.set_facecolor("#081522")
    axis.set_facecolor("#081522")
    if nav is not None and not nav.empty:
        normalized = nav["nav"] / nav["nav"].iloc[0]
        axis.plot(pd.to_datetime(nav["trade_date"]), normalized, color="#46d9ff", lw=2, label="Strategy")
    if benchmark_nav is not None and not benchmark_nav.empty:
        column = "nav" if "nav" in benchmark_nav else "close"
        normalized = benchmark_nav[column] / benchmark_nav[column].iloc[0]
        axis.plot(pd.to_datetime(benchmark_nav["trade_date"]), normalized, color="#ffc857", lw=1.6, label="CSI300")
    axis.grid(color="#20364f", alpha=.55)
    axis.tick_params(colors="#8fa5bd")
    for spine in axis.spines.values():
        spine.set_color("#20364f")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles,
            labels,
            facecolor="#0d1b2d",
            edgecolor="#20364f",
            labelcolor="#edf5ff",
        )
    axis.set_title("Strategy NAV vs CSI300", color="#edf5ff", loc="left", weight="bold")
    _save_chart(fig, path)


def chart_ic(rolling_ic: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4.2), dpi=130)
    fig.patch.set_facecolor("#081522")
    axis.set_facecolor("#081522")
    colors = {20: "#46d9ff", 40: "#4ce0a2", 60: "#ffc857"}
    if rolling_ic is not None and not rolling_ic.empty:
        for horizon in (20, 40, 60):
            column = f"rolling_ic_{horizon}"
            if column in rolling_ic:
                axis.plot(pd.to_datetime(rolling_ic["trade_date"]), rolling_ic[column], color=colors[horizon], label=f"IC{horizon}")
    axis.axhline(0, color="#8fa5bd", lw=.8)
    axis.grid(color="#20364f", alpha=.55)
    axis.tick_params(colors="#8fa5bd")
    for spine in axis.spines.values():
        spine.set_color("#20364f")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles,
            labels,
            facecolor="#0d1b2d",
            edgecolor="#20364f",
            labelcolor="#edf5ff",
        )
    axis.set_title("Rolling Rank IC", color="#edf5ff", loc="left", weight="bold")
    _save_chart(fig, path)


def chart_quantiles(quantiles: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.2), dpi=130)
    fig.patch.set_facecolor("#081522")
    axis.set_facecolor("#081522")
    if quantiles is not None and not quantiles.empty:
        axis.bar(quantiles["quantile"].astype(str), quantiles["mean"], color="#46d9ff")
    axis.axhline(0, color="#8fa5bd", lw=.8)
    axis.grid(axis="y", color="#20364f", alpha=.55)
    axis.tick_params(colors="#8fa5bd")
    for spine in axis.spines.values():
        spine.set_color("#20364f")
    axis.set_title("Quantile Monotonicity", color="#edf5ff", loc="left", weight="bold")
    _save_chart(fig, path)


@dataclass
class ReportContext:
    data_date: str
    tickflow_status: str
    model_version: str
    training_cutoff: str
    predictions: pd.DataFrame
    generated_at: str = ""
    prediction_fingerprint: str = ""
    prediction_rows: int = 0
    training_cutoff_semantics: str = "legacy_data_cutoff"
    rolling_ic: pd.DataFrame | None = None
    nav: pd.DataFrame | None = None
    benchmark_nav: pd.DataFrame | None = None
    quantiles: pd.DataFrame | None = None
    metrics: Mapping[str, Any] | None = None
    research: Mapping[str, Any] | None = None
    quality: Mapping[str, Any] | None = None


class StaticReportPublisher:
    def __init__(self, docs_dir: str | Path, title: str = "TickFlow Neural Alpha") -> None:
        self.docs_dir = Path(docs_dir)
        self.title = title

    def publish(self, context: ReportContext, include_weekly: bool = True) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="report-staging-", dir=self.docs_dir.parent))
        try:
            assets = staging / "assets" / "generated" / context.data_date
            chart_nav(context.nav if context.nav is not None else pd.DataFrame(), context.benchmark_nav, assets / "nav.svg")
            chart_ic(context.rolling_ic if context.rolling_ic is not None else pd.DataFrame(), assets / "rolling_ic.svg")
            chart_quantiles(context.quantiles if context.quantiles is not None else pd.DataFrame(), assets / "quantiles.svg")
            daily = self._daily(context)
            existing_weekly = self.docs_dir / "weekly.html"
            if include_weekly or not existing_weekly.exists():
                weekly = self._weekly(context)
            else:
                previous = existing_weekly.read_text(encoding="utf-8")
                # A fresh clone can contain a short placeholder weekly page.
                # DAILY must not fail because an unrelated old artifact is
                # incomplete; safely regenerate it from the current context.
                weekly = (
                    previous
                    if _valid_existing_document(previous, self.docs_dir)
                    else self._weekly(context)
                )
            index = self._index(context)
            publish_status = self._publish_status(context)
            files = {
                "daily.html": daily,
                "weekly.html": weekly,
                "index.html": index,
                "publish.html": publish_status,
            }
            for filename, content in files.items():
                path = staging / filename
                path.write_text(content, encoding="utf-8")
                if not _valid_document(content):
                    raise ValueError(f"refusing to publish invalid {filename}")
            archive = staging / "reports" / context.data_date
            archive.mkdir(parents=True, exist_ok=True)
            (archive / "daily.html").write_text(daily.replace('href="index.html"', 'href="../../index.html"').replace('href="daily.html"', 'href="../../daily.html"').replace('href="weekly.html"', 'href="../../weekly.html"').replace('href="publish.html"', 'href="../../publish.html"'), encoding="utf-8")
            (archive / "weekly.html").write_text(weekly.replace('href="index.html"', 'href="../../index.html"').replace('href="daily.html"', 'href="../../daily.html"').replace('href="weekly.html"', 'href="../../weekly.html"').replace('href="publish.html"', 'href="../../publish.html"').replace('src="assets/', 'src="../../assets/'), encoding="utf-8")
            self._commit_staging(staging, context.data_date, files.keys())
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _commit_staging(self, staging: Path, data_date: str, pages: Iterable[str]) -> None:
        generated = self.docs_dir / "assets" / "generated" / data_date
        generated.parent.mkdir(parents=True, exist_ok=True)
        if not generated.exists():
            shutil.copytree(staging / "assets" / "generated" / data_date, generated)
        archive = self.docs_dir / "reports" / data_date
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            shutil.rmtree(archive)
        shutil.copytree(staging / "reports" / data_date, archive)
        # Replace the entry points only after every page and chart passed validation.
        for filename in pages:
            temporary = self.docs_dir / f".{filename}.new"
            shutil.copy2(staging / filename, temporary)
            os.replace(temporary, self.docs_dir / filename)
        (self.docs_dir / ".nojekyll").touch(exist_ok=True)

    def _daily(self, context: ReportContext) -> str:
        latest = (
            context.predictions.sort_values("NeuralRank").head(30)
            if context.predictions is not None
            and "NeuralRank" in context.predictions
            else pd.DataFrame()
        )
        research = dict(context.research or {})
        cutoff_label = (
            "Training Cutoff"
            if context.training_cutoff_semantics == "last_train_signal_date"
            else "Legacy Data Cutoff"
        )
        cards = _metrics_cards(
            [
                ("TickFlow", context.tickflow_status, False),
                ("Current Model", context.model_version, False),
                (cutoff_label, context.training_cutoff, False),
                ("Stock Universe", len(context.predictions) if context.predictions is not None else 0, False),
                ("Signal Source", "Champion MLP", False),
                ("Min Feature Coverage", research.get("min_feature_coverage", "N/A"), False),
                ("Generated At", context.generated_at or "N/A", False),
                ("Prediction ID", context.prediction_fingerprint or "N/A", False),
            ]
        )
        table = _table(
            latest,
            [
                ("NeuralRank", "Rank"),
                ("symbol", "Symbol"),
                ("name", "Name"),
                ("Alpha20", "Alpha20"),
                ("Alpha40", "Alpha40"),
                ("Alpha60", "Alpha60"),
                ("NeuralAlpha", "NeuralAlpha"),
                ("FeatureCoverage", "Feature Coverage"),
            ],
        )
        body = cards + f'<section class="panel"><h2 class="section-title">Neural Stock Top 30</h2><div class="small">股票资格与数据完整度只决定可研究范围；范围内排序 100% 由 Champion 多任务 MLP 输出。Walk Forward folds 仅用于历史 OOS 评估，不直接产生本页每日排名。</div>{table}</section>'
        return _page(f"{self.title} · 日报", context.data_date, body, "daily")

    def _weekly(self, context: ReportContext) -> str:
        metrics = dict(context.metrics or {})
        research = dict(context.research or {})
        quality = dict(context.quality or {})
        cards = _metrics_cards(
            [
                ("Total Return", metrics.get("total_return"), True),
                ("Sharpe", metrics.get("sharpe"), False),
                ("Max Drawdown", metrics.get("max_drawdown"), True),
                ("Calmar", metrics.get("calmar"), False),
                ("Turnover", metrics.get("turnover"), False),
                ("Trading Costs", metrics.get("trading_costs"), False),
            ]
        )
        zones = "".join(
            f'<div class="card zone"><div class="label">{name}</div><div class="value">{_badge(status)}</div><div class="small">{html.escape(detail)}</div></div>'
            for name, status, detail in [
                ("IN-SAMPLE", research.get("in_sample_status", "RECORDED"), research.get("train_period", "N/A")),
                ("VALIDATION", research.get("validation_status", "RECORDED"), research.get("validation_period", "N/A")),
                ("HISTORICAL OOS", research.get("historical_oos_status", "PENDING"), research.get("test_period", "N/A")),
                ("FORWARD SHADOW OOS", research.get("shadow_status", "PENDING"), research.get("shadow_period", "N/A")),
            ]
        )
        audit_rows = pd.DataFrame(
            [
                ("Champion / Challenger", research.get("champion_challenger", "N/A")),
                ("Train / Validation / Test", research.get("split_summary", "N/A")),
                ("Purge / Embargo", research.get("purge_embargo", "N/A")),
                ("Mature Labels", research.get("mature_labels", "N/A")),
                ("Data Quality", quality.get("status", context.tickflow_status)),
                ("PIT Status", quality.get("pit_status", "N/A")),
                ("Survivorship Status", quality.get("survivorship_status", "N/A")),
                ("ICIR 20 / 40 / 60", research.get("icir", "N/A")),
                ("Newey-West t 20 / 40 / 60", research.get("newey_west_t", "N/A")),
                ("Yearly / Regime IC", research.get("yearly_regime_ic", "N/A")),
                ("Top-K Performance", research.get("topk_performance", "N/A")),
                ("Selection Universe", research.get("selection_universe", "N/A")),
                ("Min Feature Coverage", research.get("min_feature_coverage", "N/A")),
                ("Prediction Generated At", context.generated_at or "N/A"),
                ("Prediction Rows", context.prediction_rows),
                ("Prediction Fingerprint", context.prediction_fingerprint or "N/A"),
                (
                    "Training Data Cutoff",
                    research.get("training_data_cutoff", "N/A"),
                ),
                (
                    "TrainingCutoff Semantics",
                    (
                        "LAST TRAIN SIGNAL DATE"
                        if context.training_cutoff_semantics
                        == "last_train_signal_date"
                        else "LEGACY CHECKPOINT · STORED VALUE WAS DATA CUTOFF"
                    ),
                ),
                ("IC Decay", research.get("ic_decay", "N/A")),
            ],
            columns=["item", "value"],
        )
        charts = f'<section class="grid two"><div class="panel"><img class="chart" src="assets/generated/{context.data_date}/nav.svg" alt="Strategy NAV vs CSI300"></div><div class="panel"><img class="chart" src="assets/generated/{context.data_date}/rolling_ic.svg" alt="Rolling Rank IC"></div></section><section class="grid two"><div class="panel"><img class="chart" src="assets/generated/{context.data_date}/quantiles.svg" alt="Quantile monotonicity"></div><div class="panel"><h2 class="section-title">Research Audit</h2>{_table(audit_rows, [("item","Item"),("value","Value")])}</div></section>'
        body = cards + f'<section class="grid zones">{zones}</section>' + charts
        return _page(f"{self.title} · 周报", context.data_date, body, "weekly")

    def _index(self, context: ReportContext) -> str:
        latest = (
            context.predictions.sort_values("NeuralRank").head(30)
            if context.predictions is not None
            and "NeuralRank" in context.predictions
            else pd.DataFrame()
        )
        research = dict(context.research or {})
        cutoff_label = (
            "TrainingCutoff"
            if context.training_cutoff_semantics == "last_train_signal_date"
            else "Legacy Data Cutoff"
        )
        cards = _metrics_cards(
            [
                ("最新数据", context.data_date, False),
                ("TickFlow 状态", context.tickflow_status, False),
                ("当前模型", context.model_version, False),
                (cutoff_label, context.training_cutoff, False),
                ("生成时间", context.generated_at or "N/A", False),
                ("Prediction ID", context.prediction_fingerprint or "N/A", False),
            ]
        )
        history = []
        report_dates = {context.data_date}
        reports = self.docs_dir / "reports"
        if reports.exists():
            report_dates.update(path.name for path in reports.iterdir() if path.is_dir())
        for date in sorted(report_dates, reverse=True)[:40]:
            history.append(f'<a class="badge info" href="reports/{date}/daily.html">{html.escape(date)}</a>')
        body = cards
        body += f'<section class="panel"><h2 class="section-title">Neural Stock Top 30</h2><div class="small">Signal Source: Champion MLP · Selection Universe: {html.escape(str(research.get("selection_universe", "N/A")))}</div>{_table(latest, [("NeuralRank","Rank"),("symbol","Symbol"),("name","Name"),("Alpha20","Alpha20"),("Alpha40","Alpha40"),("Alpha60","Alpha60"),("NeuralAlpha","NeuralAlpha"),("FeatureCoverage","Feature Coverage")])}</section>'
        body += f'<section class="panel"><h2 class="section-title">历史报告</h2><div class="nav">{"".join(history) or "<span class=muted>首份报告生成后显示</span>"}</div></section>'
        return _page(self.title, context.data_date, body, "index")

    def _publish_status(self, context: ReportContext) -> str:
        quality = dict(context.quality or {})
        research = dict(context.research or {})
        rows = pd.DataFrame(
            [
                ("Local page generation", "PASS"),
                ("Core page validation", "PASS"),
                ("Data cutoff", context.data_date),
                ("Model", context.model_version),
                ("Prediction generated at", context.generated_at or "N/A"),
                ("Prediction rows", context.prediction_rows),
                ("Prediction fingerprint", context.prediction_fingerprint or "N/A"),
                ("Signal source", "Champion MLP"),
                ("Selection universe", research.get("selection_universe", "N/A")),
                ("PIT status", quality.get("pit_status", "N/A")),
                (
                    "GitHub publication",
                    (
                        "ENABLED after DAILY / WEEKLY → "
                        + str(research.get("pages_target", "origin/main"))
                        if research.get("pages_auto_push")
                        else "DISABLED · GUI/CLI manual publish remains available"
                    ),
                ),
                ("Deployment safety", "CI and complete artifact validation before Pages switch"),
            ],
            columns=["item", "value"],
        )
        body = _metrics_cards(
            [
                ("Local Build", "PASS", False),
                ("Core Pages", 4, False),
                ("Data Date", context.data_date, False),
                ("Remote Deploy", "GitHub Actions", False),
            ]
        )
        body += f'<section class="panel"><h2 class="section-title">Pages 发布链路</h2><div class="small">DAILY / WEEKLY 原子生成 → 本地完整性校验 → 仅覆盖远端 main 的 docs/ → CI → GitHub Pages。任何生成、推送、CI 或部署失败都保留上一版健康页面。</div>{_table(rows, [("item","Item"),("value","Value")])}</section>'
        return _page(f"{self.title} · 发布状态", context.data_date, body, "publish")


def _valid_document(content: str) -> bool:
    return len(content) >= 800 and "</html>" in content.lower()


def _valid_existing_document(content: str, docs_dir: Path) -> bool:
    if not _valid_document(content):
        return False
    root = docs_dir.resolve()
    for reference in LOCAL_ASSET_REFERENCE.findall(content):
        clean = reference.split("?", 1)[0].split("#", 1)[0]
        target = (root / clean).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return False
        if not target.exists():
            return False
    return True


def write_status_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
