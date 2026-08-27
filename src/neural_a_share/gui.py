from __future__ import annotations

import json
import queue
import threading
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import load_config
from .model import ModelRegistry, select_device
from .pipeline import NeuralAlphaPipeline

WALK_FORWARD_SCOPES: dict[str, int | None] = {
    "最近 2 折（快速）": 2,
    "最近 5 折": 5,
    "最近 10 折": 10,
    "全部折（完整范围）": None,
}


def _walk_forward_max_folds(label: str) -> int | None:
    if label not in WALK_FORWARD_SCOPES:
        raise ValueError(f"unknown Walk Forward scope: {label}")
    return WALK_FORWARD_SCOPES[label]


def _model_is_degraded(champion: str | None, model: dict[str, Any]) -> bool:
    """Recover the explicit GUI research mode from persisted model metadata."""

    status = str(model.get("survivorship_status", "")).upper()
    return status == "DEGRADED" or str(champion or "").lower().endswith("-degraded")


def _training_cutoff_text(model: dict[str, Any]) -> str:
    cutoff = str(model.get("training_cutoff", "—"))
    semantics = str(
        model.get("training_cutoff_semantics", "legacy_data_cutoff")
    )
    return cutoff if semantics == "last_train_signal_date" else f"{cutoff} · LEGACY"


def _cuda_text() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA · {torch.cuda.get_device_name(0)}"
        return "CPU fallback · CUDA unavailable"
    except Exception:
        return "CPU fallback · PyTorch check failed"


class NeuralAlphaApp:
    def __init__(self, config_path: str) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.config_path = config_path
        self.config = load_config(config_path)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.root = tk.Tk()
        self.root.title("TickFlow Neural Alpha")
        self.root.geometry("1380x860")
        self.root.minsize(1080, 680)
        self.root.configure(bg="#07111f")
        self.busy = False
        self.status_vars: dict[str, tk.StringVar] = {}
        self.buttons: list[Any] = []
        self.readonly_controls: list[Any] = []
        self.allow_degraded_survivorship = False
        self.walk_forward_max_folds: int | None = 2
        self._survivorship_mode_initialized = False
        self._style()
        self._build()
        self._refresh_status()
        self.root.after(100, self._drain_events)

    def _style(self) -> None:
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        style.configure("TFrame", background="#07111f")
        style.configure("Panel.TFrame", background="#0d1b2d")
        style.configure("TLabel", background="#07111f", foreground="#edf5ff")
        style.configure("Card.TLabel", background="#0d1b2d", foreground="#edf5ff")
        style.configure("Muted.TLabel", background="#0d1b2d", foreground="#8fa5bd")
        style.configure("TButton", padding=(11, 8), background="#11243a", foreground="#edf5ff")
        style.map("TButton", background=[("active", "#173653"), ("disabled", "#132032")])
        style.configure(
            "TCheckbutton",
            background="#07111f",
            foreground="#edf5ff",
            indicatorcolor="#11243a",
        )
        style.map(
            "TCheckbutton",
            background=[("active", "#07111f"), ("disabled", "#07111f")],
            foreground=[("disabled", "#60758c")],
            indicatorcolor=[("selected", "#46d9ff")],
        )
        style.configure("Treeview", background="#0d1b2d", foreground="#edf5ff", fieldbackground="#0d1b2d", rowheight=27)
        style.configure("Treeview.Heading", background="#11243a", foreground="#8fa5bd")
        style.configure("Horizontal.TProgressbar", troughcolor="#11243a", background="#46d9ff")

    def _build(self) -> None:
        header = self.ttk.Frame(self.root)
        header.pack(fill="x", padx=22, pady=(18, 10))
        self.ttk.Label(header, text="TickFlow Neural Alpha", font=("Segoe UI", 24, "bold")).pack(side="left")
        self.ttk.Label(header, text="Pure NN · PIT · Historical OOS", foreground="#46d9ff").pack(side="left", padx=18, pady=(10, 0))

        cards = self.ttk.Frame(self.root)
        cards.pack(fill="x", padx=22, pady=8)
        for key, label in [
            ("latest", "TickFlow 最新日期"),
            ("gpu", "GPU / CUDA"),
            ("model", "模型版本"),
            ("cutoff", "TrainingCutoff"),
            ("ic", "Rolling IC"),
        ]:
            frame = self.ttk.Frame(cards, style="Panel.TFrame", padding=14)
            frame.pack(side="left", fill="x", expand=True, padx=(0, 9))
            self.ttk.Label(frame, text=label, style="Muted.TLabel", font=("Segoe UI", 9)).pack(anchor="w")
            var = self.tk.StringVar(value="—")
            self.status_vars[key] = var
            self.ttk.Label(frame, textvariable=var, style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(5, 0))

        mode_bar = self.ttk.Frame(self.root)
        mode_bar.pack(fill="x", padx=22, pady=(4, 0))
        self.degraded_mode_var = self.tk.BooleanVar(value=False)
        self.degraded_mode_text = self.tk.StringVar()
        mode_toggle = self.ttk.Checkbutton(
            mode_bar,
            text="DEGRADED 研究模式（历史 universe 不可追溯）",
            variable=self.degraded_mode_var,
            command=self._sync_degraded_mode,
        )
        mode_toggle.pack(side="left")
        self.buttons.append(mode_toggle)
        self.ttk.Label(
            mode_bar,
            textvariable=self.degraded_mode_text,
            foreground="#ffb45f",
        ).pack(side="left", padx=14)
        self.walk_forward_scope_var = self.tk.StringVar(
            value="最近 2 折（快速）"
        )
        self.ttk.Label(
            mode_bar,
            text="WF 范围（仅历史 OOS 评估）",
            foreground="#8fa5bd",
        ).pack(side="right", padx=(12, 6))
        self.walk_forward_scope = self.ttk.Combobox(
            mode_bar,
            textvariable=self.walk_forward_scope_var,
            values=tuple(WALK_FORWARD_SCOPES),
            state="readonly",
            width=20,
        )
        self.walk_forward_scope.pack(side="right")
        self.walk_forward_scope.bind(
            "<<ComboboxSelected>>", self._sync_walk_forward_scope
        )
        self.readonly_controls.append(self.walk_forward_scope)
        self._sync_degraded_mode()
        self._sync_walk_forward_scope()

        toolbar = self.ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=22, pady=9)
        actions: list[tuple[str, Callable[[], Any]]] = [
            ("更新 TickFlow", lambda: self._pipeline().update_tickflow(False)),
            ("每日运行", lambda: self._pipeline().daily(False)),
            ("训练模型", self._train_model),
            ("Walk Forward", self._walk_forward),
            ("回测", lambda: self._pipeline().run_backtest()),
            ("日报", lambda: self._pipeline().daily(True)),
            ("周报", lambda: self._pipeline().weekly()),
            ("发布 Pages", lambda: self._pipeline().publish_pages()),
            ("模型管理", self._show_models),
        ]
        for label, callback in actions:
            button = self.ttk.Button(toolbar, text=label, command=lambda cb=callback, name=label: self._run(name, cb))
            button.pack(side="left", padx=(0, 7))
            self.buttons.append(button)
        self.ttk.Button(toolbar, text="打开 Pages 本地文件", command=self._open_report).pack(side="right")

        progress_frame = self.ttk.Frame(self.root)
        progress_frame.pack(fill="x", padx=22, pady=(0, 8))
        self.progress = self.ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_text = self.tk.StringVar(value="就绪")
        self.ttk.Label(progress_frame, textvariable=self.progress_text).pack(side="left", padx=12)

        pane = self.ttk.Panedwindow(self.root, orient="vertical")
        pane.pack(fill="both", expand=True, padx=22, pady=(0, 18))
        top_frame = self.ttk.Frame(pane, style="Panel.TFrame", padding=10)
        log_frame = self.ttk.Frame(pane, style="Panel.TFrame", padding=10)
        pane.add(top_frame, weight=3)
        pane.add(log_frame, weight=2)

        columns = ("rank", "symbol", "name", "a20", "a40", "a60", "alpha")
        self.tree = self.ttk.Treeview(top_frame, columns=columns, show="headings")
        headings = ["NeuralRank", "Symbol", "Name", "Alpha20", "Alpha40", "Alpha60", "NeuralAlpha"]
        widths = [95, 110, 150, 105, 105, 105, 120]
        for column, heading, width in zip(columns, headings, widths):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="e" if column not in {"symbol", "name"} else "w")
        scrollbar = self.ttk.Scrollbar(top_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.log = self.tk.Text(
            log_frame,
            bg="#081522",
            fg="#b8cbe0",
            insertbackground="#edf5ff",
            relief="flat",
            font=("Cascadia Mono", 9),
            wrap="word",
        )
        self.log.pack(fill="both", expand=True)

    def _pipeline(self) -> NeuralAlphaPipeline:
        return NeuralAlphaPipeline(
            load_config(self.config_path),
            progress=lambda message, value: self.events.put(("progress", (message, value))),
        )

    def _sync_degraded_mode(self, enabled: bool | None = None) -> None:
        if enabled is not None:
            self.degraded_mode_var.set(bool(enabled))
        self.allow_degraded_survivorship = bool(self.degraded_mode_var.get())
        if self.allow_degraded_survivorship:
            self.degraded_mode_text.set(
                "已开启：训练与 Walk Forward 会明确标记 DEGRADED（非 strict OOS）"
            )
        else:
            self.degraded_mode_text.set(
                "严格模式：历史早于首份 universe snapshot 时会拒绝运行"
            )

    def _train_model(self) -> Any:
        return self._pipeline().train(
            allow_degraded_survivorship=self.allow_degraded_survivorship
        )

    def _sync_walk_forward_scope(self, _event: Any | None = None) -> None:
        self.walk_forward_max_folds = _walk_forward_max_folds(
            self.walk_forward_scope_var.get()
        )

    def _walk_forward(self) -> Any:
        return self._pipeline().walk_forward(
            max_folds=getattr(self, "walk_forward_max_folds", 2),
            allow_degraded_survivorship=self.allow_degraded_survivorship,
            resume=True,
        )

    def _run(self, name: str, callback: Callable[[], Any]) -> None:
        if self.busy:
            self._append_log("已有任务在运行，请等待。")
            return
        self.busy = True
        self._set_buttons(False)
        self.progress.configure(value=0)
        self.progress_text.set(name)

        def worker() -> None:
            try:
                result = callback()
                self.events.put(("done", (name, result)))
            except Exception:
                self.events.put(("error", (name, traceback.format_exc())))

        threading.Thread(target=worker, name=f"neural-alpha-{name}", daemon=True).start()

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                message, value = payload
                self.progress_text.set(message)
                if value is not None:
                    self.progress.configure(value=max(0, min(100, float(value) * 100)))
                self._append_log(message)
            elif kind == "done":
                name, result = payload
                self._append_log(f"{name} 完成")
                if isinstance(result, pd.DataFrame) and "NeuralRank" in result:
                    self._populate_predictions(result)
                self.busy = False
                self._set_buttons(True)
                self.progress.configure(value=100)
                self.progress_text.set("完成")
                self._refresh_status()
            elif kind == "error":
                name, detail = payload
                self._append_log(f"{name} 失败\n{detail}")
                self.busy = False
                self._set_buttons(True)
                self.progress_text.set("失败；详见日志")
        self.root.after(100, self._drain_events)

    def _set_buttons(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.buttons:
            button.configure(state=state)
        for control in self.readonly_controls:
            control.configure(state="readonly" if enabled else "disabled")

    def _append_log(self, message: str) -> None:
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")

    def _refresh_status(self) -> None:
        registry = ModelRegistry(self.config.paths.models_dir).read()
        champion = registry.get("champion")
        model = registry.get("models", {}).get(champion, {}) if champion else {}
        if not self._survivorship_mode_initialized:
            self._sync_degraded_mode(_model_is_degraded(champion, model))
            self._survivorship_mode_initialized = True
        manifest_path = self.config.paths.cache_dir / "manifests" / "tickflow.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        self.status_vars["latest"].set(str(manifest.get("latest_date", "未更新"))[:10])
        self.status_vars["gpu"].set(_cuda_text())
        self.status_vars["model"].set(champion or "UNTRAINED")
        self.status_vars["cutoff"].set(_training_cutoff_text(model))
        metrics = model.get("metrics", {})
        self.status_vars["ic"].set(" / ".join(f"{metrics.get(f'rank_ic_{h}', float('nan')):.3f}" for h in (20, 40, 60)))
        files = sorted(self.config.paths.predictions_dir.glob("*.parquet"))
        if files:
            self._populate_predictions(pd.read_parquet(files[-1]))

    def _populate_predictions(self, frame: pd.DataFrame) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for _, row in frame.sort_values("NeuralRank").head(100).iterrows():
            values = [
                int(row["NeuralRank"]),
                row["symbol"],
                row.get("name", ""),
                f"{row['Alpha20']:.4%}",
                f"{row['Alpha40']:.4%}",
                f"{row['Alpha60']:.4%}",
                f"{row['NeuralAlpha']:.4%}",
            ]
            self.tree.insert("", "end", values=values)

    def _show_models(self) -> None:
        registry = ModelRegistry(self.config.paths.models_dir)
        data = registry.read()
        window = self.tk.Toplevel(self.root)
        window.title("模型管理")
        window.geometry("820x440")
        window.configure(bg="#07111f")
        tree = self.ttk.Treeview(window, columns=("role", "version", "cutoff", "ic"), show="headings")
        for key, label, width in [("role", "Role", 100), ("version", "Version", 360), ("cutoff", "TrainingCutoff", 130), ("ic", "IC20/40/60", 170)]:
            tree.heading(key, text=label)
            tree.column(key, width=width)
        champion = data.get("champion")
        for version, info in data.get("models", {}).items():
            metrics = info.get("metrics", {})
            ic = "/".join(f"{metrics.get(f'rank_ic_{h}', float('nan')):.3f}" for h in (20, 40, 60))
            tree.insert("", "end", iid=version, values=("CHAMPION" if version == champion else "CHALLENGER", version, _training_cutoff_text(info), ic))
        tree.pack(fill="both", expand=True, padx=12, pady=12)

        def promote() -> None:
            selected = tree.selection()
            if selected:
                registry.promote(selected[0])
                window.destroy()
                self._refresh_status()

        self.ttk.Button(window, text="Promote to Champion", command=promote).pack(pady=(0, 12))

    def _open_report(self) -> None:
        path = (self.config.paths.docs_dir / "index.html").resolve()
        webbrowser.open(path.as_uri())

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(config_path: str = "config/default.yaml") -> None:
    NeuralAlphaApp(config_path).run()
