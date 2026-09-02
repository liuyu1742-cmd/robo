"""Recordable GUI/CLI for reproducible full-chain action-parser testing."""

from __future__ import annotations

import argparse
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.action_parser_demo_cases import generate_demo_cases
from tools.action_parser_demo_output import CATEGORY_ZH, DemoRunOutput
from tools.action_parser_demo_runner import (
    DemoBatchRunner,
    DemoResult,
    FAILURE_CATEGORIES,
    summarize_results,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "models" / "action_transformer_project_generation_semantic.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "action_parser_demo"


@dataclass(frozen=True)
class RunConfig:
    count: int
    seed: int
    checkpoint: Path
    device: str | None
    output_dir: Path
    beam_width: int
    max_actions: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--no-gui", action="store_true", help="命令行执行，不打开图形页面")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    return args


def run_batch(
    config: RunConfig,
    emit: Callable[[dict[str, Any]], None],
    stop_event: threading.Event,
    *,
    runner_factory: type[DemoBatchRunner] = DemoBatchRunner,
) -> tuple[dict[str, Any], Path]:
    cases = generate_demo_cases(config.count, config.seed)
    output = DemoRunOutput(config.output_dir, seed=config.seed, requested=config.count)
    output.write_generated_cases(cases)
    emit({"type": "generated", "count": len(cases), "run_dir": str(output.run_dir)})
    runner = runner_factory(
        config.checkpoint,
        device=config.device,
        beam_width=config.beam_width,
        max_actions=config.max_actions,
    )
    passed = 0

    def on_result(result: DemoResult, completed: int) -> None:
        nonlocal passed
        output.append_result(result)
        passed += int(result.passed)
        emit({"type": "result", "result": result.to_dict()})
        emit({
            "type": "progress",
            "completed": completed,
            "total": config.count,
            "passed": passed,
            "failed": completed - passed,
            "accuracy_percent": round(passed / completed * 100.0, 2),
        })

    results = runner.run(cases, on_result=on_result, stop_requested=stop_event.is_set)
    summary = summarize_results(results, requested=config.count, seed=config.seed)
    summary["run_dir"] = str(output.run_dir)
    output.finalize(summary)
    emit({"type": "summary", "summary": summary, "run_dir": str(output.run_dir)})
    return summary, output.run_dir


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        count=args.count,
        seed=args.seed,
        checkpoint=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        beam_width=args.beam_width,
        max_actions=args.max_actions,
    )


def run_cli(config: RunConfig) -> int:
    def emit(event: dict[str, Any]) -> None:
        if event["type"] == "generated":
            print(f"已生成 {event['count']} 条可复现口语化测试命令。")
        elif event["type"] == "progress":
            completed = event["completed"]
            if completed == 1 or completed % 10 == 0 or completed == event["total"]:
                print(
                    f"进度 {completed}/{event['total']}，通过 {event['passed']}，"
                    f"未通过 {event['failed']}，当前正确率 {event['accuracy_percent']:.2f}%",
                    flush=True,
                )
        elif event["type"] == "summary":
            summary = event["summary"]
            print(
                f"测试完成：{summary['passed']}/{summary['completed']} 通过，"
                f"正确率 {summary['accuracy_percent']:.2f}%"
            )
            print(f"结果目录：{event['run_dir']}")

    try:
        run_batch(config, emit, threading.Event())
        return 0
    except Exception as exc:
        print(f"批次级错误：{type(exc).__name__}: {exc}")
        return 1


class DemoApp:
    def __init__(self, config: RunConfig) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("动作解析完整链路测试演示（500条可复现口语命令）")
        self.root.geometry("1480x900")
        self.root.minsize(1120, 720)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.running = False
        self.last_run_dir: Path | None = None
        self.all_rows: list[dict[str, Any]] = []
        self.count_var = tk.StringVar(value=str(config.count))
        self.seed_var = tk.StringVar(value=str(config.seed))
        self.checkpoint_var = tk.StringVar(value=str(config.checkpoint))
        self.device_var = tk.StringVar(value=config.device or "自动")
        self.output_var = tk.StringVar(value=str(config.output_dir))
        self.filter_var = tk.StringVar(value="全部")
        self.current_var = tk.StringVar(value="尚未开始")
        self.expected_var = tk.StringVar(value="-")
        self.actual_var = tk.StringVar(value="-")
        self.actions_var = tk.StringVar(value="-")
        self.progress_var = tk.DoubleVar(value=0)
        self.summary_vars = {
            key: tk.StringVar(value="0")
            for key in ("completed", "passed", "failed", "accuracy")
        }
        self.failure_vars = {name: tk.StringVar(value="0") for name in FAILURE_CATEGORIES}
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        ttk = self.ttk
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)
        params = ttk.LabelFrame(root, text="测试参数", padding=10)
        params.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        for col in (1, 3, 5):
            params.columnconfigure(col, weight=1)
        items = [
            ("测试数量", self.count_var, 0, 0), ("随机种子", self.seed_var, 0, 2),
            ("运行设备", self.device_var, 0, 4), ("模型检查点", self.checkpoint_var, 1, 0),
            ("日志输出目录", self.output_var, 2, 0),
        ]
        for label, variable, row, col in items:
            ttk.Label(params, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=4)
            entry = ttk.Entry(params, textvariable=variable)
            entry.grid(row=row, column=col + 1, sticky="ew", padx=4, pady=4,
                       columnspan=5 if row > 0 else 1)
        buttons = ttk.Frame(root, padding=(10, 4))
        buttons.grid(row=1, column=0, sticky="ew")
        self.generate_button = ttk.Button(buttons, text="生成测试集预览", command=self._preview)
        self.start_button = ttk.Button(buttons, text="开始完整链路测试", command=self._start)
        self.stop_button = ttk.Button(buttons, text="完成当前用例后停止", command=self._stop, state="disabled")
        self.open_button = ttk.Button(buttons, text="打开结果目录", command=self._open_output)
        for button in (self.generate_button, self.start_button, self.stop_button, self.open_button):
            button.pack(side="left", padx=(0, 8))

        live = ttk.LabelFrame(root, text="实时执行状态", padding=8)
        live.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        live.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate((
            ("当前命令", self.current_var), ("期望结果", self.expected_var),
            ("实际结果", self.actual_var), ("输出动作序列", self.actions_var),
        )):
            ttk.Label(live, text=label, width=12).grid(row=row, column=0, sticky="nw", pady=2)
            ttk.Label(live, textvariable=variable, wraplength=1250).grid(row=row, column=1, sticky="w", pady=2)
        ttk.Progressbar(live, variable=self.progress_var, maximum=100).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(8, 2)
        )

        summary = ttk.Frame(root, padding=(10, 4))
        summary.grid(row=3, column=0, sticky="ew")
        for label, key in (("已完成", "completed"), ("通过", "passed"),
                           ("未通过", "failed"), ("正确率", "accuracy")):
            box = ttk.LabelFrame(summary, text=label, padding=(18, 7))
            box.pack(side="left", padx=(0, 8))
            ttk.Label(box, textvariable=self.summary_vars[key], font=("Microsoft YaHei UI", 16, "bold")).pack()
        failure_box = ttk.LabelFrame(summary, text="失败原因统计", padding=6)
        failure_box.pack(side="left", fill="x", expand=True)
        for index, name in enumerate(FAILURE_CATEGORIES):
            text = CATEGORY_ZH[name]
            ttk.Label(failure_box, text=f"{text}:").grid(row=index // 6, column=(index % 6) * 2, sticky="e", padx=(5, 1))
            ttk.Label(failure_box, textvariable=self.failure_vars[name], width=3).grid(row=index // 6, column=(index % 6) * 2 + 1, sticky="w")

        table_box = ttk.LabelFrame(root, text="逐条测试结果", padding=6)
        table_box.grid(row=4, column=0, sticky="nsew", padx=10, pady=(5, 10))
        table_box.columnconfigure(0, weight=1)
        table_box.rowconfigure(1, weight=1)
        filter_row = ttk.Frame(table_box)
        filter_row.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(filter_row, text="显示：").pack(side="left")
        combo = ttk.Combobox(filter_row, textvariable=self.filter_var,
                             values=("全部", "仅通过", "仅未通过"), state="readonly", width=12)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_table())
        columns = ("id", "instruction", "expected", "actual", "status", "failure")
        self.tree = ttk.Treeview(table_box, columns=columns, show="headings")
        headings = ("用例ID", "口语化命令", "期望任务/物体/操作", "实际任务/物体/操作", "判定", "失败类型")
        widths = (90, 440, 280, 280, 70, 130)
        for name, heading, width in zip(columns, headings, widths):
            self.tree.heading(name, text=heading)
            self.tree.column(name, width=width, anchor="w")
        scroll = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

    def _read_config(self) -> RunConfig:
        device = self.device_var.get().strip()
        return RunConfig(
            count=int(self.count_var.get()), seed=int(self.seed_var.get()),
            checkpoint=Path(self.checkpoint_var.get()),
            device=None if device in ("", "自动") else device,
            output_dir=Path(self.output_var.get()), beam_width=3, max_actions=12,
        )

    def _preview(self) -> None:
        from tkinter import messagebox
        try:
            config = self._read_config()
            cases = generate_demo_cases(config.count, config.seed)
        except Exception as exc:
            messagebox.showerror("生成失败", str(exc))
            return
        self.current_var.set(cases[0].instruction if cases else "-")

    def _start(self) -> None:
        from tkinter import messagebox
        if self.running:
            return
        try:
            config = self._read_config()
            if config.count <= 0:
                raise ValueError("测试数量必须大于 0")
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.running = True
        self.stop_event.clear()
        self.all_rows.clear()
        self._refresh_table()
        self.progress_var.set(0)
        for variable in self.summary_vars.values():
            variable.set("0")
        for variable in self.failure_vars.values():
            variable.set("0")
        self.start_button.configure(state="disabled")
        self.generate_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        def worker() -> None:
            try:
                run_batch(config, self.events.put, self.stop_event)
            except Exception as exc:
                self.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

        threading.Thread(target=worker, daemon=True).start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.stop_button.configure(state="disabled")

    def _poll_events(self) -> None:
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "generated":
            self.last_run_dir = Path(event["run_dir"])
        elif kind == "result":
            row = event["result"]
            self.all_rows.append(row)
            self._show_result(row)
            self._insert_row(row)
        elif kind == "progress":
            self.progress_var.set(event["completed"] / event["total"] * 100)
            self.summary_vars["completed"].set(str(event["completed"]))
            self.summary_vars["passed"].set(str(event["passed"]))
            self.summary_vars["failed"].set(str(event["failed"]))
            self.summary_vars["accuracy"].set(f"{event['accuracy_percent']:.2f}%")
        elif kind == "summary":
            self.last_run_dir = Path(event["run_dir"])
            self._finish(event["summary"])
        elif kind == "error":
            from tkinter import messagebox
            messagebox.showerror("批次运行失败", event["message"])
            self._set_idle()

    def _show_result(self, row: dict[str, Any]) -> None:
        expected, actual = row["expected"], row["actual"]
        self.current_var.set(f"{row['case_id']} {row['instruction']}")
        self.expected_var.set(f"{expected.get('task')} / {expected.get('object')} / {expected.get('operation')}")
        self.actual_var.set(f"{actual.get('task', '-')} / {actual.get('object', '-')} / {actual.get('operation', '-')}")
        self.actions_var.set(" → ".join(actual.get("actions") or []) or "未输出动作序列")
        if row.get("failure_category"):
            name = row["failure_category"]
            self.failure_vars[name].set(str(int(self.failure_vars[name].get()) + 1))

    def _insert_row(self, row: dict[str, Any]) -> None:
        mode = self.filter_var.get()
        if mode == "仅通过" and not row["passed"]:
            return
        if mode == "仅未通过" and row["passed"]:
            return
        expected, actual = row["expected"], row["actual"]
        compact = lambda value: " / ".join(str(value.get(key, "-")) for key in ("task", "object", "operation"))
        self.tree.insert("", "end", values=(
            row["case_id"], row["instruction"], compact(expected), compact(actual),
            "通过" if row["passed"] else "未通过", CATEGORY_ZH.get(row.get("failure_category"), ""),
        ))
        children = self.tree.get_children()
        if children:
            self.tree.see(children[-1])

    def _refresh_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for row in self.all_rows:
            self._insert_row(row)

    def _finish(self, summary: dict[str, Any]) -> None:
        for name, count in summary["failure_categories"].items():
            if name in self.failure_vars:
                self.failure_vars[name].set(str(count))
        self._set_idle()

    def _set_idle(self) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.generate_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _open_output(self) -> None:
        path = self.last_run_dir or Path(self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path.resolve()))

    def _close(self) -> None:
        from tkinter import messagebox
        if self.running and not messagebox.askyesno("测试尚未完成", "测试正在运行。确定关闭窗口吗？"):
            return
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)
    if args.no_gui:
        return run_cli(config)
    DemoApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
