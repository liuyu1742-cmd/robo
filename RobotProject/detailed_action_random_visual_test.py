"""Visual/CLI random colloquial benchmark for the isolated detailed-action entry."""

from __future__ import annotations

import argparse
from collections import Counter
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.detailed_action_random_cases import generate_object_subcommand, generate_random_cases
from tools.detailed_action_keyword_router import KeywordEnhancedDetailedActionParser
from tools.detailed_action_random_output import RandomRunOutput
from tools.detailed_action_random_runner import run_random_cases, summarize_random_results
from tools.detailed_action_runtime import (
    DEFAULT_BGE_CHECKPOINT_PATH,
    DEFAULT_BGE_ENCODER_PATH,
    DetailedActionParser,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "detailed_action_random_visual_test"


def failure_reason_zh(row: dict[str, Any]) -> str:
    """Classify one failed row into a concise, stable Chinese category."""
    if row.get("passed"):
        return ""
    error = str(row.get("error", ""))
    clarification = str(row.get("clarification_reason", ""))
    failure = str(row.get("failure_reason", ""))
    if error or failure == "parser_error":
        return "解析器异常"
    if "执行门控" in clarification:
        return "执行门控未通过"
    if any(term in clarification for term in ("操作或用途不兼容", "对象与操作", "属性或操作不兼容")):
        return "对象与操作不兼容"
    if any(term in clarification for term in ("同名对象", "补充房间", "补充位置", "补充信息", "缺少可执行操作")):
        return "对象信息或位置不足"
    if failure == "label_mismatch":
        return "标签不匹配"
    if failure == "result_type_mismatch":
        return "结果类型不匹配"
    if failure == "child_operation_failed":
        return "联动子操作失败"
    if failure == "coordination_requires_multiple_objects":
        return "联动物体数量不足"
    return "其他需要澄清"


def format_failure_counts(rows: list[dict[str, Any]]) -> str:
    categories = [failure_reason_zh(row) for row in rows if not row.get("passed")]
    counts = Counter(category for category in categories if category)
    return "｜".join(f"{category} {count} 条" for category, count in counts.items()) or "无"


def result_detail(row: dict[str, Any]) -> dict[str, str]:
    """Create the upper-panel strings for one immutable result row."""
    children = list(row.get("child_operations", ()))
    if row.get("case_type") == "scene" and children:
        blocks = []
        for offset, child in enumerate(children, 1):
            index = int(child.get("index", offset))
            child_actions = " → ".join(str(action) for action in child.get("detailed_actions", ())) or "未输出细分动作"
            status = "通过" if child.get("passed") else "未通过"
            blocks.append(
                f"子操作{index}｜命令：{child.get('instruction', '')}｜"
                f"标签：{child.get('expected_label', '-')} → {child.get('actual_label') or '-'}｜"
                f"{status}｜{float(child.get('elapsed_ms', 0.0)):.6f} ms｜动作：{child_actions}"
            )
        actions = "\n".join(blocks)
    else:
        actions = " → ".join(str(action) for action in row.get("detailed_actions", ()))
    passed = bool(row.get("passed"))
    if not actions:
        actions = "未输出细分动作"
    elif not passed:
        actions = "非最终采用：" + actions
    if passed:
        failure = "-"
    else:
        failure = (
            str(row.get("error", ""))
            or str(row.get("clarification_reason", ""))
            or failure_reason_zh(row)
        )
        child_failures = [
            f"子操作{child.get('index', index)}：{child.get('error') or child.get('clarification_reason') or child.get('failure_reason')}"
            for index, child in enumerate(children, 1)
            if not child.get("passed")
        ]
        if child_failures:
            failure += "；" + "；".join(child_failures)
    if row.get("case_type") == "scene":
        elapsed = (
            f"父场景解析 {float(row.get('scene_parse_elapsed_ms', 0.0)):.6f} ms｜"
            f"子操作合计 {float(row.get('child_operation_total_elapsed_ms', 0.0)):.6f} ms｜"
            f"联动总计 {float(row.get('elapsed_ms', 0.0)):.6f} ms"
        )
    else:
        elapsed = f"{float(row.get('elapsed_ms', 0.0)):.6f} ms"
    return {
        "instruction": str(row.get("instruction", "")),
        "expected": f"{row.get('case_type', '-')} / {row.get('expected_label', '-')}",
        "actual": (
            f"{row.get('actual_result_type') or '-'} / {row.get('actual_label') or '-'}"
            f" / 路径：{row.get('resolution_backend') or '-'}"
        ),
        "actions": actions,
        "failure": failure,
        "elapsed": elapsed,
    }


def resolve_device(selection: str, *, cuda_available: bool | None = None) -> str:
    """Resolve an explicit/automatic UI choice without silently changing it."""
    import torch

    available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    requested = str(selection).strip().lower()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested == "cuda":
        if not available:
            raise RuntimeError("已选择 CUDA，但当前 Python 环境检测不到可用 CUDA GPU")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"不支持的运行设备：{selection}")


@dataclass(frozen=True)
class RunConfig:
    count: int
    seed: int
    device: str
    checkpoint: Path
    encoder: Path
    output_dir: Path


def warm_up_parser(parser: Any) -> dict[str, Any]:
    """Run one deterministic model forward pass excluded from formal evidence."""
    warmup_case = generate_object_subcommand("object:vla:vla_080", 20260828)
    raw = parser.resolve(warmup_case.instruction)
    return {
        "instruction": warmup_case.instruction,
        "result_type": str(raw.get("result_type", "")),
        "label": str(raw.get("label", "")),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_BGE_CHECKPOINT_PATH)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_BGE_ENCODER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    return args


def run_batch(
    config: RunConfig,
    emit: Callable[[dict[str, Any]], None],
    stop_event: threading.Event,
    *,
    parser_factory: Callable[..., Any] = DetailedActionParser,
    case_generator: Callable[[int, int], Any] = generate_random_cases,
) -> tuple[dict[str, Any], Path]:
    effective_device = resolve_device(config.device)
    cases = case_generator(config.count, config.seed)
    output = RandomRunOutput(config.output_dir, seed=config.seed, requested=config.count)
    output.write_generated_cases(cases)
    emit({"type": "generated", "count": len(cases), "run_dir": str(output.run_dir)})
    emit({
        "type": "device",
        "requested_device": config.device,
        "effective_device": effective_device,
    })
    base_parser = parser_factory(
        semantic_backend="bge",
        bge_checkpoint_path=config.checkpoint,
        bge_encoder_path=config.encoder,
        device=effective_device,
    )
    parser = KeywordEnhancedDetailedActionParser(base_parser)
    warmup = warm_up_parser(parser)
    emit({"type": "warmup", **warmup})

    passed = 0

    def on_result(row, completed: int) -> None:
        nonlocal passed
        output.append_result(row)
        passed += int(row.passed)
        emit({"type": "result", "result": row.to_dict()})
        emit({
            "type": "progress",
            "completed": completed,
            "total": config.count,
            "passed": passed,
            "failed": completed - passed,
            "accuracy_percent": round(passed / completed * 100.0, 2),
        })

    rows = run_random_cases(
        parser, cases, on_result=on_result, stop_requested=stop_event.is_set
    )
    summary = summarize_random_results(rows, requested=config.count, seed=config.seed)
    summary["requested_device"] = config.device
    summary["effective_device"] = effective_device
    if effective_device == "cuda":
        import torch
        summary["device_name"] = torch.cuda.get_device_name(0)
    else:
        summary["device_name"] = "CPU"
    summary["run_dir"] = str(output.run_dir)
    output.finalize(summary)
    emit({"type": "summary", "summary": summary, "run_dir": str(output.run_dir)})
    return summary, output.run_dir


def run_cli(config: RunConfig) -> int:
    def emit(event: dict[str, Any]) -> None:
        if event["type"] == "generated":
            print(f"已现场生成 {event['count']} 条随机口语命令。", flush=True)
        elif event["type"] == "device":
            print(
                f"运行设备：{event['effective_device']}（选择：{event['requested_device']}）",
                flush=True,
            )
        elif event["type"] == "result":
            row = event["result"]
            print(
                f"{row['case_id']} {'通过' if row['passed'] else '未通过'} "
                f"{row['elapsed_ms']:.3f} ms | {row['instruction']}", flush=True,
            )
        elif event["type"] == "summary":
            summary = event["summary"]
            print(
                f"完成 {summary['completed']}/{summary['requested']}，总体准确率 "
                f"{summary['overall']['accuracy_percent']:.2f}%"
            )
            print(f"独立物体平均耗时：{summary['object']['average_elapsed_ms']} ms")
            print(f"场景解析平均耗时：{summary['scene']['average_scene_parse_elapsed_ms']} ms")
            print(f"子操作平均耗时：{summary['scene']['average_child_operation_elapsed_ms']} ms")
            print(f"多设备联动总平均耗时：{summary['scene']['average_elapsed_ms']} ms")
            print(f"结果目录：{event['run_dir']}")

    try:
        summary, _ = run_batch(config, emit, threading.Event())
        return 0 if summary["acceptance"]["passed"] else 2
    except Exception as exc:
        print(f"批次级错误：{type(exc).__name__}: {exc}")
        return 1


class RandomVisualApp:
    FILTERS = ("全部", "仅通过", "仅未通过", "独立物体", "多设备联动")

    def __init__(self, config: RunConfig) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("细分动作随机口语可视化测试")
        self.root.geometry("1580x920")
        self.root.minsize(1180, 740)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.running = False
        self.last_run_dir: Path | None = None
        self.rows: list[dict[str, Any]] = []
        self.row_by_iid: dict[str, dict[str, Any]] = {}

        self.count_var = tk.StringVar(value=str(config.count))
        self.seed_var = tk.StringVar(value=str(config.seed))
        self.device_var = tk.StringVar(value=config.device)
        self.checkpoint_var = tk.StringVar(value=str(config.checkpoint))
        self.encoder_var = tk.StringVar(value=str(config.encoder))
        self.output_var = tk.StringVar(value=str(config.output_dir))
        self.filter_var = tk.StringVar(value="全部")
        self.current_var = tk.StringVar(value="尚未开始")
        self.expected_var = tk.StringVar(value="-")
        self.actual_var = tk.StringVar(value="-")
        self.actions_var = tk.StringVar(value="-")
        self.failure_detail_var = tk.StringVar(value="-")
        self.failure_counts_var = tk.StringVar(value="无")
        self.case_time_var = tk.StringVar(value="-")
        self.progress_var = tk.DoubleVar(value=0)
        self.summary_vars = {key: tk.StringVar(value="-") for key in (
            "completed", "passed", "failed", "accuracy",
            "device", "object_avg", "scene_parse_avg", "child_operation_avg",
            "coordination_total_avg", "overall_avg",
        )}
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        ttk, root = self.ttk, self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)
        params = ttk.LabelFrame(root, text="测试参数", padding=10)
        params.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        for col in (1, 3, 5):
            params.columnconfigure(col, weight=1)
        short = (("测试数量", self.count_var, 0), ("随机种子", self.seed_var, 2), ("运行设备", self.device_var, 4))
        for label, variable, column in short:
            ttk.Label(params, text=label).grid(row=0, column=column, sticky="w", padx=4, pady=3)
            if label == "运行设备":
                widget = ttk.Combobox(
                    params, textvariable=variable,
                    values=("auto", "cuda", "cpu"), state="readonly",
                )
                self.device_combo = widget
            else:
                widget = ttk.Entry(params, textvariable=variable)
            widget.grid(row=0, column=column + 1, sticky="ew", padx=4, pady=3)
        for row, (label, variable) in enumerate((("BGE 检查点", self.checkpoint_var), ("本地编码器", self.encoder_var), ("结果目录", self.output_var)), 1):
            ttk.Label(params, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(params, textvariable=variable).grid(row=row, column=1, columnspan=5, sticky="ew", padx=4, pady=3)

        buttons = ttk.Frame(root, padding=(10, 4))
        buttons.grid(row=1, column=0, sticky="ew")
        self.preview_button = ttk.Button(buttons, text="生成随机预览", command=self._preview)
        self.start_button = ttk.Button(buttons, text="开始测试", command=self._start)
        self.stop_button = ttk.Button(buttons, text="完成当前用例后停止", command=self._stop, state="disabled")
        self.open_button = ttk.Button(buttons, text="打开结果目录", command=self._open_output)
        for button in (self.preview_button, self.start_button, self.stop_button, self.open_button):
            button.pack(side="left", padx=(0, 8))

        live = ttk.LabelFrame(root, text="实时结果（计时仅包含 parser.resolve）", padding=8)
        live.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        live.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate((("当前口语命令", self.current_var), ("期望类型/标签", self.expected_var), ("实际类型/标签", self.actual_var), ("细分动作", self.actions_var), ("失败原因", self.failure_detail_var), ("本条耗时", self.case_time_var))):
            ttk.Label(live, text=label, width=15).grid(row=row, column=0, sticky="nw", pady=2)
            ttk.Label(live, textvariable=variable, wraplength=1340).grid(row=row, column=1, sticky="w", pady=2)
        ttk.Progressbar(live, variable=self.progress_var, maximum=100).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(7, 1))

        summary = ttk.Frame(root, padding=(10, 4))
        summary.grid(row=3, column=0, sticky="ew")
        cards = ttk.Frame(summary)
        cards.pack(fill="x")
        labels = (("已完成", "completed"), ("通过", "passed"), ("未通过", "failed"), ("总体准确率", "accuracy"), ("实际设备", "device"), ("独立平均耗时", "object_avg"), ("场景解析平均耗时", "scene_parse_avg"), ("子操作平均耗时", "child_operation_avg"), ("联动总平均耗时", "coordination_total_avg"), ("总体平均耗时", "overall_avg"))
        for label, key in labels:
            box = ttk.LabelFrame(cards, text=label, padding=(10, 6))
            box.pack(side="left", padx=(0, 5))
            ttk.Label(box, textvariable=self.summary_vars[key], font=("Microsoft YaHei UI", 12, "bold")).pack()
        failure_box = ttk.LabelFrame(summary, text="失败原因统计", padding=(10, 5))
        failure_box.pack(fill="x", pady=(5, 0))
        ttk.Label(failure_box, textvariable=self.failure_counts_var, wraplength=1480).pack(anchor="w")

        table = ttk.LabelFrame(root, text="逐条测试结果", padding=6)
        table.grid(row=4, column=0, sticky="nsew", padx=10, pady=(5, 10))
        table.columnconfigure(0, weight=1)
        table.rowconfigure(1, weight=1)
        top = ttk.Frame(table)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(top, text="显示：").pack(side="left")
        combo = ttk.Combobox(top, textvariable=self.filter_var, values=self.FILTERS, state="readonly", width=14)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_table())
        columns = ("id", "type", "instruction", "expected", "actual", "status", "time")
        self.tree = ttk.Treeview(table, columns=columns, show="headings")
        headings = ("编号", "类型", "随机口语命令", "期望标签", "实际标签", "判定", "耗时(ms)")
        widths = (90, 90, 500, 230, 230, 70, 100)
        for name, heading, width in zip(columns, headings, widths):
            self.tree.heading(name, text=heading)
            self.tree.column(name, width=width, anchor="w")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

    def _config(self) -> RunConfig:
        count = int(self.count_var.get())
        if count <= 0:
            raise ValueError("测试数量必须大于 0")
        device = self.device_var.get().strip().lower()
        resolve_device(device)
        return RunConfig(count, int(self.seed_var.get()), device, Path(self.checkpoint_var.get()), Path(self.encoder_var.get()), Path(self.output_var.get()))

    def _preview(self) -> None:
        from tkinter import messagebox
        try:
            rows = generate_random_cases(self._config().count, self._config().seed)
            self.current_var.set(" | ".join(row.instruction for row in rows[:3]))
        except Exception as exc:
            messagebox.showerror("预览失败", str(exc))

    def _start(self) -> None:
        from tkinter import messagebox
        if self.running:
            return
        try:
            config = self._config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.running = True
        self.stop_event.clear()
        self.rows.clear()
        self._refresh_table()
        self.progress_var.set(0)
        for value in self.summary_vars.values():
            value.set("-")
        self.failure_counts_var.set("无")
        self.failure_detail_var.set("-")
        self.start_button.configure(state="disabled")
        self.preview_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.device_combo.configure(state="disabled")

        def worker() -> None:
            try:
                run_batch(config, self.events.put, self.stop_event)
            except Exception as exc:
                self.events.put({
                    "type": "error",
                    "message": (
                        f"{type(exc).__name__}: {exc}\n\n"
                        f"当前解释器：{sys.executable}\n"
                        "请优先使用项目 run_detailed_action_random_visual_test.bat 启动。"
                    ),
                })
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
        elif kind == "device":
            self.summary_vars["device"].set(event["effective_device"].upper())
            self.current_var.set(f"正在使用 {event['effective_device'].upper()} 加载 BGE 模型……")
        elif kind == "warmup":
            self.current_var.set(
                f"预热完成（不计入正式结果）：{event['instruction']} → {event['result_type']}"
            )
        elif kind == "result":
            row = event["result"]
            self.rows.append(row)
            self._show_row(row)
            self._insert(row)
            self.failure_counts_var.set(format_failure_counts(self.rows))
        elif kind == "progress":
            self.progress_var.set(event["completed"] / event["total"] * 100)
            for key in ("completed", "passed", "failed"):
                self.summary_vars[key].set(str(event[key]))
            self.summary_vars["accuracy"].set(f"{event['accuracy_percent']:.2f}%")
        elif kind == "summary":
            self.last_run_dir = Path(event["run_dir"])
            summary = event["summary"]
            self.summary_vars["object_avg"].set(self._format_ms(summary["object"]["average_elapsed_ms"]))
            self.summary_vars["scene_parse_avg"].set(self._format_ms(summary["scene"]["average_scene_parse_elapsed_ms"]))
            self.summary_vars["child_operation_avg"].set(self._format_ms(summary["scene"]["average_child_operation_elapsed_ms"]))
            self.summary_vars["coordination_total_avg"].set(self._format_ms(summary["scene"]["average_elapsed_ms"]))
            self.summary_vars["overall_avg"].set(self._format_ms(summary["overall"]["average_elapsed_ms"]))
            self.failure_counts_var.set(format_failure_counts(self.rows))
            self._idle()
        elif kind == "error":
            from tkinter import messagebox
            messagebox.showerror("测试失败", event["message"])
            self._idle()

    @staticmethod
    def _format_ms(value: Any) -> str:
        return "无样本" if value is None else f"{float(value):.3f} ms"

    def _visible(self, row: dict[str, Any]) -> bool:
        mode = self.filter_var.get()
        return mode == "全部" or (mode == "仅通过" and row["passed"]) or (mode == "仅未通过" and not row["passed"]) or (mode == "独立物体" and row["case_type"] == "object") or (mode == "多设备联动" and row["case_type"] == "scene")

    def _insert(self, row: dict[str, Any]) -> None:
        if not self._visible(row):
            return
        iid = str(row["case_id"])
        self.row_by_iid[iid] = row
        self.tree.insert("", "end", iid=iid, values=(row["case_id"], "独立物体" if row["case_type"] == "object" else "多设备联动", row["instruction"], row["expected_label"], row["actual_label"] or row["actual_result_type"], "通过" if row["passed"] else "未通过", f"{row['elapsed_ms']:.6f}"))
        children = self.tree.get_children()
        if children:
            self.tree.see(children[-1])

    def _refresh_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.row_by_iid.clear()
        for row in self.rows:
            self._insert(row)

    def _show_row(self, row: dict[str, Any]) -> None:
        detail = result_detail(row)
        self.current_var.set(detail["instruction"])
        self.expected_var.set(detail["expected"])
        self.actual_var.set(detail["actual"])
        self.actions_var.set(detail["actions"])
        self.failure_detail_var.set(detail["failure"])
        self.case_time_var.set(detail["elapsed"])

    def _on_tree_select(self, _event: Any = None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        row = self.row_by_iid.get(str(selected[0]))
        if row is not None:
            self._show_row(row)

    def _idle(self) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.preview_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.device_combo.configure(state="readonly")

    def _open_output(self) -> None:
        path = self.last_run_dir or Path(self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path.resolve()))

    def _close(self) -> None:
        from tkinter import messagebox
        if self.running and not messagebox.askyesno("测试尚未完成", "测试正在运行，确定关闭吗？"):
            return
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(args.count, args.seed, args.device, args.checkpoint, args.encoder, args.output_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)
    if args.no_gui:
        return run_cli(config)
    RandomVisualApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
