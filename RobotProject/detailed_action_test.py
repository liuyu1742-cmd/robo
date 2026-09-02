"""Standalone GUI/CLI entry point for detailed household-action resolution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tools.detailed_action_catalog import CATALOGUE_PATH
from tools.detailed_action_runtime import (
    DEFAULT_CHECKPOINT_PATH,
    DetailedActionParser,
    format_detailed_action_result,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_PATH
DEFAULT_CATALOG = CATALOGUE_PATH
DEFAULT_REPORT = ROOT / "outputs" / "detailed_action_test" / "latest_result.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for both PyCharm and shell invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", help="要解析的中文自然语言指令")
    parser.add_argument("--no-gui", action="store_true", help="不显示 Tkinter 输入或结果窗口")
    parser.add_argument(
        "--checkpoint",
        "--model",
        dest="checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="DetailedActionParser checkpoint 路径",
    )
    parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG, help="细分动作目录 JSON 路径"
    )
    parser.add_argument("--device", default="cpu", help="解析模型设备，例如 cpu 或 cuda")
    parser.add_argument(
        "--semantic-backend", choices=("character", "bge"), default="character",
        help="semantic scorer; bge uses the local dual-head checkpoint",
    )
    parser.add_argument(
        "--bge-checkpoint", type=Path,
        default=ROOT / "models" / "detailed_action_bge" / "stage1_e50.pt",
    )
    parser.add_argument(
        "--bge-encoder", type=Path,
        default=ROOT / "models" / "pretrained" / "BAAI_bge-small-zh-v1.5",
    )
    parser.add_argument("--model-weight", type=float, default=0.35)
    parser.add_argument("--minimum-confidence", type=float, default=0.52)
    parser.add_argument("--minimum-margin", type=float, default=0.08)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON 报告路径")
    return parser.parse_args(argv)


def parser_from_args(args: argparse.Namespace) -> DetailedActionParser:
    """Construct the dedicated parser without consulting legacy entry points."""
    return DetailedActionParser(
        catalog_path=args.catalog,
        checkpoint_path=args.checkpoint,
        semantic_backend=args.semantic_backend,
        bge_checkpoint_path=args.bge_checkpoint,
        bge_encoder_path=args.bge_encoder,
        device=args.device,
        model_weight=args.model_weight,
        minimum_confidence=args.minimum_confidence,
        minimum_margin=args.minimum_margin,
    )


def timed_detailed_instruction(args: argparse.Namespace, instruction: str) -> tuple[dict[str, Any], str]:
    """Resolve and render an instruction, measuring only computation and formatting."""
    started_ns = time.perf_counter_ns()
    result = parser_from_args(args).resolve(instruction)
    text_without_time = format_detailed_action_result(result)
    ended_ns = time.perf_counter_ns()
    elapsed_ns = ended_ns - started_ns
    result.update(
        {
            "started_perf_counter_ns": started_ns,
            "ended_perf_counter_ns": ended_ns,
            "elapsed_ns": elapsed_ns,
            "elapsed_seconds": elapsed_ns / 1_000_000_000,
        }
    )
    text = f"{text_without_time}\n运行耗时：{result['elapsed_seconds']:.9f} 秒"
    return result, text


def _write_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def run_detailed_instruction(args: argparse.Namespace, instruction: str) -> dict[str, Any]:
    """Resolve one instruction and persist its completed result outside the timer."""
    result, _ = timed_detailed_instruction(args, instruction)
    _write_report(Path(args.report), result)
    return result


def _format_result_with_timing(result: dict[str, Any]) -> str:
    text = format_detailed_action_result(result)
    if "elapsed_seconds" in result:
        text += f"\n运行耗时：{float(result['elapsed_seconds']):.9f} 秒"
    return text


def _get_popup_instruction() -> str | None:
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    try:
        instruction = simpledialog.askstring(
            "输入细分动作指令",
            "请输入一条自然语言指令：\n例如：帮我把地面清理干净",
            parent=root,
        )
    finally:
        root.destroy()
    return instruction.strip() if instruction else None


def _show_popup_result(title: str, text: str) -> None:
    import tkinter as tk
    from tkinter import scrolledtext

    window = tk.Tk()
    window.title(title)
    window.geometry("720x460")
    box = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Microsoft YaHei UI", 11))
    box.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
    box.insert(tk.END, text)
    box.configure(state="disabled")
    tk.Button(window, text="关闭", command=window.destroy).pack(pady=(0, 12))
    window.mainloop()


def _terminal_instruction() -> str:
    return input("请输入细分动作指令：").strip()


def main_for_test(argv: list[str] | None = None) -> int:
    """Run the entry point and return a testable process-style exit code."""
    args = parse_args(argv)
    use_gui = not args.no_gui and args.instruction is None
    if args.instruction is not None:
        instruction = args.instruction.strip()
    elif use_gui:
        try:
            instruction = _get_popup_instruction()
        except Exception as exc:  # GUI is optional in headless PyCharm sessions.
            print(f"图形输入不可用，已切换到终端输入：{exc}", flush=True)
            instruction = _terminal_instruction()
            use_gui = False
    else:
        instruction = _terminal_instruction()

    if not instruction:
        print("未输入指令，未生成动作或报告。", flush=True)
        return 2

    try:
        result = run_detailed_instruction(args, instruction)
    except Exception as exc:  # Keep model/catalog failures readable to a PyCharm user.
        print(f"解析失败：{exc}", flush=True)
        return 1

    text = _format_result_with_timing(result)
    if use_gui:
        try:
            _show_popup_result("细分动作解析结果", text)
        except Exception as exc:
            print(f"图形结果窗口不可用，已输出到终端：{exc}", flush=True)
            print(text, flush=True)
    else:
        print(text, flush=True)
    print(f"报告：{Path(args.report).resolve()}", flush=True)
    return 0


def main() -> None:
    raise SystemExit(main_for_test())


if __name__ == "__main__":
    main()
