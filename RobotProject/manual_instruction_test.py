"""Popup or CLI test for free-form household instructions over 120 operations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from tools.action_constraints import constrain_actions
from tools.manual_instruction_entry import (
    build_manual_result,
    find_expected_actions,
    format_result_text,
    load_benchmark_records,
    resolve_manual_instruction,
)
from tools.multi_device_coordination import (
    is_multi_device_instruction,
    plan_multi_device_instruction,
)
from tools.project_runtime import generate_for_instruction, load_project_v2_model
from tools.vla_instruction_planner import plan_vla_instruction


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "models" / "action_transformer_project_generation_semantic.pt"
DEFAULT_REPORT = ROOT / "datasets" / "manual_instruction_test_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default=None, help="例如：清洗衣物、把杯子洗了、打开阅读灯")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--use-model", action="store_true", help="兼容旧参数；默认已加载checkpoint进行模型预测")
    parser.add_argument("--no-model", action="store_true", help="只测试关键词解析和标准动作序列，不加载checkpoint")
    parser.add_argument("--no-gui", action="store_true", help="不用弹窗，改用终端输入")
    return parser.parse_args()


def run_instruction(args: argparse.Namespace, instruction: str) -> dict:
    if is_multi_device_instruction(instruction):
        result = plan_multi_device_instruction(instruction)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    if vla_result := plan_vla_instruction(instruction):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(vla_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return vla_result

    records = load_benchmark_records()
    resolved = resolve_manual_instruction(instruction, benchmark_records=records)
    predicted_actions: list[str] = []
    prediction_error = None
    generation_metadata = None
    checkpoint_metadata = None
    model_used = not args.no_model
    if model_used:
        try:
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, checkpoint_metadata = load_project_v2_model(
                args.checkpoint, device=device
            )
            generation = generate_for_instruction(
                model,
                resolved,
                device=device,
                beam_width=args.beam_width,
                max_actions=args.max_actions,
            )
            predicted_actions = generation.actions
            generation_metadata = asdict(generation)
            if generation.invalid:
                prediction_error = "模型生成了空动作序列。"
            elif generation.truncated or not generation.terminated:
                prediction_error = "模型生成未在最大步数内完成。"
        except Exception as exc:  # noqa: BLE001 - report model failures without answer fallback.
            prediction_error = str(exc)
            model_used = False
    else:
        prediction_error = "模型生成已禁用。"
    # Standard actions are retrieved after parsing and constrain model output.
    expected_actions = find_expected_actions(resolved, records)
    constraint = None
    final_actions = list(expected_actions)
    if model_used and prediction_error is None:
        constraint = constrain_actions(resolved, predicted_actions)
        final_actions = constraint.final_actions
    result = build_manual_result(
        instruction,
        resolved,
        predicted_actions,
        expected_actions,
        final_actions=final_actions,
        constraint_applied=constraint is not None and not constraint.accepted,
        constraint_reason=constraint.reason if constraint is not None else None,
        constraint_violations=constraint.violations if constraint is not None else [],
        model_used=model_used,
        prediction_error=prediction_error,
        generation_metadata=generation_metadata,
        checkpoint=str(args.checkpoint) if checkpoint_metadata is not None else None,
        template_selection=None,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


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
    button = tk.Button(window, text="关闭", command=window.destroy, font=("Microsoft YaHei UI", 10))
    button.pack(pady=(0, 12))
    window.mainloop()


def _get_popup_instruction() -> str | None:
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    instruction = simpledialog.askstring(
        "输入家政任务指令",
        "请输入一句自然语言指令：\n例如：清洗衣物、帮我洗鞋、把杯子洗了、打开阅读灯",
    )
    root.destroy()
    return instruction.strip() if instruction else None


def build_failure_result(instruction: str, exc: Exception) -> dict:
    error = str(exc)
    return {
        'instruction': instruction,
        'resolved': {
            'task': 'unresolved',
            'object': 'unresolved',
            'target': 'unresolved',
            'operation': 'unresolved',
        },
        'predicted_actions': [],
        'final_actions': [],
        'expected_actions': [],
        'prediction_available': False,
        'model_used': False,
        'prediction_error': error,
        'generation': None,
        'checkpoint': None,
        'template_selection': None,
        'constraint_applied': False,
        'constraint_reason': None,
        'constraint_violations': [],
        'completed': False,
        'completion_status': '未通过',
        'completion_explanation': f'指令解析或动作生成失败：{error}',
    }


def main() -> None:
    args = parse_args()
    use_gui = not args.no_gui and args.instruction is None
    try:
        instruction = args.instruction or (
            _get_popup_instruction() if use_gui else input("请输入任务指令：").strip()
        )
    except Exception:
        instruction = input("请输入任务指令：").strip()
        use_gui = False
    if not instruction:
        print("未输入指令，已退出。", flush=True)
        return

    try:
        result = run_instruction(args, instruction)
        text = format_result_text(result)
    except Exception as exc:
        result = build_failure_result(instruction, exc)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        text = f"输入指令：{instruction}\n完成情况：未通过\n原因：{exc}"
    if use_gui:
        try:
            _show_popup_result("动作序列结果", text)
        except Exception:
            print(text, flush=True)
    else:
        print(text, flush=True)
    print(f"报告：{args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
