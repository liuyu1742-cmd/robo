"""Strictly evaluate standard instructions against frozen action sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.checkpoint import load_model
from tools.encode import decode_plan, encode_object, encode_target, encode_task
from tools.generation import beam_search
from tools.instruction_benchmark import score_predictions, validate_benchmark
from tools.instruction_parser import parse_instruction


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "datasets" / "standard_instruction_action_benchmark.json"
DEFAULT_CHECKPOINT = ROOT / "models" / "instruction_action_transformer_v1.pt"
DEFAULT_REPORT = ROOT / "datasets" / "instruction_action_benchmark_report.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "INSTRUCTION_ACTION_BENCHMARK_REPORT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fail-below-threshold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records = json.loads(args.benchmark.read_text(encoding="utf-8"))
    present_splits = tuple(dict.fromkeys(record["split"] for record in all_records))
    audit = validate_benchmark(all_records, expected_splits=present_splits)
    records = [record for record in all_records if record["split"] == args.split]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device=device)
    predictions = {}
    cases = []
    parse_correct = 0
    parse_failures = 0

    for index, record in enumerate(records, start=1):
        parsed_payload = None
        error = None
        prediction = []
        try:
            parsed = parse_instruction(record["instruction"])
            parsed_payload = {
                "task": parsed.task,
                "object": parsed.object,
                "target": parsed.target,
            }
            parse_matches = (
                parsed.task == record["task"]
                and parsed.object == record["object"]
                and parsed.target == record["target"]
            )
            parse_correct += parse_matches
            generated = beam_search(
                model,
                encode_task(parsed.task),
                encode_object(parsed.object),
                encode_target(parsed.target),
                beam_width=args.beam_width,
                max_actions=12,
                device=device,
            )
            prediction = decode_plan(generated)
        except (ValueError, KeyError) as exc:
            parse_failures += 1
            error = str(exc)
        predictions[record["id"]] = prediction
        cases.append(
            {
                "id": record["id"],
                "instruction": record["instruction"],
                "parsed": parsed_payload,
                "expected_actions": record["actions"],
                "predicted_actions": prediction,
                "exact": prediction == record["actions"],
                "error": error,
            }
        )
        if index % 20 == 0 or index == len(records):
            print(f"generated {index}/{len(records)}", flush=True)

    metrics = score_predictions(records, predictions)
    metrics["instruction_parse_accuracy"] = parse_correct / len(records) if records else 0.0
    metrics["instruction_parse_failures"] = parse_failures
    passed = metrics["exact_sequence_accuracy"] >= args.threshold
    report = {
        "format": "frozen_instruction_action_benchmark_report_v1",
        "benchmark": str(args.benchmark.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "threshold": args.threshold,
        "passed": passed,
        "input_policy": "Only instruction is provided to parser/model; labels are used after generation for scoring.",
        "audit": audit,
        "metrics": metrics,
        "cases": cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 标准指令到动作序列冻结测试报告",
        "",
        f"- 测试集合：{args.split}",
        f"- 测试样例：{metrics['cases']}",
        f"- 指令解析准确率：{metrics['instruction_parse_accuracy']:.2%}",
        f"- 整序列准确率：{metrics['exact_sequence_accuracy']:.2%}",
        f"- 步骤准确率：{metrics['step_accuracy']:.2%}",
        f"- 遗漏率：{metrics['omission_rate']:.2%}",
        f"- 顺序错误率：{metrics['order_error_rate']:.2%}",
        f"- 验收阈值：{args.threshold:.0%}",
        f"- 结果：{'通过' if passed else '未通过'}",
        "",
        "## 输入隔离",
        "",
        "生成阶段只向系统提供instruction；task、object、target和expected_actions仅在生成完成后用于判分。",
        "",
        "## 未通过样例（最多20条）",
        "",
    ]
    failures = [case for case in cases if not case["exact"]]
    if not failures:
        lines.append("无。")
    else:
        for case in failures[:20]:
            lines.append(f"- `{case['id']}`：预测 `{case['predicted_actions']}`；标准 `{case['expected_actions']}`")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    print(f"report: {args.report.resolve()}", flush=True)
    print(f"markdown: {args.markdown.resolve()}", flush=True)
    if args.fail_below_threshold and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
