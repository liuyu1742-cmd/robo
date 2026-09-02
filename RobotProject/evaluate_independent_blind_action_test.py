"""Evaluate a frozen human-authored blind test with raw and final scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from tools.action_constraints import constrain_actions
from tools.blind_error_diagnostics import summarize_failures
from tools.household_catalog import acceptance_task
from tools.independent_blind_evaluation import evaluate_blind_cases
from tools.blind_paraphrase import validate_frozen_blind_bundle
from tools.manual_instruction_entry import resolve_manual_instruction
from tools.project_runtime import generate_for_instruction, load_project_v2_model


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "datasets" / "independent_blind_action_test_414.json"
DEFAULT_SOURCE_CSV = ROOT / "datasets" / "instruction_blind_paraphrase_template_414.csv"
DEFAULT_BENCHMARK = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_CHECKPOINT = ROOT / "models" / "action_transformer_project_generation_semantic.pt"
DEFAULT_REPORT = ROOT / "datasets" / "independent_blind_action_test_414_report.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "INDEPENDENT_BLIND_ACTION_EVALUATION_REPORT.md"


def checkpoint_report_metadata(checkpoint: Mapping) -> dict:
    """Return checkpoint provenance without tensor weight/optimizer payloads."""
    excluded = {"model_state_dict", "optimizer_state_dict"}
    return {
        str(key): value
        for key, value in checkpoint.items()
        if key not in excluded
    }


def _final_actions(resolved: object, raw_actions: Sequence[str]) -> tuple[list[str], bool, str | None]:
    result = constrain_actions(resolved, list(raw_actions))
    return list(result.final_actions), not result.accepted, result.reason


def evaluate_cases(
    bundle: Mapping,
    *,
    resolver: Callable[[str], object],
    generator: Callable[[object], Sequence[str]],
    constraint: Callable[[object, Sequence[str]], Sequence[str] | object],
) -> dict:
    """Generate raw actions first, then evaluate separately constrained actions."""
    raw_predictions: dict[str, list[str]] = {}
    final_predictions: dict[str, list[str]] = {}
    runtime: dict[str, dict] = {}
    parse_correct = 0
    for index, case in enumerate(bundle["cases"], start=1):
        case_id = str(case["id"])
        raw_actions: list[str] = []
        final_actions: list[str] = []
        error = None
        resolved_payload = None
        constraint_reason = None
        constraint_applied = False
        try:
            resolved = resolver(str(case["instruction"]))
            resolved_task = acceptance_task(str(resolved.task))
            resolved_payload = {
                "task": str(resolved.task),
                "acceptance_task": resolved_task,
                "object": str(resolved.object),
            }
            parse_correct += int(
                resolved_task == case["acceptance_task"]
                and str(resolved.object) == case["object"]
            )
            raw_actions = list(generator(resolved))
            constrained = constraint(resolved, raw_actions)
            if hasattr(constrained, "final_actions"):
                final_actions = list(constrained.final_actions)
                constraint_applied = not bool(getattr(constrained, "accepted", True))
                constraint_reason = getattr(constrained, "reason", None)
            else:
                final_actions = list(constrained)
                constraint_applied = final_actions != raw_actions
        except Exception as exc:  # preserve a scored failure instead of leaking labels.
            error = str(exc)
        raw_predictions[case_id] = raw_actions
        final_predictions[case_id] = final_actions
        runtime[case_id] = {
            "resolved": resolved_payload,
            "error": error,
            "constraint_applied": constraint_applied,
            "constraint_reason": constraint_reason,
        }
        if index % 20 == 0 or index == len(bundle["cases"]):
            print(f"evaluated {index}/{len(bundle['cases'])}", flush=True)
    report = evaluate_blind_cases(bundle, raw_predictions, final_predictions)
    for case in report["cases"]:
        case.update(runtime[case["id"]])
    report["diagnostic_failures"] = summarize_failures(report["cases"])
    metrics = report["metrics"]
    metrics["instruction_parse_accuracy"] = parse_correct / len(bundle["cases"])
    metrics["instruction_parse_failures"] = len(bundle["cases"]) - parse_correct
    report["input_policy"] = (
        "Each blind instruction is resolved and generated before its hidden actions are read for scoring; "
        "constraints are applied only after raw actions are recorded."
    )
    return report


def load_frozen_bundle(
    bundle_path: Path,
    benchmark_path: Path,
    source_csv_path: Path | None = None,
) -> dict:
    """Load a structurally valid frozen bundle and optionally verify its source CSV."""
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    expected_relations = sorted(str(record["relation_key"]) for record in benchmark)
    validate_frozen_blind_bundle(bundle, expected_relations)
    if source_csv_path is not None:
        source_csv = Path(source_csv_path)
        actual_fingerprint = hashlib.sha256(source_csv.read_bytes()).hexdigest()
        expected_fingerprint = bundle["source_csv_sha256"]
        if expected_fingerprint != actual_fingerprint:
            raise ValueError(
                "blind source CSV fingerprint mismatch: "
                f"expected={expected_fingerprint}; actual={actual_fingerprint}; "
                f"path={source_csv.resolve()}"
            )
    return bundle


def run_evaluation(
    bundle_path: Path,
    checkpoint: Path,
    *,
    device: str,
    beam_width: int,
    max_actions: int,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    source_csv_path: Path | None = None,
) -> dict:
    bundle = load_frozen_bundle(bundle_path, benchmark_path, source_csv_path)
    model, checkpoint_metadata = load_project_v2_model(Path(checkpoint), device=device)

    def generator(resolved: object) -> Sequence[str]:
        generation = generate_for_instruction(
            model,
            resolved,
            device=device,
            beam_width=beam_width,
            max_actions=max_actions,
        )
        return generation.actions

    report = evaluate_cases(
        bundle,
        resolver=resolve_manual_instruction,
        generator=generator,
        constraint=constrain_actions,
    )
    report["checkpoint"] = str(Path(checkpoint).resolve())
    report["checkpoint_metadata"] = checkpoint_report_metadata(checkpoint_metadata)
    report["device"] = device
    report["beam_width"] = beam_width
    report["max_actions"] = max_actions
    return report


def markdown_report(report: Mapping, *, threshold: float) -> str:
    metrics = report["metrics"]
    raw_low, raw_high = metrics["raw_wilson_95"]
    final_low, final_high = metrics["final_wilson_95"]
    return "\n".join(
        [
            "# 独立人工同义指令盲测报告",
            "",
            "本报告严格分开原始模型输出和受约束系统输出；后者不能替代前者。",
            "",
            f"- 样本数：{metrics['cases']}",
            f"- 原始模型整序列准确率：{metrics['raw_exact_sequence_accuracy']:.2%}",
            f"- 原始模型 95% Wilson 区间：[{raw_low:.2%}, {raw_high:.2%}]",
            f"- 最终系统整序列准确率：{metrics['final_exact_sequence_accuracy']:.2%}",
            f"- 最终系统 95% Wilson 区间：[{final_low:.2%}, {final_high:.2%}]",
            f"- 指令解析准确率：{metrics['instruction_parse_accuracy']:.2%}",
            f"- 原始模型是否达到 {threshold:.0%}：{'通过' if metrics['raw_exact_sequence_accuracy'] >= threshold else '未通过'}",
            "",
            "## 解释",
            "",
            "“原始模型整序列准确率”不调用动作约束；“最终系统整序列准确率”允许在原始输出记录后执行安全约束。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--source-csv", type=Path, default=None)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--fail-below-raw-threshold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = run_evaluation(
        args.bundle,
        args.checkpoint,
        device=device,
        beam_width=args.beam_width,
        max_actions=args.max_actions,
        source_csv_path=args.source_csv,
        benchmark_path=args.benchmark,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown_report(report, threshold=args.threshold), encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_exact_sequence_accuracy": report["metrics"]["raw_exact_sequence_accuracy"],
                "final_exact_sequence_accuracy": report["metrics"]["final_exact_sequence_accuracy"],
            },
            ensure_ascii=False,
        )
    )
    print(f"report: {args.report.resolve()}")
    print(f"markdown: {args.markdown.resolve()}")
    if args.fail_below_raw_threshold and report["metrics"]["raw_exact_sequence_accuracy"] < args.threshold:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
