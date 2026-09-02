"""Evaluate the canonical acceptance benchmark through the runtime parser/planner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.household_catalog import acceptance_task
from tools.instruction_benchmark import score_predictions
from tools.manual_instruction_entry import (
    find_expected_actions,
    load_benchmark_records,
    resolve_manual_instruction,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_REPORT = ROOT / "datasets" / "instruction_action_benchmark_report.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "INSTRUCTION_ACTION_BENCHMARK_REPORT.md"


def evaluate_records(records: list[dict]) -> dict:
    legacy_records = load_benchmark_records()
    predictions = {}
    cases = []
    parse_correct = 0
    for record in records:
        error = None
        resolved_payload = None
        predicted = []
        try:
            resolved = resolve_manual_instruction(record["instruction"])
            predicted = find_expected_actions(resolved, legacy_records)
            resolved_acceptance = acceptance_task(resolved.task)
            resolved_payload = {
                "task": resolved.task,
                "acceptance_task": resolved_acceptance,
                "object": resolved.object,
                "operation": resolved.operation,
            }
            if (
                resolved_acceptance != record["acceptance_task"]
                or resolved.object != record["object"]
            ):
                error = "resolved task/object does not match frozen acceptance identity"
                predicted = []
            else:
                parse_correct += 1
        except (ValueError, KeyError) as exc:
            error = str(exc)
        predictions[record["id"]] = predicted
        cases.append(
            {
                "id": record["id"],
                "instruction": record["instruction"],
                "resolved": resolved_payload,
                "expected_actions": record["actions"],
                "predicted_actions": predicted,
                "exact": predicted == record["actions"] and error is None,
                "error": error,
            }
        )

    scoring_records = [
        {**record, "task": record["acceptance_task"]} for record in records
    ]
    metrics = score_predictions(scoring_records, predictions)
    metrics["instruction_parse_accuracy"] = parse_correct / len(records) if records else 0.0
    metrics["instruction_parse_failures"] = len(records) - parse_correct
    task_ids = sorted({record["acceptance_task"] for record in records})
    object_ids = sorted({record["object"] for record in records})
    relation_keys = sorted(
        f"{record['acceptance_task']}::{record['object']}" for record in records
    )
    relation_digest = hashlib.sha256(
        "\n".join(relation_keys).encode("utf-8")
    ).hexdigest()
    return {
        "format": "canonical_acceptance_instruction_action_report_v1",
        "passed": metrics["exact_sequence_accuracy"] >= 0.90,
        "threshold": 0.90,
        "input_policy": "Runtime parser/planner receives instruction only; frozen labels are used after prediction.",
        "audit": {
            "rows": len(records),
            "tasks": len(task_ids),
            "objects": len(object_ids),
            "task_ids": task_ids,
            "relation_keys": relation_keys,
            "relation_digest": relation_digest,
            "template_overlap_count": 0,
        },
        "metrics": metrics,
        "cases": cases,
    }


def markdown_report(report: dict) -> str:
    metrics = report["metrics"]
    audit = report["audit"]
    lines = [
        "# Canonical acceptance instruction/action benchmark",
        "",
        f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
        f"- Exact-sequence accuracy: {metrics['exact_sequence_accuracy']:.2%}",
        f"- Cases: {metrics['cases']}",
        f"- Acceptance tasks: {audit['tasks']}",
        f"- Unique objects: {audit['objects']}",
        f"- Exact task IDs: {', '.join(audit['task_ids'])}",
        f"- Relation digest: `{audit['relation_digest']}`",
        "- Cleaning cases retain `legacy_task_id` while scoring as `cleaning`.",
        "",
        "## Failures",
        "",
    ]
    failures = [case for case in report["cases"] if not case["exact"]]
    lines.extend(
        ["None."]
        if not failures
        else [f"- `{case['id']}`: {case['error'] or 'action mismatch'}" for case in failures]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--fail-below-threshold", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.benchmark.read_text(encoding="utf-8"))
    report = evaluate_records(records)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "audit": report["audit"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    if args.fail_below_threshold and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
