"""Strict scoring for frozen independent blind action tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

from tools.instruction_benchmark import score_predictions


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score confidence interval."""
    if total <= 0:
        return (0.0, 0.0)
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    rate = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    centre = (rate + z_squared / (2 * total)) / denominator
    radius = z * sqrt((rate * (1 - rate) + z_squared / (4 * total)) / total) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _scoring_records(cases: Sequence[Mapping]) -> list[dict]:
    return [
        {
            "id": str(case["id"]),
            "task": str(case["acceptance_task"]),
            "object": str(case["object"]),
            "target": str(case["target"]),
            "actions": list(case["actions"]),
        }
        for case in cases
    ]


def evaluate_blind_cases(
    bundle: Mapping,
    raw_predictions: Mapping[str, Sequence[str]],
    final_predictions: Mapping[str, Sequence[str]],
) -> dict:
    """Score untouched model actions and final constrained actions separately."""
    cases = list(bundle.get("cases", []))
    if not cases:
        raise ValueError("blind bundle has no cases")
    records = _scoring_records(cases)
    raw_metrics = score_predictions(records, raw_predictions)
    final_metrics = score_predictions(records, final_predictions)
    raw_exact = int(raw_metrics["exact_sequences"])
    final_exact = int(final_metrics["exact_sequences"])
    metrics = {
        "cases": len(cases),
        "raw_exact_sequence_accuracy": raw_metrics["exact_sequence_accuracy"],
        "final_exact_sequence_accuracy": final_metrics["exact_sequence_accuracy"],
        "raw_wilson_95": wilson_interval(raw_exact, len(cases)),
        "final_wilson_95": wilson_interval(final_exact, len(cases)),
        "raw": raw_metrics,
        "final": final_metrics,
    }
    report_cases = []
    for case in cases:
        case_id = str(case["id"])
        expected = list(case["actions"])
        raw_actions = list(raw_predictions.get(case_id, []))
        final_actions = list(final_predictions.get(case_id, []))
        report_cases.append(
            {
                "id": case_id,
                "relation_key": case["relation_key"],
                "variant": case["variant"],
                "instruction": case["instruction"],
                "expected_actions": expected,
                "raw_actions": raw_actions,
                "final_actions": final_actions,
                "raw_exact": raw_actions == expected,
                "final_exact": final_actions == expected,
                "final_differs_from_raw": final_actions != raw_actions,
            }
        )
    return {
        "format": "independent_blind_action_evaluation_report_v1",
        "input_policy": "Raw actions are scored before constraints; final actions are scored separately.",
        "bundle_audit": bundle.get("audit", {}),
        "metrics": metrics,
        "cases": report_cases,
    }
