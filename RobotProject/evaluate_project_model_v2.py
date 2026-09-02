"""Evaluate project v2 checkpoints with autoregressive generated plans."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from tools.project_encode import (
    encode_object,
    encode_target,
    encode_task,
    parse_action,
)
from tools.project_generation import GeneratedPlan, joint_beam_search
from tools.project_runtime import load_project_v2_model
from train_project_transformer_v2 import ProjectActionDatasetV2, SOURCE_VOCAB, VERB_VOCAB


ROOT = Path(__file__).resolve().parent


def levenshtein_distance(first: list[str], second: list[str]) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for value in first:
        current = [previous[0] + 1]
        for index, other in enumerate(second, start=1):
            current.append(
                min(
                    current[index - 1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + (value != other),
                )
            )
        previous = current
    return previous[-1]


def lcs_length(first: list[str], second: list[str]) -> int:
    previous = [0] * (len(second) + 1)
    for value in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            current.append(previous[index - 1] + 1 if value == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def soft_sequence_metrics(expected: list[str], generated: list[str]) -> dict[str, float]:
    expected_set, generated_set = set(expected), set(generated)
    matched_steps = len(expected_set & generated_set)
    denominator = max(len(expected), len(generated), 1)
    return {
        "edit_similarity": 1.0 - levenshtein_distance(expected, generated) / denominator,
        "step_coverage": matched_steps / len(expected_set) if expected_set else 1.0,
        "step_precision": matched_steps / len(generated_set) if generated_set else 1.0,
        "ordered_step_coverage": lcs_length(expected, generated) / len(expected) if expected else 1.0,
    }


def _record_metrics(item: dict, plan: GeneratedPlan) -> dict:
    expected = item["actions"]
    generated = plan.actions
    expected_pairs = [parse_action(action) for action in expected]
    generated_pairs = [parse_action(action) for action in generated]
    action_correct = sum(
        expected_pair[0] == generated_pair[0]
        for expected_pair, generated_pair in zip(expected_pairs, generated_pairs)
    )
    argument_correct = sum(
        expected_pair[1] == generated_pair[1]
        for expected_pair, generated_pair in zip(expected_pairs, generated_pairs)
    )
    exact = (
        plan.terminated
        and not plan.truncated
        and not plan.invalid
        and expected == generated
    )
    return {
        "exact": exact,
        "action_correct": action_correct,
        "argument_correct": argument_correct,
        "action_total": len(expected_pairs),
        "argument_total": len(expected_pairs),
        **soft_sequence_metrics(expected, generated),
    }


def summarize_generated(raw_items: list[dict], plans: list[GeneratedPlan]) -> dict:
    if len(raw_items) != len(plans):
        raise ValueError("raw_items and plans must have the same length")
    total = len(raw_items)
    totals = Counter()
    by_source = defaultdict(Counter)
    by_task = defaultdict(Counter)
    failures = []
    metric_names = (
        "edit_similarity",
        "step_coverage",
        "step_precision",
        "ordered_step_coverage",
    )
    for item, plan in zip(raw_items, plans):
        metrics = _record_metrics(item, plan)
        totals["sequence_correct"] += int(metrics["exact"])
        totals["truncated_count"] += int(plan.truncated)
        totals["invalid_count"] += int(plan.invalid)
        totals["action_correct"] += metrics["action_correct"]
        totals["action_total"] += metrics["action_total"]
        totals["argument_correct"] += metrics["argument_correct"]
        totals["argument_total"] += metrics["argument_total"]
        for name in metric_names:
            totals[name] += metrics[name]
        for counter in (by_source[item["source"]], by_task[item["task"]]):
            counter["total"] += 1
            counter["sequence_correct"] += int(metrics["exact"])
            counter["action_correct"] += metrics["action_correct"]
            counter["action_total"] += metrics["action_total"]
            for name in metric_names:
                counter[name] += metrics[name]
        if not metrics["exact"] and len(failures) < 30:
            failures.append(
                {
                    "id": item["id"],
                    "source": item["source"],
                    "verb": item["verb"],
                    "task": item["task"],
                    "object": item["object"],
                    "target": item["target"],
                    "expected": item["actions"],
                    "generated": plan.actions,
                    "terminated": plan.terminated,
                    "truncated": plan.truncated,
                    "invalid": plan.invalid,
                    "score": plan.score,
                    "soft_metrics": {
                        name: metrics[name] for name in metric_names
                    },
                }
            )

    def summarize(counter: Counter) -> dict:
        item_total = counter["total"]
        return {
            "total": item_total,
            "sequence_accuracy": counter["sequence_correct"] / item_total if item_total else 0.0,
            "action_token_accuracy": counter["action_correct"] / counter["action_total"] if counter["action_total"] else 0.0,
            **{name: counter[name] / item_total if item_total else 0.0 for name in metric_names},
        }

    return {
        "total": total,
        "sequence_accuracy": totals["sequence_correct"] / total if total else 0.0,
        "action_token_accuracy": totals["action_correct"] / totals["action_total"] if totals["action_total"] else 0.0,
        "argument_token_accuracy": totals["argument_correct"] / totals["argument_total"] if totals["argument_total"] else 0.0,
        "truncated_count": totals["truncated_count"],
        "invalid_count": totals["invalid_count"],
        **{name: totals[name] / total if total else 0.0 for name in metric_names},
        "by_source": {name: summarize(counter) for name, counter in sorted(by_source.items())},
        "by_task": {name: summarize(counter) for name, counter in sorted(by_task.items())},
        "failure_examples": failures,
    }


@torch.no_grad()
def evaluate(model, raw_items: list[dict], device: torch.device, beam_width: int, max_actions: int) -> dict:
    plans = [
        joint_beam_search(
            model,
            source_id=SOURCE_VOCAB[item["source"]],
            verb_id=VERB_VOCAB[item["verb"]],
            task_id=encode_task(item["task"]),
            object_id=encode_object(item["object"]),
            target_id=encode_target(item["target"]),
            device=device,
            beam_width=beam_width,
            max_actions=max_actions,
        )
        for item in raw_items
    ]
    return summarize_generated(raw_items, plans)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "action_transformer_project_generation_final.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "standard_generation_test.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "datasets" / "project_model_generation_eval_report.json",
    )
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, checkpoint = load_project_v2_model(args.checkpoint, args.device)
    scoped_dataset = ProjectActionDatasetV2(args.dataset, visual_features=None)
    metrics = evaluate(
        model,
        scoped_dataset.data,
        torch.device(args.device),
        args.beam_width,
        args.max_actions,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "generation_method": "joint_beam_search",
        "beam_width": args.beam_width,
        "max_actions": args.max_actions,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_teacher_forced_metrics": checkpoint.get("val_metrics"),
        "checkpoint_val_autoregressive_metrics": checkpoint.get(
            "val_autoregressive_metrics"
        ),
        "raw_dataset_total": scoped_dataset.raw_item_count,
        "excluded_out_of_scope_count": scoped_dataset.excluded_item_count,
        "excluded_out_of_scope_reasons": dict(scoped_dataset.excluded_reasons),
        **metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
