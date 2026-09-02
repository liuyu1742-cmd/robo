"""Validation and strict scoring helpers for the instruction benchmark."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from tools.encode import parse_action


REQUIRED_FIELDS = {
    "id",
    "instruction",
    "task",
    "object",
    "target",
    "actions",
    "split",
}
EXPECTED_SPLITS = ("train", "validation", "test")


def template_fingerprint(record: Mapping) -> str:
    """Normalize the object token so paraphrase templates can be compared."""
    instruction = str(record["instruction"]).strip().lower()
    object_name = str(record["object"]).strip().lower()
    normalized = re.sub(re.escape(object_name), "{object}", instruction)
    return re.sub(r"\s+", " ", normalized).strip()


def validate_benchmark(
    records: Sequence[Mapping],
    *,
    expected_tasks: int = 15,
    expected_objects: int = 120,
    expected_splits: Sequence[str] = EXPECTED_SPLITS,
) -> dict:
    """Validate schema, coverage, uniqueness, action syntax, and split leakage."""
    if not records:
        raise ValueError("benchmark is empty")
    ids: set[str] = set()
    instructions: set[str] = set()
    expected_splits = tuple(expected_splits)
    split_records: dict[str, list[Mapping]] = {split: [] for split in expected_splits}
    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"record {index} missing fields: {sorted(missing)}")
        case_id = str(record["id"])
        instruction = str(record["instruction"]).strip()
        if case_id in ids:
            raise ValueError(f"duplicate id: {case_id}")
        if instruction in instructions:
            raise ValueError(f"duplicate instruction: {instruction}")
        ids.add(case_id)
        instructions.add(instruction)
        split = record["split"]
        if split not in split_records:
            raise ValueError(f"unsupported split: {split}")
        if not record["actions"]:
            raise ValueError(f"record {case_id} has no actions")
        for action in record["actions"]:
            parse_action(action)
        split_records[split].append(record)

    all_tasks = {record["task"] for record in records}
    all_objects = {record["object"] for record in records}
    if len(all_tasks) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, found {len(all_tasks)}")
    if len(all_objects) != expected_objects:
        raise ValueError(f"expected {expected_objects} objects, found {len(all_objects)}")

    fingerprints = {
        split: {template_fingerprint(record) for record in split_records[split]}
        for split in expected_splits
    }
    overlaps = {}
    for left_index, left in enumerate(expected_splits):
        for right in expected_splits[left_index + 1 :]:
            overlap = fingerprints[left] & fingerprints[right]
            overlaps[f"{left}__{right}"] = sorted(overlap)
    overlap_count = sum(len(values) for values in overlaps.values())
    if overlap_count:
        raise ValueError(f"template leakage across splits: {overlaps}")

    split_summary = {}
    for split, subset in split_records.items():
        tasks = {record["task"] for record in subset}
        objects = {record["object"] for record in subset}
        if tasks != all_tasks or objects != all_objects:
            raise ValueError(
                f"split {split} lacks full coverage: tasks={len(tasks)}, objects={len(objects)}"
            )
        split_summary[split] = {
            "rows": len(subset),
            "tasks": len(tasks),
            "objects": len(objects),
            "templates": len(fingerprints[split]),
        }

    return {
        "rows": len(records),
        "tasks": len(all_tasks),
        "objects": len(all_objects),
        "unique_ids": len(ids),
        "unique_instructions": len(instructions),
        "template_overlap_count": overlap_count,
        "splits": split_summary,
    }


def score_predictions(
    expected_records: Sequence[Mapping],
    predictions_by_id: Mapping[str, Sequence[str]],
) -> dict:
    """Score full sequences and diagnostic step-level errors."""
    exact = positional_matches = expected_steps = omissions = extras = 0
    verb_matches = argument_matches = order_errors = missing_predictions = 0
    per_task = defaultdict(lambda: {"cases": 0, "exact": 0})
    for record in expected_records:
        expected = list(record["actions"])
        prediction = list(predictions_by_id.get(record["id"], []))
        if record["id"] not in predictions_by_id:
            missing_predictions += 1
        is_exact = prediction == expected
        exact += is_exact
        task_metrics = per_task[record["task"]]
        task_metrics["cases"] += 1
        task_metrics["exact"] += is_exact
        expected_steps += len(expected)
        omissions += max(0, len(expected) - len(prediction))
        extras += max(0, len(prediction) - len(expected))
        if prediction != expected and Counter(prediction) == Counter(expected):
            order_errors += 1
        for index, expected_action in enumerate(expected):
            if index >= len(prediction):
                continue
            predicted_action = prediction[index]
            positional_matches += predicted_action == expected_action
            expected_verb, expected_argument = parse_action(expected_action)
            predicted_verb, predicted_argument = parse_action(predicted_action)
            verb_matches += predicted_verb == expected_verb
            argument_matches += predicted_argument == expected_argument

    count = len(expected_records)
    return {
        "cases": count,
        "exact_sequences": exact,
        "exact_sequence_accuracy": exact / count if count else 0.0,
        "step_accuracy": positional_matches / expected_steps if expected_steps else 0.0,
        "verb_accuracy": verb_matches / expected_steps if expected_steps else 0.0,
        "argument_accuracy": argument_matches / expected_steps if expected_steps else 0.0,
        "omission_rate": omissions / expected_steps if expected_steps else 0.0,
        "extra_step_rate": extras / expected_steps if expected_steps else 0.0,
        "order_error_rate": order_errors / count if count else 0.0,
        "missing_predictions": missing_predictions,
        "per_task": {
            task: {
                **values,
                "exact_sequence_accuracy": values["exact"] / values["cases"],
            }
            for task, values in sorted(per_task.items())
        },
    }
