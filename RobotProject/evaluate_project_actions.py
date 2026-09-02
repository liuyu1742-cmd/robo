import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTION_RE = re.compile(r"^([a-zA-Z0-9_&-]+)\(([^()]*)\)$")
REQUIRED_FIELDS = {
    "id", "split", "task", "scene", "object", "target", "actions",
    "instruction", "source", "modality",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit/evaluate project action datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        default=[
            ROOT / "datasets" / "project_train_dataset.json",
            ROOT / "datasets" / "project_val_dataset.json",
            ROOT / "datasets" / "project_test_dataset.json",
        ],
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "datasets" / "project_action_eval_report.json",
    )
    parser.add_argument("--min-records", type=int, default=1000)
    parser.add_argument("--encoder", choices=["project", "legacy"], default="project")
    return parser.parse_args()


def load(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def action_parts(action):
    match = ACTION_RE.match(action)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def audit_record(item):
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(item))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
    if not isinstance(item.get("actions"), list) or not item.get("actions"):
        errors.append("empty_actions")
    else:
        for action in item["actions"]:
            verb, argument = action_parts(action)
            if verb is None:
                errors.append(f"bad_action_syntax:{action}")
                continue
            if not verb:
                errors.append(f"empty_action_verb:{action}")
            if argument is None or argument == "":
                errors.append(f"empty_action_argument:{action}")
    return errors


def main():
    args = parse_args()
    if args.encoder == "project":
        from tools.project_encode import ACTION_VOCAB, ARG_VOCAB, OBJECT_VOCAB, TARGET_VOCAB, TASK_VOCAB
    else:
        from tools.encode import ACTION_VOCAB, ARG_VOCAB, OBJECT_VOCAB, TARGET_VOCAB, TASK_VOCAB

    records = []
    dataset_files = {}
    for path in args.datasets:
        real_path = path if path.is_absolute() else ROOT / path
        data = load(real_path)
        dataset_files[real_path.name] = len(data)
        records.extend(data)

    ids = set()
    duplicate_ids = []
    errors = defaultdict(list)
    split_counts = Counter()
    source_counts = Counter()
    task_counts = Counter()
    object_counts = Counter()
    action_counts = Counter()
    argument_counts = Counter()
    target_counts = Counter()
    source_task_counts = defaultdict(Counter)

    oov_tasks = Counter()
    oov_objects = Counter()
    oov_targets = Counter()
    oov_actions = Counter()
    oov_arguments = Counter()

    for item in records:
        item_id = item.get("id", "<missing-id>")
        if item_id in ids:
            duplicate_ids.append(item_id)
        ids.add(item_id)

        item_errors = audit_record(item)
        if item_errors:
            errors[item_id].extend(item_errors)

        split = item.get("split", "<missing>")
        source = item.get("source", "<missing>")
        task = item.get("task", "<missing>")
        obj = item.get("object", "<missing>")
        target = item.get("target", "<missing>")
        split_counts[split] += 1
        source_counts[source] += 1
        task_counts[task] += 1
        object_counts[obj] += 1
        target_counts[target] += 1
        source_task_counts[source][task] += 1

        if task not in TASK_VOCAB:
            oov_tasks[task] += 1
        if obj not in OBJECT_VOCAB:
            oov_objects[obj] += 1
        if target not in TARGET_VOCAB:
            oov_targets[target] += 1

        for action in item.get("actions", []):
            verb, argument = action_parts(action)
            if verb is None:
                continue
            action_counts[verb] += 1
            argument_counts[argument] += 1
            if verb not in ACTION_VOCAB:
                oov_actions[verb] += 1
            if argument not in ARG_VOCAB:
                oov_arguments[argument] += 1

    total = len(records)
    syntax_valid = total - len(errors)
    hard_failures = []
    if total < args.min_records:
        hard_failures.append(f"record_count_below_min:{total}<{args.min_records}")
    if duplicate_ids:
        hard_failures.append(f"duplicate_ids:{len(duplicate_ids)}")
    if errors:
        hard_failures.append(f"record_errors:{len(errors)}")
    if not {"train", "validation", "test"}.issubset(split_counts):
        hard_failures.append("missing_required_split")

    report = {
        "total_records": total,
        "dataset_files": dataset_files,
        "splits": dict(split_counts),
        "sources": dict(source_counts),
        "unique_ids": len(ids),
        "duplicate_ids": duplicate_ids[:50],
        "syntax_and_required_field_accuracy": syntax_valid / total if total else 0,
        "record_error_count": len(errors),
        "record_error_examples": dict(list(errors.items())[:20]),
        "tasks": dict(task_counts.most_common()),
        "objects_top50": dict(object_counts.most_common(50)),
        "targets": dict(target_counts.most_common()),
        "actions": dict(action_counts.most_common()),
        "arguments_top50": dict(argument_counts.most_common(50)),
        "source_task_counts": {source: dict(counts.most_common()) for source, counts in source_task_counts.items()},
        "current_vocab_compatibility": {
            "encoder": args.encoder,
            "task_oov": dict(oov_tasks.most_common()),
            "object_oov_top50": dict(oov_objects.most_common(50)),
            "target_oov": dict(oov_targets.most_common()),
            "action_oov": dict(oov_actions.most_common()),
            "argument_oov_top50": dict(oov_arguments.most_common(50)),
            "fully_compatible_with_current_encoder": not any(
                [oov_tasks, oov_objects, oov_targets, oov_actions, oov_arguments]
            ),
        },
        "pass": not hard_failures,
        "hard_failures": hard_failures,
    }

    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps({
        "total_records": report["total_records"],
        "splits": report["splits"],
        "sources": report["sources"],
        "unique_ids": report["unique_ids"],
        "syntax_and_required_field_accuracy": report["syntax_and_required_field_accuracy"],
        "record_error_count": report["record_error_count"],
        "encoder": args.encoder,
        "current_encoder_fully_compatible": report["current_vocab_compatibility"]["fully_compatible_with_current_encoder"],
        "oov_task_count": len(oov_tasks),
        "oov_object_count": len(oov_objects),
        "oov_action_count": len(oov_actions),
        "pass": report["pass"],
        "hard_failures": report["hard_failures"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
