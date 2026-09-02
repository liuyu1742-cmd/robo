import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "datasets" / "unified_actions"
OUT_DIR = ROOT / "datasets"
REPLACEMENTS_PATH = ROOT / "meta" / "object_replacements_20260704.json"
REPLACEMENTS = json.loads(REPLACEMENTS_PATH.read_text(encoding="utf-8"))["replacements"]
CATALOG = json.loads((ROOT / "meta" / "household_task_catalog.json").read_text(encoding="utf-8"))
COMPATIBILITY_TASK_MAP = CATALOG.get("compatibility_task_map", {})


ACTION_RE = re.compile(r"^([a-zA-Z0-9_&-]+)\((.*)\)$")
SELF_ARG_ACTIONS = {
    "locate", "reach", "grasp", "grasp_handle", "pick_up", "pickup", "carry",
    "carry_with_both_hands", "put_down", "putdown", "release", "place", "open",
    "close", "press", "turn_on", "turn_off", "inspect", "wash", "rinse", "dry",
    "fold", "scrub", "fill", "cut", "cut_with", "staple_with", "dump", "pour",
    "tilt", "remove", "lock", "unlock", "mix",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_action(action):
    match = ACTION_RE.match(action)
    if not match:
        return action, []
    name, raw_args = match.groups()
    args = [item.strip() for item in raw_args.split(",") if item.strip()]
    return name, args


def infer_target(record):
    obj = record.get("object") or "none"
    for action in record.get("ordered_actions", []):
        name, args = parse_action(action)
        if name in {"move", "insert"} and len(args) >= 2:
            return args[1]
        if name == "move" and len(args) == 1 and args[0] != obj:
            return args[0]
        if name not in SELF_ARG_ACTIONS:
            for arg in args:
                if arg != obj:
                    return arg
    return "none"


def convert(record):
    original_object = record["object"]
    object_name = REPLACEMENTS.get(original_object, original_object)
    ordered_actions = [
        action.replace(f"({original_object})", f"({object_name})")
        .replace(f"({original_object},", f"({object_name},")
        .replace(f",{original_object})", f",{object_name})")
        for action in record["ordered_actions"]
    ]
    normalized_record = {**record, "object": object_name, "ordered_actions": ordered_actions}
    annotations = dict(record.get("annotations", {}))
    if object_name != original_object:
        annotations["ontology_replacement"] = {
            "original_object": original_object,
            "replacement_object": object_name,
        }
    return {
        "id": record["id"],
        "split": record["split"],
        "task": record["task"],
        "acceptance_task": COMPATIBILITY_TASK_MAP.get(record["task"], record["task"]),
        "scene": record.get("media", {}).get("video_id", record["source"]),
        "object": object_name,
        "target": infer_target(normalized_record),
        "actions": ordered_actions,
        "verb": record["verb"],
        "instruction": record["instruction"],
        "source": record["source"],
        "modality": record["modality"],
        "media": record.get("media", {}),
        "annotations": annotations,
    }


def main():
    outputs = {
        "train": OUT_DIR / "project_train_dataset.json",
        "validation": OUT_DIR / "project_val_dataset.json",
        "test": OUT_DIR / "project_test_dataset.json",
    }
    by_split = {split: [] for split in outputs}
    counts = Counter()
    source_counts = Counter()
    task_counts = Counter()
    object_counts = Counter()
    target_counts = Counter()

    for split in ("train", "validation", "test"):
        for record in load_jsonl(IN_DIR / f"{split}.jsonl"):
            item = convert(record)
            by_split[split].append(item)
            counts[split] += 1
            source_counts[item["source"]] += 1
            task_counts[item["task"]] += 1
            object_counts[item["object"]] += 1
            target_counts[item["target"]] += 1

    for split, path in outputs.items():
        with path.open("w", encoding="utf-8") as file:
            json.dump(by_split[split], file, ensure_ascii=False, indent=2)

    all_items = [item for split in ("train", "validation", "test") for item in by_split[split]]
    combined_path = OUT_DIR / "project_action_dataset.json"
    with combined_path.open("w", encoding="utf-8") as file:
        json.dump(all_items, file, ensure_ascii=False, indent=2)

    summary = {
        "total": len(all_items),
        "splits": dict(counts),
        "sources": dict(source_counts),
        "tasks": dict(task_counts.most_common()),
        "objects": dict(object_counts.most_common()),
        "targets": dict(target_counts.most_common(30)),
        "outputs": {split: str(path.relative_to(ROOT)).replace("\\", "/") for split, path in outputs.items()},
        "combined_output": str(combined_path.relative_to(ROOT)).replace("\\", "/"),
    }
    summary_path = OUT_DIR / "project_action_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
