import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "meta" / "compliance_catalog_15x120.json"
CANONICAL_CATALOG_PATH = ROOT / "meta" / "household_task_catalog.json"
BENCHMARK_PATH = ROOT / "datasets" / "standard_instruction_action_benchmark.json"
PROJECT_DATASET_PATH = ROOT / "datasets" / "project_action_dataset.json"
OUT = ROOT / "meta" / "project_action_vocab.json"


def load(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_action(action):
    if "(" not in action or not action.endswith(")"):
        raise ValueError(f"Bad action: {action}")
    verb, argument = action[:-1].split("(", 1)
    return verb, argument


def append_unique(values, value):
    if value not in values:
        values.append(value)


def main():
    catalog = load(CATALOG_PATH)
    canonical_catalog = load(CANONICAL_CATALOG_PATH)
    canonical_objects = set(canonical_catalog["objects"])
    benchmark = load(BENCHMARK_PATH)
    project = load(PROJECT_DATASET_PATH)

    tasks = [task["id"] for task in canonical_catalog["acceptance_tasks"]]
    objects = []
    targets = []
    actions = []
    arguments = []

    for task in catalog["tasks"]:
        for obj in task["objects"]:
            append_unique(objects, obj)
            append_unique(arguments, obj)

    for obj in canonical_catalog["objects"]:
        append_unique(objects, obj)
        append_unique(arguments, obj)

    for sample in benchmark:
        if sample["object"] in canonical_objects:
            append_unique(objects, sample["object"])
        append_unique(targets, sample["target"])
        append_unique(arguments, sample["object"])
        append_unique(arguments, sample["target"])
        for action in sample["actions"]:
            verb, argument = parse_action(action)
            append_unique(actions, verb)
            append_unique(arguments, argument)

    for sample in project:
        if sample["object"] in canonical_objects:
            append_unique(objects, sample["object"])
        append_unique(targets, sample["target"])
        append_unique(arguments, sample["object"])
        append_unique(arguments, sample["target"])
        for action in sample["actions"]:
            verb, argument = parse_action(action)
            append_unique(actions, verb)
            append_unique(arguments, argument)

    vocab = {
        "source_files": {
            "catalog_15x120": str(CATALOG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "standard_benchmark": str(BENCHMARK_PATH.relative_to(ROOT)).replace("\\", "/"),
            "project_dataset": str(PROJECT_DATASET_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "counts": {
            "tasks": len(tasks),
            "objects": len(objects),
            "targets": len(targets),
            "actions": len(actions),
            "arguments": len(arguments),
        },
        "tasks": tasks,
        "objects": objects,
        "targets": targets,
        "actions": actions,
        "arguments": arguments,
    }
    with OUT.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)
    print(json.dumps(vocab["counts"], ensure_ascii=False, indent=2))
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
