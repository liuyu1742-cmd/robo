"""Materialize model-ready splits from the standard instruction benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tools.household_catalog import acceptance_task
from tools.project_conditions import MANUAL_SOURCE, OPERATION_TO_VERB


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "datasets" / "standard_instruction_action_benchmark.json"
OUTPUTS = {
    "train": ROOT / "datasets" / "standard_generation_train.json",
    "validation": ROOT / "datasets" / "standard_generation_val.json",
    "test": ROOT / "datasets" / "standard_generation_test.json",
}
SUMMARY_PATH = ROOT / "datasets" / "standard_generation_dataset_summary.json"


def convert_record(record: dict) -> dict:
    """Keep labels intact while adapting benchmark conditions for the v2 model."""
    operation = record["operation"]
    try:
        verb = OPERATION_TO_VERB[operation]
    except KeyError as exc:
        raise ValueError(f"No model verb mapping for benchmark operation: {operation}") from exc
    return {
        "id": f"standard_{record['id']}",
        "split": record["split"],
        "source": MANUAL_SOURCE,
        "verb": verb,
        "task": acceptance_task(record["task"]),
        "object": record["object"],
        "target": record["target"],
        "actions": list(record["actions"]),
        "instruction": record["instruction"],
        "operation": operation,
        "source_benchmark_id": record["id"],
    }


def build_datasets(benchmark_path: Path = BENCHMARK_PATH) -> dict[str, list[dict]]:
    records = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    outputs = {split: [] for split in OUTPUTS}
    for record in records:
        split = record.get("split")
        if split not in outputs:
            raise ValueError(f"Unexpected benchmark split: {split}")
        outputs[split].append(convert_record(record))
    return outputs


def main() -> None:
    datasets = build_datasets()
    for split, path in OUTPUTS.items():
        path.write_text(
            json.dumps(datasets[split], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = {
        split: {
            "records": len(records),
            "tasks": len({record["task"] for record in records}),
            "objects": len({record["object"] for record in records}),
            "operations": dict(Counter(record["operation"] for record in records)),
        }
        for split, records in datasets.items()
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
