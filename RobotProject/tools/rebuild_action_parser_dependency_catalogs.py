"""Rebuild parser compatibility catalogs from the retained acceptance benchmark."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

try:
    from tools.vla_action_templates import load_vla_action_templates
except ModuleNotFoundError:
    from vla_action_templates import load_vla_action_templates


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
META = ROOT / "meta"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    tasks: dict[str, list[str]] = defaultdict(list)
    compatibility: dict[str, str] = {}
    for record in records:
        task = str(record["acceptance_task"])
        obj = str(record["object"])
        if obj not in tasks[task]:
            tasks[task].append(obj)
        compatibility[str(record["legacy_task_id"])] = task

    task_entries = [
        {"id": task, "name_zh": task, "objects": objects}
        for task, objects in tasks.items()
    ]
    vla_catalog = load_vla_action_templates()
    catalog = {
        "format": "action_parser_compatibility_catalog_v1",
        "tasks": task_entries,
        "acceptance_tasks": [
            {"id": entry["id"], "name_zh": entry["name_zh"]}
            for entry in task_entries
        ],
        "vla_tasks": vla_catalog["tasks"],
        "vla_objects": vla_catalog["candidates"],
    }
    write_json(META / "compliance_catalog_15x120.json", catalog)

    household = {
        "format": "action_parser_household_catalog_v1",
        "objects": [obj for objects in tasks.values() for obj in objects],
        "acceptance_tasks": catalog["acceptance_tasks"],
        "task_object_relations": [
            {"task_id": task, "object_id": obj}
            for task, objects in tasks.items()
            for obj in objects
        ],
        "compatibility_task_map": compatibility,
        "vla_tasks": vla_catalog["tasks"],
        "vla_objects": vla_catalog["candidates"],
    }
    write_json(META / "household_task_catalog.json", household)

    benchmark = [
        {
            "id": record["id"],
            "split": "test",
            "task": record["acceptance_task"],
            "object": record["object"],
            "operation": "canonical",
            "target": record["target"],
            "actions": record["actions"],
            "instruction": record["instruction"],
            "action_parse": {},
        }
        for record in records
    ]
    write_json(ROOT / "datasets" / "standard_instruction_action_benchmark.json", benchmark)


if __name__ == "__main__":
    main()
