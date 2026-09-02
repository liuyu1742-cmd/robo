"""Access and validate the canonical household task/object catalog."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "meta" / "household_task_catalog.json"
LEGACY_CLEANING_TASKS = ("floor_cleaning", "dish_washing", "bathroom_cleaning")


def load_household_catalog(path: Path | None = None) -> dict:
    catalog_path = path or DEFAULT_CATALOG
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def acceptance_task(task_id: str) -> str:
    catalog = load_household_catalog()
    return catalog.get("compatibility_task_map", {}).get(task_id, task_id)


def tasks_for_object(object_id: str) -> tuple[str, ...]:
    catalog = load_household_catalog()
    return tuple(
        relation["task_id"]
        for relation in catalog.get("task_object_relations", [])
        if relation.get("object_id") == object_id
    )


def validate_catalog(catalog: dict) -> list[str]:
    errors: list[str] = []
    objects = catalog.get("objects", [])
    object_ids: list[str] = []
    for index, obj in enumerate(objects):
        object_id = obj.get("id") if isinstance(obj, dict) else obj
        if not isinstance(object_id, str) or not object_id:
            errors.append(f"malformed object at index {index}")
            continue
        object_ids.append(object_id)
    duplicates = sorted(obj for obj, count in Counter(object_ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate global object IDs: {duplicates}")

    acceptance_tasks = catalog.get("acceptance_tasks", [])
    task_id_list: list[str] = []
    for index, task in enumerate(acceptance_tasks):
        task_id = task.get("id") if isinstance(task, dict) else task
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"malformed acceptance task at index {index}")
            continue
        task_id_list.append(task_id)
    duplicate_tasks = sorted(
        task_id for task_id, count in Counter(task_id_list).items() if count > 1
    )
    if duplicate_tasks:
        errors.append(f"duplicate acceptance task IDs: {duplicate_tasks}")
    task_ids = set(task_id_list)
    if len(acceptance_tasks) != 15:
        errors.append(
            f"acceptance task count must be 15, got {len(acceptance_tasks)}"
        )

    cleaning_objects: set[str] = set()
    known_objects = set(object_ids)
    for index, relation in enumerate(catalog.get("task_object_relations", [])):
        if not isinstance(relation, dict) or not isinstance(relation.get("task_id"), str) or not isinstance(relation.get("object_id"), str):
            errors.append(f"malformed relation at index {index}")
            continue
        task_id = relation.get("task_id")
        object_id = relation.get("object_id")
        if task_id not in task_ids:
            errors.append(f"missing task target in relation: {task_id}")
        if object_id not in known_objects:
            errors.append(f"missing object target in relation: {object_id}")
        if task_id == "cleaning":
            cleaning_objects.add(object_id)
    if len(cleaning_objects) != 23:
        errors.append(f"cleaning object count must be 23, got {len(cleaning_objects)}")

    compatibility = catalog.get("compatibility_task_map", {})
    for legacy_id in LEGACY_CLEANING_TASKS:
        if compatibility.get(legacy_id) != "cleaning":
            errors.append(f"missing compatibility mapping: {legacy_id} -> cleaning")
    return errors
