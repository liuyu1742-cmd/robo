"""Build, load, and validate the unified fine-grained action catalogue."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from tools.build_all_object_coordination_catalog import build_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = ROOT / "meta" / "detailed_action_entry_catalog.json"
FORMAT = "detailed_action_entry_catalog_v1"
SOURCE_NAMES = {"原有目录": "original", "VLA": "vla"}
FORBIDDEN_ACTION_FIELDS = {
    "original_actions",
    "original_action_sequence",
    "final_actions",
}


def _object_label(source: str, object_id: str) -> str:
    return f"object:{source}:{object_id}"


def _has_forbidden_action_field(value: object) -> bool:
    if isinstance(value, Mapping):
        if any(key in FORBIDDEN_ACTION_FIELDS for key in value):
            return True
        return any(_has_forbidden_action_field(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_forbidden_action_field(item) for item in value)
    return False


def _is_nonempty_action_sequence(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(action, str) and action.strip() for action in value)
    )


def build_detailed_action_catalog() -> dict[str, Any]:
    """Project the reviewed coordination catalogue into the new unified entry format."""
    reviewed = build_catalog()
    object_entries: list[dict[str, Any]] = []
    object_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for detail in reviewed["object_details"]:
        source = SOURCE_NAMES[detail["source"]]
        object_entry = {
            "label": _object_label(source, detail["object_id"]),
            "entry_type": "object",
            "source": source,
            "object_id": detail["object_id"],
            "object_name_zh": detail["object_name_zh"],
            "task": detail["task"],
            "detailed_actions": list(detail["detailed_actions"]),
            "control_parameters": detail["control_parameters"],
        }
        object_entries.append(object_entry)
        object_by_key[(detail["source"], detail["object_id"])] = object_entry

    scene_entries: list[dict[str, Any]] = []
    details_by_scene: dict[str, list[dict[str, Any]]] = {}
    for detail in reviewed["object_details"]:
        details_by_scene.setdefault(detail["scenario_id"], []).append(detail)

    for scenario in reviewed["scenarios"]:
        objects = []
        for detail in details_by_scene[scenario["scenario_id"]]:
            object_entry = object_by_key[(detail["source"], detail["object_id"])]
            objects.append({
                "object_label": object_entry["label"],
                "object_name_zh": object_entry["object_name_zh"],
                "detailed_actions": list(object_entry["detailed_actions"]),
                "control_parameters": object_entry["control_parameters"],
            })
        scene_entries.append({
            "label": f"scene:{scenario['scenario_id']}",
            "entry_type": "scene",
            "scenario_id": scenario["scenario_id"],
            "command": scenario["command"],
            "objects": objects,
        })

    entries = [*scene_entries, *object_entries]
    catalogue = {
        "format": FORMAT,
        "summary": {
            "scene_count": len(scene_entries),
            "object_entry_count": len(object_entries),
            "intent_count": len(entries),
        },
        "entries": entries,
    }
    errors = validate_detailed_action_catalog(catalogue)
    if errors:
        raise ValueError("; ".join(errors))
    return catalogue


def validate_detailed_action_catalog(catalogue: Mapping[str, object]) -> list[str]:
    """Return structural and content errors for a unified action catalogue."""
    errors: list[str] = []
    if catalogue.get("format") != FORMAT:
        errors.append("unexpected catalogue format")
    if _has_forbidden_action_field(catalogue):
        errors.append("original or final action field leak detected")

    entries = catalogue.get("entries")
    if not isinstance(entries, list):
        return [*errors, "entries must be a list"]

    labels = [entry.get("label") for entry in entries if isinstance(entry, Mapping)]
    if len(labels) != len(entries) or any(not isinstance(label, str) or not label for label in labels):
        errors.append("every entry must have a non-empty label")
    if len(labels) != len(set(labels)):
        errors.append("duplicate entry label")

    scenes = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("entry_type") == "scene"]
    objects = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("entry_type") == "object"]
    if len(scenes) != 49:
        errors.append(f"expected 49 scene entries, got {len(scenes)}")
    if len(objects) != 205:
        errors.append(f"expected 205 object entries, got {len(objects)}")
    if len(entries) != 254:
        errors.append(f"expected 254 intents, got {len(entries)}")

    summary = catalogue.get("summary")
    expected_summary = {
        "scene_count": len(scenes),
        "object_entry_count": len(objects),
        "intent_count": len(entries),
    }
    if summary != expected_summary:
        errors.append("summary counts do not match entries")

    source_counts = Counter()
    object_entries_by_label: dict[str, Mapping[str, object]] = {}
    for entry in objects:
        source = entry.get("source")
        object_id = entry.get("object_id")
        if source not in {"original", "vla"}:
            errors.append(f"invalid object source for {entry.get('label')}")
        else:
            source_counts[source] += 1
        if not isinstance(object_id, str) or not object_id:
            errors.append(f"missing object ID for {entry.get('label')}")
        elif entry.get("label") != _object_label(str(source), object_id):
            errors.append(f"invalid object label for {object_id}")
        if not isinstance(entry.get("object_name_zh"), str) or not entry["object_name_zh"].strip():
            errors.append(f"missing object name for {entry.get('label')}")
        if not isinstance(entry.get("task"), str) or not entry["task"].strip():
            errors.append(f"missing task for {entry.get('label')}")
        if not _is_nonempty_action_sequence(entry.get("detailed_actions")):
            errors.append(f"empty detailed actions for {entry.get('label')}")
        if not isinstance(entry.get("control_parameters"), str) or not entry["control_parameters"].strip():
            errors.append(f"missing control parameters for {entry.get('label')}")
        if isinstance(entry.get("label"), str):
            object_entries_by_label[entry["label"]] = entry
    if source_counts != Counter({"original": 123, "vla": 82}):
        errors.append("object source counts must be original=123 and vla=82")

    for entry in scenes:
        scenario_id = entry.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            errors.append(f"missing scenario ID for {entry.get('label')}")
        elif entry.get("label") != f"scene:{scenario_id}":
            errors.append(f"invalid scene label for {scenario_id}")
        if not isinstance(entry.get("command"), str) or not entry["command"].strip():
            errors.append(f"missing command for {entry.get('label')}")
        involved_objects = entry.get("objects")
        if not isinstance(involved_objects, list) or not involved_objects:
            errors.append(f"missing involved objects for {entry.get('label')}")
            continue
        for item in involved_objects:
            if not isinstance(item, Mapping):
                errors.append(f"invalid scene object for {entry.get('label')}")
                continue
            object_label = item.get("object_label")
            if not isinstance(object_label, str):
                errors.append(f"invalid scene object reference for {entry.get('label')}")
                continue
            referenced_object = object_entries_by_label.get(object_label)
            if referenced_object is None:
                errors.append(f"invalid scene object reference for {entry.get('label')}")
                continue
            if not _is_nonempty_action_sequence(item.get("detailed_actions")):
                errors.append(f"empty scene detailed actions for {entry.get('label')}")
            controls = item.get("control_parameters")
            if not isinstance(controls, str) or not controls.strip():
                errors.append(f"missing scene controls for {entry.get('label')}")
            if item.get("object_name_zh") != referenced_object.get("object_name_zh"):
                errors.append(f"object name mismatch for {entry.get('label')}")
            if item.get("detailed_actions") != referenced_object.get("detailed_actions"):
                errors.append(f"detailed actions mismatch for {entry.get('label')}")
            if controls != referenced_object.get("control_parameters"):
                errors.append(f"control parameters mismatch for {entry.get('label')}")
    return errors


def write_detailed_action_catalog(path: Path = CATALOGUE_PATH) -> Path:
    """Generate the checked-in catalogue from the reviewed source data."""
    catalogue = build_detailed_action_catalog()
    path.write_text(json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_detailed_action_catalog(path: Path = CATALOGUE_PATH) -> dict[str, Any]:
    """Load the generated unified catalogue after validating its contents."""
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_detailed_action_catalog(catalogue)
    if errors:
        raise ValueError("; ".join(errors))
    return catalogue
