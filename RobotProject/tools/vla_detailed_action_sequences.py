"""Approved fine-grained sequences for the VLA 82 action templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from tools.vla_action_templates import load_vla_action_templates


CATALOGUE_PATH = (
    Path(__file__).resolve().parents[1]
    / "meta"
    / "vla_82_detailed_action_sequences.json"
)


@dataclass(frozen=True)
class VlaDetailedActionSequence:
    object_id: str
    task_id: str
    object_name_zh: str
    source_operation_label: str
    steps: tuple[str, ...]


def validate_vla_detailed_action_sequences(
    catalogue: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    if catalogue.get("format") != "vla_82_detailed_action_sequences_v1":
        errors.append("unexpected detailed sequence catalogue format")

    rows = catalogue.get("sequences", [])
    if not isinstance(rows, list):
        return [*errors, "sequences must be a list"]

    original_rows = load_vla_action_templates()["candidates"]
    original_by_id = {row["object_id"]: row for row in original_rows}
    actual_ids = [row.get("object_id") for row in rows if isinstance(row, Mapping)]
    if len(rows) != 82:
        errors.append(f"expected 82 detailed sequences, got {len(rows)}")
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("detailed sequence object IDs must be unique")
    if set(actual_ids) != set(original_by_id):
        errors.append("detailed sequence IDs do not match the VLA 82 template IDs")

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"invalid detailed sequence record at index {index}")
            continue
        object_id = row.get("object_id")
        original = original_by_id.get(object_id)
        if original is not None:
            for field in ("task_id", "object_name_zh", "source_operation_label"):
                if row.get(field) != original[field]:
                    errors.append(f"{field} mismatch for {object_id}")
        steps = row.get("steps", [])
        if (
            not isinstance(steps, list)
            or len(steps) < 4
            or len(steps) != len(set(steps))
            or any(not isinstance(step, str) or not step.strip() for step in steps)
        ):
            errors.append(f"invalid detailed steps at sequence {index}")
    return errors


@lru_cache(maxsize=1)
def load_vla_detailed_action_sequences() -> dict:
    catalogue = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    errors = validate_vla_detailed_action_sequences(catalogue)
    if errors:
        raise ValueError("; ".join(errors))
    return catalogue


def get_vla_detailed_action_sequence(
    object_id: str,
) -> VlaDetailedActionSequence | None:
    for row in load_vla_detailed_action_sequences()["sequences"]:
        if row["object_id"] == object_id:
            return VlaDetailedActionSequence(
                object_id=row["object_id"],
                task_id=row["task_id"],
                object_name_zh=row["object_name_zh"],
                source_operation_label=row["source_operation_label"],
                steps=tuple(row["steps"]),
            )
    return None
