"""Frozen fixed-command acceptance data, isolated from all model-development text."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.semantic_action_augmentation import _TASK_CONTEXT, _action_phrase, _ambiguous_relation_pairs
from tools.semantic_action_data import (
    _object_names,
    _operation_word,
    assert_no_blind_overlap,
    normalized_instruction_hash,
)


_OFFICIAL_TEMPLATES = (
    "请执行标准操作：{action}",
    "请按标准要求{action}",
    "请帮我完成：{action}",
    "日常指令：{action}",
)

_RELATION_OVERRIDES = {
    "food_serving::water_cup": (
        "请执行标准操作：给我倒杯水",
        "给我倒水",
        "我想喝水",
        "倒点水过来",
    ),
}


def _fixed_phrases(record: Mapping, pair_counts: Mapping[tuple[str, str], int]) -> list[str]:
    relation = str(record["relation_key"])
    override = _RELATION_OVERRIDES.get(relation)
    if override is not None:
        return list(override)
    object_id = str(record["object"])
    task = str(record["acceptance_task"])
    object_name = _object_names(object_id)[0]
    action, verb = _action_phrase(object_name, list(record["actions"]))
    phrases = [template.format(action=action) for template in _OFFICIAL_TEMPLATES]
    if pair_counts[(object_id, verb)] > 1:
        phrases = [
            f"{phrase}，这是{_TASK_CONTEXT.get(task, task)}的标准事项"
            for phrase in phrases
        ]
    return phrases


def build_official_acceptance(
    records: Sequence[Mapping],
    exclusion_rows: Sequence[Mapping],
    final_hash_manifest: Path,
) -> dict:
    """Build exactly four fixed, non-training commands per canonical relation."""
    if not records:
        raise ValueError("records must not be empty")
    pair_counts = _ambiguous_relation_pairs(records)
    cases: list[dict] = []
    seen: set[str] = set()
    for relation_index, record in enumerate(records, start=1):
        phrases = _fixed_phrases(record, pair_counts)
        if len(phrases) != 4:
            raise ValueError("each relation must provide exactly four official phrases")
        for variant, instruction in enumerate(phrases, start=1):
            digest = normalized_instruction_hash(instruction)
            if digest in seen:
                raise ValueError(f"duplicate official acceptance instruction: {instruction}")
            seen.add(digest)
            cases.append(
                {
                    "id": f"official_{relation_index:03d}_{variant}",
                    "relation_key": str(record["relation_key"]),
                    "acceptance_task": str(record["acceptance_task"]),
                    "object": str(record["object"]),
                    "target": str(record["target"]),
                    "instruction": instruction,
                    "expected_actions": list(record["actions"]),
                    "variant": variant,
                }
            )
    if len(cases) != len(records) * 4:
        raise AssertionError("official case count mismatch")
    assert_no_blind_overlap(cases, [final_hash_manifest])
    exclusion_hashes = {
        normalized_instruction_hash(str(row["instruction"])) for row in exclusion_rows
    }
    overlap = [
        case["instruction"]
        for case in cases
        if normalized_instruction_hash(case["instruction"]) in exclusion_hashes
    ]
    if overlap:
        raise ValueError(f"overlap with development text: {overlap[:3]}")
    return {
        "format": "official_fixed_command_acceptance_v1",
        "evidence_scope": "fixed_command_acceptance",
        "cases": cases,
        "relations": len(records),
        "case_count": len(cases),
        "minimum_raw_passes_at_90_percent": math.ceil(len(cases) * 0.90),
        "old_human_final_hash_only": True,
        "not_training_or_development_data": True,
    }
