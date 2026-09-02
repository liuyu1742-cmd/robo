"""Helpers for independent human paraphrase blind tests.

The blind-test workflow deliberately separates what a human annotator sees
from the hidden expected action sequence used for scoring.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


TEMPLATE_COLUMNS = [
    "case_id",
    "task",
    "task_name_zh",
    "object",
    "target",
    "human_instruction",
]

VARIANT_TEMPLATE_COLUMNS = [
    "case_id",
    "relation_key",
    "variant",
    "task",
    "task_name_zh",
    "object",
    "target",
    "human_instruction",
]


def build_template_rows(
    records: Sequence[Mapping],
    *,
    task_names_zh: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Return rows for humans to fill without exposing expected actions."""
    task_names_zh = task_names_zh or {}
    rows = []
    for record in records:
        rows.append(
            {
                "case_id": str(record["id"]),
                "task": str(record["task"]),
                "task_name_zh": str(task_names_zh.get(record["task"], "")),
                "object": str(record["object"]),
                "target": str(record["target"]),
                "human_instruction": "",
            }
        )
    return rows


def _relation_key(record: Mapping) -> str:
    return str(
        record.get("relation_key")
        or f"{record.get('acceptance_task', record.get('task', ''))}::{record.get('object', '')}"
    )


def build_variant_template_rows(
    records: Sequence[Mapping],
    *,
    variants_per_relation: int = 3,
    task_names_zh: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Expand each standard relation into blind human-writing variants."""
    if variants_per_relation <= 0:
        raise ValueError("variants_per_relation must be positive")
    task_names_zh = task_names_zh or {}
    rows: list[dict[str, str]] = []
    for record in records:
        source_id = str(record["id"])
        task = str(record.get("acceptance_task", record.get("task", "")))
        relation_key = _relation_key(record)
        for variant in range(1, variants_per_relation + 1):
            rows.append(
                {
                    "case_id": f"{source_id}__v{variant}",
                    "relation_key": relation_key,
                    "variant": str(variant),
                    "task": task,
                    "task_name_zh": str(task_names_zh.get(task, "")),
                    "object": str(record.get("object", "")),
                    "target": str(record.get("target", "")),
                    "human_instruction": "",
                }
            )
    return rows


def build_human_only_retest_template_rows(
    records: Sequence[Mapping],
    *,
    variants_per_relation: int = 3,
    task_names_zh: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Create a fresh blank template reserved for a later human-only retest.

    The relation IDs stay compatible with the standard freeze builder.  Its
    independent status comes from the separate file and the rule that only a
    human author may fill ``human_instruction``.
    """
    return build_variant_template_rows(
        records,
        variants_per_relation=variants_per_relation,
        task_names_zh=task_names_zh,
    )


def _normalized_instruction(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def build_frozen_blind_bundle(
    standard_records: Sequence[Mapping],
    filled_rows: Iterable[Mapping],
    *,
    variants_per_relation: int = 3,
    source_csv_bytes: bytes | None = None,
) -> dict:
    """Attach labels after filling and create an auditable frozen blind bundle."""
    by_source_id = {str(record["id"]): record for record in standard_records}
    cases: list[dict] = []
    seen_case_ids: set[str] = set()
    seen_instructions: set[str] = set()
    variants_by_source: dict[str, set[int]] = {}
    for row in filled_rows:
        case_id = str(row.get("case_id", "")).strip()
        instruction = str(row.get("human_instruction", "")).strip()
        match = re.fullmatch(r"(.+)__v([1-9][0-9]*)", case_id)
        if not match:
            raise ValueError(f"invalid variant case_id: {case_id}")
        source_id, variant_text = match.groups()
        variant = int(variant_text)
        if source_id not in by_source_id:
            raise ValueError(f"unknown source case_id: {source_id}")
        if variant > variants_per_relation:
            raise ValueError(f"variant exceeds configured count: {case_id}")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate blind case_id: {case_id}")
        if not instruction:
            raise ValueError(f"blank human_instruction for case_id: {case_id}")
        normalized = _normalized_instruction(instruction)
        if normalized in seen_instructions:
            raise ValueError(f"duplicate normalized instruction: {instruction}")
        seen_case_ids.add(case_id)
        seen_instructions.add(normalized)
        variants_by_source.setdefault(source_id, set()).add(variant)
        source = by_source_id[source_id]
        cases.append(
            {
                "id": f"blind_{case_id}",
                "source_id": source_id,
                "relation_key": _relation_key(source),
                "variant": variant,
                "instruction": instruction,
                "acceptance_task": source.get("acceptance_task", source.get("task")),
                "legacy_task_id": source.get("legacy_task_id", source.get("task")),
                "object": source["object"],
                "target": source["target"],
                "actions": list(source["actions"]),
            }
        )
    expected_variants = set(range(1, variants_per_relation + 1))
    incomplete = sorted(
        source_id
        for source_id in by_source_id
        if variants_by_source.get(source_id, set()) != expected_variants
    )
    if incomplete:
        raise ValueError(f"incomplete variants for source cases: {incomplete[:5]}")
    ordered_cases = sorted(cases, key=lambda case: case["id"])
    csv_payload = source_csv_bytes
    if csv_payload is None:
        csv_payload = json.dumps(list(filled_rows), ensure_ascii=False, sort_keys=True).encode("utf-8")
    relation_keys = sorted({_relation_key(record) for record in standard_records})
    return {
        "format": "independent_blind_action_bundle_v1",
        "source_csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variants_per_relation": variants_per_relation,
        "audit": {
            "relations": len(relation_keys),
            "cases": len(ordered_cases),
            "relation_keys": relation_keys,
            "relation_digest": hashlib.sha256("\n".join(relation_keys).encode("utf-8")).hexdigest(),
        },
        "cases": ordered_cases,
    }


def validate_frozen_blind_bundle(bundle: Mapping, expected_relations: Sequence[str]) -> None:
    """Reject malformed or incomplete frozen bundles before model scoring."""
    if bundle.get("format") != "independent_blind_action_bundle_v1":
        raise ValueError("unsupported blind bundle format")
    variants = bundle.get("variants_per_relation")
    cases = bundle.get("cases")
    if not isinstance(variants, int) or variants <= 0 or not isinstance(cases, list):
        raise ValueError("invalid blind bundle schema")
    expected = sorted(expected_relations)
    audit = bundle.get("audit", {})
    if audit.get("relation_keys") != expected:
        raise ValueError("blind bundle relation coverage does not match expected relations")
    if len(cases) != len(expected) * variants:
        raise ValueError("blind bundle case count does not match relation variants")
    if not isinstance(bundle.get("source_csv_sha256"), str) or len(bundle["source_csv_sha256"]) != 64:
        raise ValueError("blind bundle is missing source CSV fingerprint")


def build_blind_records(
    standard_records: Sequence[Mapping],
    filled_rows: Iterable[Mapping],
) -> list[dict]:
    """Attach hidden labels/actions after independent humans fill instructions."""
    by_id = {str(record["id"]): record for record in standard_records}
    blind_records = []
    seen: set[str] = set()
    for row in filled_rows:
        case_id = str(row.get("case_id", "")).strip()
        instruction = str(row.get("human_instruction", "")).strip()
        if not case_id:
            raise ValueError("blind row missing case_id")
        if case_id not in by_id:
            raise ValueError(f"unknown case_id in blind row: {case_id}")
        if case_id in seen:
            raise ValueError(f"duplicate blind case_id: {case_id}")
        if not instruction:
            raise ValueError(f"blank human_instruction for case_id: {case_id}")
        seen.add(case_id)
        source = by_id[case_id]
        blind_records.append(
            {
                "id": f"blind_{case_id}",
                "instruction": instruction,
                "task": source["task"],
                "object": source["object"],
                "target": source["target"],
                "actions": list(source["actions"]),
                "split": "test",
                "blind_source_id": case_id,
            }
        )
    return blind_records


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    fieldnames: Sequence[str] = TEMPLATE_COLUMNS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = payload.decode(encoding)
            return list(csv.DictReader(io.StringIO(text)))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", payload, 0, len(payload), "CSV is neither UTF-8 nor GB18030")
