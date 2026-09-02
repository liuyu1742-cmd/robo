"""Leak-checked generated training data for semantic action parsing.

This module deliberately contains no blind-test text.  The only permitted
interaction with the frozen final bundle is ``write_instruction_hash_manifest``:
it produces irreversible normalized hashes, then later training checks hashes
only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.manual_instruction_entry import OBJECT_SYNONYMS


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def normalized_instruction_hash(text: str) -> str:
    """Return the irreversible hash used for data-leak checks."""
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _object_names(object_id: str) -> list[str]:
    names = [
        alias
        for alias in OBJECT_SYNONYMS.get(object_id, [])
        if re.search(r"[\u4e00-\u9fff]", alias)
    ]
    return names or [object_id.replace("_", " ")]


def _operation_word(actions: Sequence[str]) -> str:
    primitives = [action.split("(", 1)[0] for action in actions]
    for primitive, word in (
        ("turn_off", "\u5173\u6389"),
        ("turn_on", "\u6253\u5f00"),
        ("wash", "\u6e05\u6d17"),
        ("clean", "\u6e05\u6d01"),
        ("water", "\u6d47\u6c34"),
        ("organize", "\u6574\u7406"),
        ("grasp", "\u53d6\u7528"),
        ("dispose", "\u6536\u7eb3"),
        ("inspect", "\u68c0\u67e5"),
        ("open", "\u6253\u5f00"),
        ("close", "\u5173\u4e0a"),
        ("cook", "\u4f7f\u7528"),
    ):
        if primitive in primitives:
            return word
    raise ValueError(f"cannot infer an operation word from actions: {actions}")


# Each template carries the operation word: do not add generic wording such as
# "handle it" or "deal with it", because it cannot be labelled reliably.
_TEMPLATES = (
    "\u8bf7{verb}{obj}",
    "\u9ebb\u70e6\u5e2e\u6211{verb}{obj}",
    "\u5e2e\u6211\u628a{obj}{verb}\u4e00\u4e0b",
    "\u73b0\u5728{verb}{obj}",
    "\u6709\u7a7a\u65f6{verb}{obj}",
    "\u5bb6\u91cc\u9700\u8981{verb}{obj}",
    "\u80fd\u4e0d\u80fd{verb}{obj}",
    "\u8bf7\u4f60{verb}{obj}",
    "\u628a{obj}{verb}\u4e00\u4e0b",
    "\u8bb0\u5f97{verb}{obj}",
    "\u65b9\u4fbf\u7684\u8bdd{verb}{obj}",
    "\u66ff\u6211{verb}{obj}",
    "\u9700\u8981\u4f60{verb}{obj}",
    "\u5e2e\u5fd9{verb}{obj}",
    "\u9a6c\u4e0a{verb}{obj}",
    "\u52b3\u70e6{verb}{obj}",
    "\u8bf7\u628a{obj}{verb}\u597d",
    "\u628a\u8fd9\u4e2a{obj}{verb}",
    "\u8fd9\u4e2a{obj}\u9ebb\u70e6{verb}",
    "\u6211\u60f3\u8ba9\u4f60{verb}{obj}",
    "\u8bf7\u534f\u52a9{verb}{obj}",
    "\u4eca\u5929{verb}{obj}",
    "\u73b0\u5728\u9ebb\u70e6{verb}{obj}",
    "\u8bf7\u7acb\u5373{verb}{obj}",
    "\u5e2e\u6211{verb}\u8fd9\u4e2a{obj}",
    "\u5b89\u6392\u4e00\u4e0b{obj}\u7684{verb}",
    "\u8bf7\u4e3a\u6211{verb}{obj}",
    "\u8fd9\u4e2a{obj}\u8bf7{verb}",
    "\u6211\u8981{verb}{obj}",
    "\u52a9\u624b\uff0c\u8bf7{verb}{obj}",
    "\u60f3\u8bf7\u4f60{verb}{obj}",
    "\u8bf7\u5e2e\u6211{verb}\u8fd9\u4e2a{obj}",
    "\u5feb\u5e2e\u6211{verb}{obj}",
    "\u8bf7\u53ca\u65f6{verb}{obj}",
    "\u8fd9\u4e2a{obj}\u9700\u8981{verb}",
    "\u6211\u5e0c\u671b\u4f60{verb}{obj}",
    "\u8bf7\u8d1f\u8d23{verb}{obj}",
    "\u8bf7\u5c3d\u5feb{verb}{obj}",
    "\u5e2e\u6211\u628a{obj}\u5b8c\u6210{verb}",
    "\u8bf7\u7acb\u523b{verb}{obj}",
)

_TASK_NAMES = {
    "cleaning": "\u5bb6\u5ead\u6e05\u6d01",
    "bedroom_service": "\u5367\u5ba4\u670d\u52a1",
    "clothing_care": "\u8863\u7269\u62a4\u7406",
    "medical_service": "\u5bb6\u5ead\u533b\u7597\u5173\u6000",
    "security_monitoring": "\u5b89\u9632\u68c0\u67e5",
    "maintenance_management": "\u517b\u62a4\u7ba1\u7406",
    "entertainment_service": "\u5a31\u4e50\u670d\u52a1",
    "appliance_management": "\u5bb6\u7535\u7ba1\u7406",
    "object_fetching": "\u7269\u54c1\u53d6\u9001",
    "kitchen_service": "\u53a8\u623f\u670d\u52a1",
    "bathroom_cleaning": "\u536b\u6d74\u6e05\u6d01",
    "laundry": "\u8863\u7269\u6d17\u62a4",
    "home_organization": "\u5bb6\u5c45\u6574\u7406",
    "elderly_assistance": "\u533b\u7597\u5173\u6000",
}


def build_semantic_examples(
    records: Sequence[Mapping], *, variants_per_relation: int = 40
) -> list[dict]:
    """Generate labelled, operation-explicit phrases without blind-test access."""
    if variants_per_relation <= 0 or variants_per_relation > len(_TEMPLATES):
        raise ValueError(f"variants_per_relation must be 1..{len(_TEMPLATES)}")
    rows: list[dict] = []
    seen: set[str] = set()
    # The same physical object can legitimately appear in two task families.
    # If their first operation is also identical, bare wording is ambiguous;
    # include concise task context for precisely those relations.
    object_operation_counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (
            str(record["object"]),
            _operation_word(list(record.get("actions", []))),
        )
        object_operation_counts[key] = object_operation_counts.get(key, 0) + 1
    for record in records:
        relation = str(record["relation_key"])
        obj = str(record["object"])
        names = _object_names(obj)
        verb = _operation_word(list(record.get("actions", [])))
        task = str(record["acceptance_task"])
        needs_context = object_operation_counts[(obj, verb)] > 1
        for index in range(variants_per_relation):
            phrase = _TEMPLATES[index].format(verb=verb, obj=names[index % len(names)])
            if needs_context:
                phrase = f"{phrase}\uff0c\u8fd9\u662f{_TASK_NAMES.get(task, task)}\u7684\u4e8b"
            normalized = _normalize(phrase)
            if normalized in seen:
                raise ValueError(
                    "ambiguous duplicate generated instruction; add a meaningful "
                    f"task-specific template instead: {phrase}"
                )
            seen.add(normalized)
            rows.append(
                {
                    "id": f"semantic_{relation.replace('::', '__')}__v{index + 1}",
                    "relation_key": relation,
                    "acceptance_task": str(record["acceptance_task"]),
                    "object": obj,
                    "target": str(record["target"]),
                    "instruction": phrase,
                    "source": "generated_semantic_train_v1",
                    "variant": index + 1,
                }
            )
    return rows


def write_instruction_hash_manifest(bundle_path: Path, output_path: Path) -> None:
    """One-time freeze helper writing hashes only, never plaintext instructions."""
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    hashes = sorted(
        {normalized_instruction_hash(str(case["instruction"])) for case in bundle["cases"]}
    )
    output_path.write_text(
        json.dumps({"count": len(hashes), "hashes": hashes}, indent=2) + "\n",
        encoding="utf-8",
    )


def assert_no_blind_overlap(
    examples: Sequence[Mapping], blind_hash_paths: Sequence[Path]
) -> None:
    """Reject generated phrases whose normalized hash is in a blind manifest."""
    blind_hashes: set[str] = set()
    for path in blind_hash_paths:
        blind_hashes.update(
            json.loads(Path(path).read_text(encoding="utf-8")).get("hashes", [])
        )
    overlaps = [
        row["instruction"]
        for row in examples
        if normalized_instruction_hash(str(row["instruction"])) in blind_hashes
    ]
    if overlaps:
        raise ValueError(f"overlap with blind instruction hashes: {overlaps[:3]}")


def split_by_relation(
    examples: Sequence[Mapping], *, dev_variant: int = 40
) -> tuple[list[dict], list[dict]]:
    """Hold out one fixed generated variant per relation for development only."""
    train = [dict(row) for row in examples if int(row["variant"]) != dev_variant]
    dev = [dict(row) for row in examples if int(row["variant"]) == dev_variant]
    return train, dev
