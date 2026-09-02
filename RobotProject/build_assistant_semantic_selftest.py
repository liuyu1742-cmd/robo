"""Freeze an assistant-authored development self-test after model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.semantic_action_augmentation import (
    assert_disjoint_splits,
    build_augmented_split,
)
from tools.semantic_action_data import assert_no_blind_overlap, normalized_instruction_hash


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_FINAL_HASHES = ROOT / "datasets" / "human_only_final_instruction_hashes.json"
DEFAULT_TRAIN = ROOT / "datasets" / "semantic_action_augmented_train.json"
DEFAULT_DEV = ROOT / "datasets" / "semantic_action_augmented_dev.json"
DEFAULT_MODEL = ROOT / "models" / "semantic_action_augmented" / "best.pt"


def _read_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected list: {path}")
    return rows


def build_assistant_selftest(
    catalog: list[dict],
    final_hash_manifest: Path,
    train_rows: list[dict],
    dev_rows: list[dict],
) -> dict:
    """Create one assistant-authored test phrase per relation, distinct by hash."""
    candidates = build_augmented_split(catalog, "assistant_selftest", 6)
    grouped: dict[str, list[dict]] = {}
    for row in candidates:
        grouped.setdefault(str(row["relation_key"]), []).append(row)
    selected = [
        grouped[str(record["relation_key"])][index % 6]
        for index, record in enumerate(catalog)
    ]
    assert_disjoint_splits(train_rows, dev_rows, selected)
    assert_no_blind_overlap(selected, [final_hash_manifest])
    actions_by_relation = {
        str(record["relation_key"]): list(record["actions"]) for record in catalog
    }
    cases = [
        {
            "id": f"assistant_selftest_{index:03d}",
            "relation_key": row["relation_key"],
            "acceptance_task": row["acceptance_task"],
            "object": row["object"],
            "target": row["target"],
            "instruction": row["instruction"],
            "expected_actions": actions_by_relation[str(row["relation_key"])],
            "author": "assistant",
        }
        for index, row in enumerate(selected, start=1)
    ]
    return {
        "format": "assistant_semantic_selftest_v1",
        "evidence_level": "assistant_generated_development_selftest",
        "not_independent_human_evidence": True,
        "cases": cases,
        "relations": len(cases),
        "old_human_final_hash_only": True,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--final-hash-manifest", type=Path, default=DEFAULT_FINAL_HASHES)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "assistant_semantic_selftest_138.json")
    parser.add_argument("--hash-output", type=Path, default=ROOT / "datasets" / "assistant_semantic_selftest_hashes.json")
    args = parser.parse_args()

    catalog = _read_rows(args.catalog)
    bundle = build_assistant_selftest(
        catalog,
        args.final_hash_manifest,
        _read_rows(args.train),
        _read_rows(args.dev),
    )
    bundle["model_sha256_at_freeze"] = _sha256(args.model)
    _write_json(args.output, bundle)
    hashes = sorted(
        normalized_instruction_hash(str(case["instruction"])) for case in bundle["cases"]
    )
    _write_json(
        args.hash_output,
        {
            "format": "assistant_semantic_selftest_hashes_v1",
            "count": len(hashes),
            "hashes": hashes,
            "evidence_level": bundle["evidence_level"],
        },
    )
    print(
        json.dumps(
            {
                "cases": len(bundle["cases"]),
                "evidence_level": bundle["evidence_level"],
                "model_sha256_at_freeze": bundle["model_sha256_at_freeze"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
