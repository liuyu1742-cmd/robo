"""Build expanded semantic-parser train/dev data without opening human final text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.semantic_action_augmentation import (
    assert_disjoint_splits,
    build_augmented_split,
)
from tools.semantic_action_data import assert_no_blind_overlap


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_FINAL_HASHES = ROOT / "datasets" / "human_only_final_instruction_hashes.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_dataset(
    catalog_path: Path,
    hash_manifest_path: Path,
    train_output: Path,
    dev_output: Path,
    *,
    train_variants: int = 32,
    dev_variants: int = 8,
) -> dict:
    """Create train/dev only; the prior human final is used solely as hashes."""
    records = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("catalog must be a non-empty relation list")
    train = build_augmented_split(records, "train", train_variants)
    dev = build_augmented_split(records, "dev", dev_variants)
    assert_disjoint_splits(train, dev)
    assert_no_blind_overlap(train, [hash_manifest_path])
    assert_no_blind_overlap(dev, [hash_manifest_path])
    _write_json(train_output, train)
    _write_json(dev_output, dev)
    return {
        "format": "augmented_semantic_action_data_v1",
        "relations": len(records),
        "train_examples": len(train),
        "dev_examples": len(dev),
        "train_variants_per_relation": train_variants,
        "dev_variants_per_relation": dev_variants,
        "old_human_final_hash_only": True,
        "human_final_hash_manifest": str(hash_manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--final-hash-manifest", type=Path, default=DEFAULT_FINAL_HASHES)
    parser.add_argument("--train-output", type=Path, default=ROOT / "datasets" / "semantic_action_augmented_train.json")
    parser.add_argument("--dev-output", type=Path, default=ROOT / "datasets" / "semantic_action_augmented_dev.json")
    parser.add_argument("--report-output", type=Path, default=ROOT / "datasets" / "semantic_action_augmented_data_report.json")
    args = parser.parse_args()
    report = build_dataset(
        args.catalog,
        args.final_hash_manifest,
        args.train_output,
        args.dev_output,
    )
    _write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
