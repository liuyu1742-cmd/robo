"""Build leak-checked train/dev data for the hybrid semantic action parser.

The frozen human final test is accessed exactly once here to make an
irreversible SHA-256 manifest.  Its instructions and action labels are never
copied into the generated training or development data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.semantic_action_data import (
    assert_no_blind_overlap,
    build_semantic_examples,
    split_by_relation,
    write_instruction_hash_manifest,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_FINAL_BUNDLE = ROOT / "datasets" / "human_only_independent_action_retest_414.json"
DEFAULT_HASH_MANIFEST = ROOT / "datasets" / "human_only_final_instruction_hashes.json"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--final-bundle", type=Path, default=DEFAULT_FINAL_BUNDLE)
    parser.add_argument("--hash-manifest", type=Path, default=DEFAULT_HASH_MANIFEST)
    parser.add_argument("--train-output", type=Path, default=ROOT / "datasets" / "semantic_action_train.json")
    parser.add_argument("--dev-output", type=Path, default=ROOT / "datasets" / "semantic_action_dev.json")
    parser.add_argument("--report-output", type=Path, default=ROOT / "datasets" / "semantic_action_data_report.json")
    parser.add_argument("--variants-per-relation", type=int, default=40)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("catalog must be a non-empty list of task-object records")

    # This writes hashes only.  All subsequent operations consume the manifest,
    # never the final bundle.
    write_instruction_hash_manifest(args.final_bundle, args.hash_manifest)
    examples = build_semantic_examples(
        catalog, variants_per_relation=args.variants_per_relation
    )
    assert_no_blind_overlap(examples, [args.hash_manifest])
    train, dev = split_by_relation(examples, dev_variant=args.variants_per_relation)

    _write_json(args.train_output, train)
    _write_json(args.dev_output, dev)
    report = {
        "format": "semantic_action_generated_data_v1",
        "catalog_relations": len(catalog),
        "variants_per_relation": args.variants_per_relation,
        "total_examples": len(examples),
        "train_examples": len(train),
        "dev_examples": len(dev),
        "blind_manifest": str(args.hash_manifest),
        "blind_hash_count": json.loads(args.hash_manifest.read_text(encoding="utf-8"))["count"],
        "final_bundle_used_only_for_hash_freeze": True,
    }
    _write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
