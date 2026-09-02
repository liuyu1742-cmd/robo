"""Build the frozen official fixed-command acceptance suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.official_acceptance_data import build_official_acceptance
from tools.semantic_action_data import normalized_instruction_hash


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_FINAL_HASHES = ROOT / "datasets" / "human_only_final_instruction_hashes.json"


def _read_list(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return list(payload["cases"])
    raise ValueError(f"expected rows or bundle cases: {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--final-hash-manifest", type=Path, default=DEFAULT_FINAL_HASHES)
    parser.add_argument("--train", type=Path, default=ROOT / "datasets" / "semantic_action_augmented_train.json")
    parser.add_argument("--dev", type=Path, default=ROOT / "datasets" / "semantic_action_augmented_dev.json")
    parser.add_argument("--assistant-selftest", type=Path, default=ROOT / "datasets" / "assistant_semantic_selftest_138.json")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "official_fixed_command_acceptance_552.json")
    parser.add_argument("--hash-output", type=Path, default=ROOT / "datasets" / "official_fixed_command_acceptance_hashes.json")
    args = parser.parse_args()

    catalog = _read_list(args.catalog)
    exclusions = _read_list(args.train) + _read_list(args.dev) + _read_list(args.assistant_selftest)
    bundle = build_official_acceptance(catalog, exclusions, args.final_hash_manifest)
    bundle["catalog_sha256"] = _sha256(args.catalog)
    _write_json(args.output, bundle)
    hashes = sorted(
        normalized_instruction_hash(str(case["instruction"])) for case in bundle["cases"]
    )
    _write_json(
        args.hash_output,
        {
            "format": "official_fixed_command_acceptance_hashes_v1",
            "count": len(hashes),
            "hashes": hashes,
            "bundle_sha256": _sha256(args.output),
            "catalog_sha256": bundle["catalog_sha256"],
            "evidence_scope": bundle["evidence_scope"],
        },
    )
    print(
        json.dumps(
            {
                "cases": bundle["case_count"],
                "relations": bundle["relations"],
                "minimum_raw_passes_at_90_percent": bundle["minimum_raw_passes_at_90_percent"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
