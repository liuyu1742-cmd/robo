"""Evaluate the hybrid parser on its generated, held-out development split.

This is an engineering check only.  It never opens the frozen human final test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.hybrid_semantic_action_runtime import (
    DEFAULT_CATALOG,
    DEFAULT_MODEL,
    HybridSemanticActionParser,
    canonical_actions_for_relation,
    constrain_semantic_actions,
    relation_catalog,
)


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, default=ROOT / "datasets" / "semantic_action_dev.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path, default=ROOT / "models" / "semantic_action" / "development_report.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dev_rows = json.loads(args.dev.read_text(encoding="utf-8"))
    runtime = HybridSemanticActionParser(args.model, args.catalog, device=args.device)
    catalog = relation_catalog(str(args.catalog))
    relation_correct = 0
    raw_correct = 0
    final_correct = 0
    confidence_sum = 0.0
    failures: list[dict] = []

    for row in dev_rows:
        expected_relation = str(row["relation_key"])
        expected_actions = canonical_actions_for_relation(expected_relation, catalog)
        resolved, raw_actions = runtime.generate_raw_actions(str(row["instruction"]))
        constrained = constrain_semantic_actions(resolved, raw_actions, catalog)
        relation_ok = resolved.relation_key == expected_relation
        raw_ok = raw_actions == expected_actions
        final_ok = list(constrained.final_actions) == expected_actions
        relation_correct += int(relation_ok)
        raw_correct += int(raw_ok)
        final_correct += int(final_ok)
        confidence_sum += resolved.confidence
        if not raw_ok:
            failures.append(
                {
                    "id": row["id"],
                    "predicted_relation": resolved.relation_key,
                    "expected_relation": expected_relation,
                    "confidence": resolved.confidence,
                }
            )

    total = len(dev_rows)
    report = {
        "format": "hybrid_semantic_development_report_v1",
        "purpose": "generated development verification; not a human final test",
        "cases": total,
        "relation_accuracy": relation_correct / total,
        "raw_exact_sequence_accuracy": raw_correct / total,
        "final_exact_sequence_accuracy": final_correct / total,
        "mean_relation_confidence": confidence_sum / total,
        "failures": failures,
        "model": str(args.model.resolve()),
        "catalog": str(args.catalog.resolve()),
        "frozen_human_final_data_read": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
