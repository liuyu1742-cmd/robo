"""Run the frozen 414-case human final test exactly once for the hybrid parser."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate_independent_blind_action_test import evaluate_cases
from tools.blind_paraphrase import validate_frozen_blind_bundle
from tools.hybrid_semantic_action_runtime import (
    DEFAULT_CATALOG,
    DEFAULT_MODEL,
    HybridSemanticActionParser,
    constrain_semantic_actions,
    relation_catalog,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "datasets" / "human_only_independent_action_retest_414.json"
DEFAULT_REPORT = ROOT / "models" / "semantic_action" / "final_human_414_report.json"
DEFAULT_LOCK = ROOT / "models" / "semantic_action" / "final_evaluation_lock.json"


def ensure_not_previously_evaluated(lock_path: Path) -> None:
    if lock_path.exists():
        raise RuntimeError(
            "frozen human final evaluation has already been run; do not rerun it "
            "after seeing the result"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(report: dict, threshold: float) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            "# 混合语义动作解析：冻结人工终测报告",
            "",
            f"- 样本数：{metrics['cases']}",
            f"- 原始混合系统整序列准确率：{metrics['raw_exact_sequence_accuracy']:.2%}",
            f"- 约束后整序列准确率：{metrics['final_exact_sequence_accuracy']:.2%}",
            f"- 指令关系解析准确率：{metrics['instruction_parse_accuracy']:.2%}",
            f"- 原始系统是否达到 {threshold:.0%}：{'通过' if metrics['raw_exact_sequence_accuracy'] >= threshold else '未通过'}",
            "",
            "说明：原始分数先由可训练的文本关系分类器和标准动作规划器得到；"
            "约束后分数仅用于安全一致性核验，不能替代原始分数。",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs" / "HYBRID_SEMANTIC_FINAL_HUMAN_EVALUATION.md")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ensure_not_previously_evaluated(args.lock)
    if args.report.exists():
        raise RuntimeError("final report already exists; refusing to overwrite it")

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    catalog = relation_catalog(str(args.catalog))
    validate_frozen_blind_bundle(bundle, sorted(catalog))
    runtime = HybridSemanticActionParser(args.model, args.catalog, device=args.device)

    report = evaluate_cases(
        bundle,
        resolver=runtime.resolve,
        generator=lambda resolved: runtime.generate_raw_actions_for_resolved(resolved),
        constraint=lambda resolved, actions: constrain_semantic_actions(
            resolved, list(actions), catalog
        ),
    )
    report.update(
        {
            "evaluation_protocol": "frozen human final evaluated once after model lock",
            "model": str(args.model.resolve()),
            "catalog": str(args.catalog.resolve()),
            "bundle": str(args.bundle.resolve()),
            "threshold": args.threshold,
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(_markdown(report, args.threshold), encoding="utf-8")
    lock = {
        "format": "one_time_frozen_final_evaluation_lock_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_sha256": _sha256(args.bundle),
        "model_sha256": _sha256(args.model),
        "report_sha256": _sha256(args.report),
        "raw_exact_sequence_accuracy": report["metrics"]["raw_exact_sequence_accuracy"],
        "final_exact_sequence_accuracy": report["metrics"]["final_exact_sequence_accuracy"],
    }
    args.lock.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_exact_sequence_accuracy": report["metrics"]["raw_exact_sequence_accuracy"],
                "final_exact_sequence_accuracy": report["metrics"]["final_exact_sequence_accuracy"],
                "instruction_parse_accuracy": report["metrics"]["instruction_parse_accuracy"],
                "threshold_passed": report["metrics"]["raw_exact_sequence_accuracy"] >= args.threshold,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
