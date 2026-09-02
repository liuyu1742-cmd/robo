"""Evaluate the frozen assistant-generated semantic-parser self-test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.hybrid_semantic_action_runtime import (
    DEFAULT_CATALOG,
    HybridSemanticActionParser,
    constrain_semantic_actions,
    relation_catalog,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "semantic_action_augmented" / "best.pt"
DEFAULT_BUNDLE = ROOT / "datasets" / "assistant_semantic_selftest_138.json"
DEFAULT_REPORT = ROOT / "models" / "semantic_action_augmented" / "assistant_selftest_report.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "ASSISTANT_SEMANTIC_SELFTEST_REPORT.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_report(report: dict) -> str:
    return "\n".join(
        [
            "# 助手生成自测：语义动作解析报告",
            "",
            "> 证据等级：助手生成的开发自测；不能替代独立人工终测，也不能证明真实用户场景准确率超过 90%。",
            "",
            f"- 样本数：{report['cases']}",
            f"- 关系识别准确率：{report['relation_accuracy']:.2%}",
            f"- 原始整序列准确率：{report['raw_exact_sequence_accuracy']:.2%}",
            f"- 约束后整序列准确率：{report['final_exact_sequence_accuracy']:.2%}",
            f"- 低置信度案例数：{len(report.get('low_confidence_cases', []))}",
            "",
            "原始动作由可训练的关系分类器加标准动作规划器产生；约束后结果只用于一致性核验。",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--device", default=None)
    parser.add_argument("--low-confidence", type=float, default=0.75)
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("evidence_level") != "assistant_generated_development_selftest":
        raise ValueError("bundle is not an assistant-generated development self-test")
    cases = list(bundle.get("cases", []))
    if not cases:
        raise ValueError("self-test contains no cases")
    catalog = relation_catalog(str(args.catalog))
    runtime = HybridSemanticActionParser(args.model, args.catalog, device=args.device)
    relation_correct = raw_correct = final_correct = 0
    confidence_sum = 0.0
    low_confidence_cases: list[dict] = []

    for case in cases:
        resolved, raw_actions = runtime.generate_raw_actions(str(case["instruction"]))
        constrained = constrain_semantic_actions(resolved, raw_actions, catalog)
        relation_ok = resolved.relation_key == case["relation_key"]
        expected_actions = list(case["expected_actions"])
        raw_ok = raw_actions == expected_actions
        final_ok = list(constrained.final_actions) == expected_actions
        relation_correct += int(relation_ok)
        raw_correct += int(raw_ok)
        final_correct += int(final_ok)
        confidence_sum += resolved.confidence
        if resolved.confidence < args.low_confidence or not relation_ok:
            low_confidence_cases.append(
                {
                    "id": case["id"],
                    "predicted_relation": resolved.relation_key,
                    "expected_relation": case["relation_key"],
                    "confidence": resolved.confidence,
                    "relation_correct": relation_ok,
                }
            )

    count = len(cases)
    report = {
        "format": "assistant_semantic_selftest_report_v1",
        "evidence_level": "assistant_generated_development_selftest",
        "not_independent_human_evidence": True,
        "cases": count,
        "relation_accuracy": relation_correct / count,
        "raw_exact_sequence_accuracy": raw_correct / count,
        "final_exact_sequence_accuracy": final_correct / count,
        "mean_relation_confidence": confidence_sum / count,
        "low_confidence_threshold": args.low_confidence,
        "low_confidence_cases": low_confidence_cases,
        "bundle_sha256": _sha256(args.bundle),
        "model_sha256": _sha256(args.model),
        "old_human_final_data_read": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
