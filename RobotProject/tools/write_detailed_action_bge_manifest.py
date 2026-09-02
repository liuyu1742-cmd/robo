"""Write an auditable manifest for BGE model-stage artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, output: Path) -> dict:
    acceptance_report_path = root / "outputs/detailed_action_bge/dev_metrics_acceptance_v5.json"
    acceptance_report = json.loads(acceptance_report_path.read_text(encoding="utf-8"))
    type_metrics = acceptance_report["intent_by_label_type"]
    safety_metrics = acceptance_report["gate_by_safety_class"]
    acceptance = {
        "intent_accuracy": acceptance_report["intent_accuracy"],
        "object_accuracy": type_metrics["object"]["accuracy"],
        "scene_accuracy": type_metrics["scene"]["accuracy"],
        "gate_accuracy": acceptance_report["gate_calibrated_accuracy"],
        "joint_accuracy": acceptance_report["joint_accuracy"],
        "minimum_safety_accuracy": min(row["accuracy"] for row in safety_metrics.values()),
    }
    dev_accepted = all(value >= 0.95 for value in acceptance.values())
    paths = [
        root / "models/detailed_action_bge/best.pt",
        root / "models/detailed_action_bge/training_report.json",
        root / "models/pretrained/BAAI_bge-small-zh-v1.5/manifest.json",
        root / "datasets/detailed_action_spoken_v2/manifest.json",
        root / "outputs/detailed_action_bge/dev_metrics.json",
        root / "outputs/detailed_action_bge/runtime_timing.json",
        root / "outputs/detailed_action_bge/bge_stage_report.md",
        root / "models/detailed_action_bge/best_hybrid_dual_intent_object_operation_v2.pt",
        root / "models/detailed_action_bge/best_hybrid_dual_intent_object_operation_v3.pt",
        root / "models/detailed_action_bge/best_hybrid_dual_intent_object_operation_v4.pt",
        root / "models/detailed_action_bge/best_hybrid_dual_intent_object_operation_v5.pt",
        root / "outputs/detailed_action_bge/dev_metrics_low_to_high_best.json",
        root / "outputs/detailed_action_bge/dev_metrics_low_to_high_best_v2.json",
        root / "outputs/detailed_action_bge/dev_metrics_acceptance_v5.json",
        root / "datasets/detailed_action_spoken_v2/train_scene_router_qwen_v1.json",
        root / "datasets/detailed_action_spoken_v2/train_object_router_qwen_v1.json",
        root / "models/detailed_action_bge/candidate_object_operation_ontology_refined.calibration.json",
        root / "models/detailed_action_bge/candidate_short_object_gate.calibration.json",
        root / "models/detailed_action_bge/candidate_object_operation_final_refined.calibration.json",
        root / "tools/detailed_action_runtime.py",
        root / "tools/evaluate_detailed_action_bge_dev.py",
        root / "tools/object_operation_augmentation.py",
        root / "tools/train_object_operation_specialist.py",
        root / "tools/calibrate_object_operation_specialist.py",
        root / "tools/calibrate_short_object_gate.py",
        root / "tools/train_augmented_bge_intent_head.py",
        root / "tools/generate_scene_router_augmentation.py",
        root / "tools/generate_object_router_augmentation.py",
        root / "tools/diagnose_scene_rescue.py",
        root / "tools/calibrate_scene_text_rescue.py",
        root / "tools/calibrate_object_text_rescue.py",
        root / "tools/set_character_gate_threshold.py",
    ]
    files = [{"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]
    manifest = {
        "format": "detailed_action_bge_stage_manifest_v1",
        "status": "DEV_ACCEPTED_INDEPENDENT_TEST_FROZEN" if dev_accepted else "CHANGES_REQUIRED",
        "independent_test_policy": "frozen; not read, trained, or tuned in this stage",
        "acceptance_threshold": 0.95,
        "acceptance": acceptance,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/detailed_action_bge/artifact_manifest.json"))
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
