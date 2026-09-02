"""Dev-only calibration of the scene/object intent balance."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_detailed_action_bge_dev import evaluate_dev


def select_scene_candidate(trials: list[dict], *, minimum_object: float) -> dict | None:
    safe = [row for row in trials if float(row["object"]) >= minimum_object]
    if not safe:
        return None
    return max(
        safe,
        key=lambda row: (float(row["scene"]), float(row["overall"]), float(row["object"])),
    )


def calibrate(
    checkpoint: Path, output: Path, dataset_dir: Path, encoder_dir: Path,
    biases: list[float], minimum_object: float,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("scene_specialist"), dict):
        raise ValueError("checkpoint does not contain scene_specialist")
    trials = []
    with tempfile.TemporaryDirectory(prefix="scene-intent-calibration-") as directory:
        temporary = Path(directory) / "candidate.pt"
        for bias in biases:
            payload["scene_specialist"]["scene_bias"] = bias
            torch.save(payload, temporary)
            report = evaluate_dev(temporary, dataset_dir, encoder_dir)
            row = {
                "bias": bias,
                "scene": report["intent_by_label_type"]["scene"]["accuracy"],
                "object": report["intent_by_label_type"]["object"]["accuracy"],
                "overall": report["intent_accuracy"],
                "joint": report["joint_accuracy"],
                "gate": report["gate_calibrated_accuracy"],
            }
            trials.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    selected = select_scene_candidate(trials, minimum_object=minimum_object)
    if selected is not None:
        payload["scene_specialist"]["scene_bias"] = selected["bias"]
        payload["scene_specialist"]["calibration"] = {
            "split": "dev",
            "minimum_object_accuracy": minimum_object,
            "objective": "maximize_scene_then_overall",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
    result = {
        "checkpoint": str(checkpoint),
        "output": str(output) if selected is not None else None,
        "minimum_object": minimum_object,
        "selected": selected,
        "trials": trials,
    }
    output.with_suffix(".calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def evaluate_without_scene_specialist(
    checkpoint: Path, output: Path, dataset_dir: Path, encoder_dir: Path
) -> dict:
    """Build and evaluate a reversible candidate using only the primary intent head."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    removed = payload.pop("scene_specialist", None)
    if not isinstance(removed, dict):
        raise ValueError("checkpoint does not contain scene_specialist")
    payload["scene_specialist_disabled"] = {
        "reason": "dev diagnosis: primary lexical rerank has higher scene top1",
        "source": removed.get("source"),
        "former_scene_bias": removed.get("scene_bias"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = evaluate_dev(output, dataset_dir, encoder_dir)
    return {
        "scene": report["intent_by_label_type"]["scene"]["accuracy"],
        "object": report["intent_by_label_type"]["object"]["accuracy"],
        "overall": report["intent_accuracy"],
        "joint": report["joint_accuracy"],
        "gate": report["gate_calibrated_accuracy"],
    }


def calibrate_primary_scene_override(
    checkpoint: Path, output: Path, dataset_dir: Path, encoder_dir: Path,
    margins: list[float], minimum_object: float,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    specialist = payload.get("scene_specialist")
    if not isinstance(specialist, dict):
        raise ValueError("checkpoint does not contain scene_specialist")
    trials = []
    with tempfile.TemporaryDirectory(prefix="scene-override-calibration-") as directory:
        temporary = Path(directory) / "candidate.pt"
        for margin in margins:
            specialist["primary_scene_override_margin"] = margin
            torch.save(payload, temporary)
            report = evaluate_dev(temporary, dataset_dir, encoder_dir)
            row = {
                "bias": margin,
                "scene": report["intent_by_label_type"]["scene"]["accuracy"],
                "object": report["intent_by_label_type"]["object"]["accuracy"],
                "overall": report["intent_accuracy"],
                "joint": report["joint_accuracy"],
                "gate": report["gate_calibrated_accuracy"],
            }
            trials.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    selected = select_scene_candidate(trials, minimum_object=minimum_object)
    if selected is not None:
        specialist["primary_scene_override_margin"] = selected["bias"]
        specialist["override_calibration"] = {
            "split": "dev",
            "minimum_object_accuracy": minimum_object,
            "objective": "maximize_scene_then_overall",
        }
        torch.save(payload, output)
    result = {"selected": selected, "trials": trials}
    output.with_suffix(".calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_scene_rescue_candidate(
    checkpoint: Path,
    output: Path,
    dataset_dir: Path,
    encoder_dir: Path,
    threshold: float,
    minimum_object: float,
    method: str = "specialist_gap",
) -> dict:
    """Embed one dev-selected conservative scene rescue rule and evaluate it."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["scene_rescue"] = {
        "method": method,
        "threshold": float(threshold),
        "calibration": {
            "split": "dev",
            "minimum_object_accuracy": float(minimum_object),
            "objective": "improve_scene_without_object_regression",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = evaluate_dev(output, dataset_dir, encoder_dir)
    result = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "threshold": float(threshold),
        "method": method,
        "scene": report["intent_by_label_type"]["scene"]["accuracy"],
        "object": report["intent_by_label_type"]["object"]["accuracy"],
        "overall": report["intent_accuracy"],
        "joint": report["joint_accuracy"],
        "gate": report["gate_calibrated_accuracy"],
    }
    output.with_suffix(".calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--biases", nargs="+", type=float, default=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--minimum-object", type=float, default=0.9149606299212598)
    parser.add_argument("--disable-specialist", action="store_true")
    parser.add_argument("--override-margins", nargs="+", type=float)
    parser.add_argument("--rescue-threshold", type=float)
    parser.add_argument("--rescue-method", choices=("specialist_gap", "lexical_gap"), default="specialist_gap")
    args = parser.parse_args()
    if args.rescue_threshold is not None:
        result = build_scene_rescue_candidate(
            args.checkpoint,
            args.output,
            args.dataset_dir,
            args.encoder_dir,
            args.rescue_threshold,
            args.minimum_object,
            args.rescue_method,
        )
    elif args.override_margins:
        result = calibrate_primary_scene_override(
            args.checkpoint, args.output, args.dataset_dir, args.encoder_dir,
            args.override_margins, args.minimum_object,
        )
    elif args.disable_specialist:
        result = evaluate_without_scene_specialist(
            args.checkpoint, args.output, args.dataset_dir, args.encoder_dir
        )
        args.output.with_suffix(".calibration.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        result = calibrate(
            args.checkpoint, args.output, args.dataset_dir, args.encoder_dir,
            args.biases, args.minimum_object,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
