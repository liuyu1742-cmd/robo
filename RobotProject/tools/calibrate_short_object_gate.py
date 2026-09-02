"""Dev-only calibration of the generic short known-object margin gate."""

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

from tools.calibrate_object_operation_specialist import select_safe_candidate
from tools.evaluate_detailed_action_bge_dev import evaluate_dev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-chars", type=int, default=12)
    parser.add_argument("--margins", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    args = parser.parse_args()

    payload = torch.load(args.candidate, map_location="cpu", weights_only=False)
    trials = []
    with tempfile.TemporaryDirectory(prefix="short-object-calibration-") as directory:
        temporary = Path(directory) / "candidate.pt"
        for margin in args.margins:
            payload["short_object_gate"] = {
                "maximum_chars": args.maximum_chars,
                "additional_margin": margin,
            }
            torch.save(payload, temporary)
            report = evaluate_dev(temporary, args.dataset_dir, args.encoder_dir)
            safety = report["gate_by_safety_class"]
            row = {
                "threshold": margin,
                "gate": report["gate_calibrated_accuracy"],
                "joint": report["joint_accuracy"],
                "core": safety["core_executable"]["accuracy"],
                "incompat": safety["same_name_object_ambiguity"]["accuracy"],
            }
            trials.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    selected = select_safe_candidate(
        trials, minimum_gate=0.9532908797264099, minimum_core=0.9606299212598425
    )
    if selected is not None:
        payload["short_object_gate"] = {
            "maximum_chars": args.maximum_chars,
            "additional_margin": selected["threshold"],
        }
        torch.save(payload, args.output)
    result = {"selected": selected, "trials": trials}
    args.output.with_suffix(".calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
