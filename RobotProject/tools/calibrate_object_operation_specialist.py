"""Dev-only calibration for the object-operation specialist veto threshold."""

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


def select_safe_candidate(reports: list[dict], *, minimum_gate: float, minimum_core: float) -> dict | None:
    safe = [
        row for row in reports
        if float(row["gate"]) >= minimum_gate and float(row["core"]) >= minimum_core
    ]
    if not safe:
        return None
    return max(
        safe,
        key=lambda row: (
            float(row["incompat"]), float(row["gate"]), float(row["joint"]),
            -abs(float(row["threshold"])),
        ),
    )


def calibrate(
    candidate: Path, output: Path, dataset_dir: Path, encoder_dir: Path,
    thresholds: list[float], minimum_gate: float, minimum_core: float,
) -> dict:
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="object-operation-calibration-") as directory:
        temporary = Path(directory) / "candidate.pt"
        for threshold in thresholds:
            payload["object_operation_specialist"]["config"]["fusion_threshold"] = threshold
            torch.save(payload, temporary)
            report = evaluate_dev(temporary, dataset_dir, encoder_dir)
            safety = report["gate_by_safety_class"]
            row = {
                "threshold": threshold,
                "gate": report["gate_calibrated_accuracy"],
                "joint": report["joint_accuracy"],
                "core": safety["core_executable"]["accuracy"],
                "incompat": safety["object_operation_incompatibility"]["accuracy"],
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    selected = select_safe_candidate(
        rows, minimum_gate=minimum_gate, minimum_core=minimum_core
    )
    if selected is not None:
        payload["object_operation_specialist"]["config"]["fusion_threshold"] = selected["threshold"]
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
    result = {
        "candidate": str(candidate),
        "output": str(output) if selected is not None else None,
        "constraints": {"minimum_gate": minimum_gate, "minimum_core": minimum_core},
        "selected": selected,
        "trials": rows,
    }
    output.with_suffix(".calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-20, -12, -8, -6, -5, -4, -3, -2, -1, 0])
    parser.add_argument("--minimum-gate", type=float, default=0.950106143951416)
    parser.add_argument("--minimum-core", type=float, default=0.9606299212598425)
    args = parser.parse_args()
    result = calibrate(
        args.candidate, args.output, args.dataset_dir, args.encoder_dir,
        args.thresholds, args.minimum_gate, args.minimum_core,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
