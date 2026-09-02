"""Create a checkpoint candidate with a dev-calibrated character-gate threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("threshold", type=float)
    parser.add_argument("--known-only", action="store_true")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.known_only:
        payload["known_object_gate_threshold"] = float(args.threshold)
    else:
        payload["character_gate"]["threshold"] = float(args.threshold)
    payload["character_gate"]["threshold_calibration"] = {
        "split": "dev",
        "independent_test_used": False,
        "objective": "maximize_joint_subject_to_each_safety_class_at_least_0.95",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
