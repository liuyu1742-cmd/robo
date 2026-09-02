"""Evaluate the real detailed-action runtime on dev only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.detailed_action_runtime import DetailedActionParser
from train_detailed_action_bge import load_training_splits


def evaluate(checkpoint: Path, dataset_dir: Path, encoder_dir: Path) -> dict:
    _, dev = load_training_splits(dataset_dir)
    parser = DetailedActionParser(
        semantic_backend="bge",
        bge_checkpoint_path=checkpoint,
        bge_encoder_path=encoder_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    intent: dict[str, list[bool]] = defaultdict(list)
    gate: dict[str, list[bool]] = defaultdict(list)
    results = []
    for index, row in enumerate(dev, 1):
        resolved = parser.resolve(str(row["text"]))
        executable = resolved.get("result_type") != "clarification_required"
        expected = row.get("expected_executable") is True
        gate[str(row.get("safety_class") or "core_executable")].append(executable == expected)
        if expected:
            kind = str(row["label_id"]).split(":", 1)[0]
            intent[kind].append(executable and resolved.get("label") == row.get("label_id"))
        results.append(
            {
                "sample_id": row.get("sample_id"),
                "expected_label": row.get("label_id"),
                "predicted_label": resolved.get("label"),
                "expected_executable": expected,
                "predicted_executable": executable,
                "reason": resolved.get("reason"),
            }
        )
        if index % 100 == 0 or index == len(dev):
            print(json.dumps({"processed": index, "total": len(dev)}, ensure_ascii=False), flush=True)
    all_intent = [value for values in intent.values() for value in values]
    all_gate = [value for values in gate.values() for value in values]
    return {
        "checkpoint": str(checkpoint),
        "dev_records": len(dev),
        "intent_accuracy": sum(all_intent) / len(all_intent),
        "intent_by_label_type": {
            key: {"accuracy": sum(values) / len(values), "records": len(values)}
            for key, values in sorted(intent.items())
        },
        "gate_accuracy": sum(all_gate) / len(all_gate),
        "gate_by_safety_class": {
            key: {"accuracy": sum(values) / len(values), "records": len(values)}
            for key, values in sorted(gate.items())
        },
        "errors": [
            row for row in results
            if row["predicted_executable"] != row["expected_executable"]
            or (row["expected_executable"] and row["predicted_label"] != row["expected_label"])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--output", type=Path, default=Path("outputs/detailed_action_bge/runtime_dev_metrics.json"))
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.dataset_dir, args.encoder_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "errors"}
    summary["error_count"] = len(result["errors"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
