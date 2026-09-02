"""Restore model-locked runtime vocabularies from a retained checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "action_transformer_project_generation_semantic.pt"


def ordered_values(vocabulary: dict[str, int], excluded: set[str] = set()) -> list[str]:
    return [
        value
        for value, _ in sorted(vocabulary.items(), key=lambda item: item[1])
        if value not in excluded
    ]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    action_vocab = checkpoint["action_vocab"]
    argument_vocab = checkpoint["argument_vocab"]
    write_json(
        ROOT / "meta" / "project_action_vocab.json",
        {
            "format": "checkpoint_locked_project_action_vocab_v1",
            "checkpoint": CHECKPOINT.name,
            "tasks": ordered_values(checkpoint["task_vocab"]),
            "objects": ordered_values(checkpoint["object_vocab"]),
            "targets": ordered_values(checkpoint["target_vocab"]),
            "actions": ordered_values(action_vocab, {"<pad>", "<bos>", "<eos>"}),
            "arguments": ordered_values(argument_vocab, {"<pad>", "<none>"}),
        },
    )
    write_json(
        ROOT / "meta" / "project_condition_vocab.json",
        {
            "format": "checkpoint_locked_project_condition_vocab_v1",
            "checkpoint": CHECKPOINT.name,
            "sources": ordered_values(checkpoint["source_vocab"]),
            "verbs": ordered_values(checkpoint["verb_vocab"]),
        },
    )


if __name__ == "__main__":
    main()
