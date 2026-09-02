"""Offline evaluation for the detailed-action semantic checkpoint."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from tools.semantic_action_model import load_checkpoint, vectorize_batch


def evaluate_semantic_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    *,
    device: torch.device | str = "cpu",
    top_k: int = 3,
) -> dict:
    """Return top-k, scene/object, and confusion metrics for one dataset."""
    rows = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation dataset must be a non-empty list")
    model, vocabulary, labels = load_checkpoint(str(checkpoint_path), device=device)
    values, offsets = vectorize_batch(
        [str(row["text"]) for row in rows], vocabulary, device=torch.device(device),
        normalize_text=getattr(model, "text_normalization", "none") == "label_free_zh_v1",
    )
    with torch.inference_mode():
        indices = model(values, offsets).topk(min(top_k, len(labels)), dim=1).indices.cpu()

    top1_correct = 0
    topk_correct = 0
    group_totals = Counter()
    group_top1 = Counter()
    group_topk = Counter()
    confusions = Counter()
    for row, prediction_indices in zip(rows, indices):
        gold = str(row["label_id"])
        predictions = [labels[int(index)] for index in prediction_indices]
        group = "scene" if gold.startswith("scene:") else "object"
        group_totals[group] += 1
        if predictions[0] == gold:
            top1_correct += 1
            group_top1[group] += 1
        else:
            confusions[(gold, predictions[0])] += 1
        if gold in predictions:
            topk_correct += 1
            group_topk[group] += 1

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "examples": len(rows),
        "top1_accuracy": ratio(top1_correct, len(rows)),
        f"top{top_k}_accuracy": ratio(topk_correct, len(rows)),
        "scene_examples": group_totals["scene"],
        "scene_top1_accuracy": ratio(group_top1["scene"], group_totals["scene"]),
        f"scene_top{top_k}_accuracy": ratio(group_topk["scene"], group_totals["scene"]),
        "object_examples": group_totals["object"],
        "object_top1_accuracy": ratio(group_top1["object"], group_totals["object"]),
        f"object_top{top_k}_accuracy": ratio(group_topk["object"], group_totals["object"]),
        "major_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusions.most_common(20)
        ],
    }
