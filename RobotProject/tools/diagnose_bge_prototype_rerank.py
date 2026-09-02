"""Evaluate train-only BGE label prototypes as a dev intent reranker."""

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

from tools.detailed_action_bge_model import (
    DetailedActionBgeClassifier,
    apply_scene_rescue,
    combine_scene_object_logits,
    load_local_bge_encoder,
    rerank_finite_scores,
    rerank_topk_scores,
)
from tools.detailed_action_runtime import _cosine, _text_features
from train_detailed_action_bge import _encode_records, build_label_order, build_targets, load_training_splits


def _metrics(logits: torch.Tensor, targets: torch.Tensor, scene_rows: torch.Tensor) -> dict:
    correct = logits.argmax(1).eq(targets)
    return {
        "scene": float(correct[scene_rows].float().mean()),
        "object": float(correct[~scene_rows].float().mean()),
        "overall": float(correct.float().mean()),
    }


def diagnose(checkpoint: Path, dataset_dir: Path, encoder_dir: Path, weights: list[float]) -> dict:
    train, dev = load_training_splits(dataset_dir)
    train_exec = [row for row in train if row.get("expected_executable") is True]
    dev_exec = [row for row in dev if row.get("expected_executable") is True]
    labels = build_label_order(train)
    lookup = {label: index for index, label in enumerate(labels)}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    if payload.get("encoder_state"):
        encoder.load_state_dict(payload["encoder_state"])
    encoder.eval()
    pooling = str(payload.get("pooling", "mean"))
    train_vectors = _encode_records(tokenizer, encoder, train_exec, device, 128, pooling=pooling).cpu()
    dev_vectors = _encode_records(tokenizer, encoder, dev_exec, device, 128, pooling=pooling).cpu()

    classifier = DetailedActionBgeClassifier(dev_vectors.shape[1], len(labels))
    classifier.load_state_dict(payload["state_dict"])
    classifier.eval()
    with torch.no_grad():
        primary_raw = classifier(dev_vectors)["intent_logits"]
    label_texts: dict[str, list[str]] = defaultdict(list)
    for row in train_exec:
        label_texts[str(row["label_id"])].append(
            "；".join([str(row["text"]), *row.get("object_mentions", ()), *row.get("operation_evidence", ())])
        )
    lexical_features = {label: _text_features("".join(label_texts[label])) for label in labels}
    lexical = torch.tensor([
        [_cosine(_text_features(str(row["text"])), lexical_features[label]) for label in labels]
        for row in dev_exec
    ])
    rerank = payload["intent_rerank"]
    primary = rerank_topk_scores(primary_raw, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"]))
    specialist = payload["scene_specialist"]
    state = specialist["intent_head_state"]
    scene = torch.nn.functional.linear(dev_vectors, state["weight"], state["bias"])
    scene = rerank_topk_scores(scene, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"]))
    scene_mask = torch.tensor([label.startswith("scene:") for label in labels])
    final = combine_scene_object_logits(primary, scene, scene_mask, scene_bias=float(specialist["scene_bias"]))
    rescue = payload.get("scene_rescue")
    if isinstance(rescue, dict):
        final = apply_scene_rescue(final, primary, scene, scene_mask, threshold=float(rescue["threshold"]))

    train_vectors = torch.nn.functional.normalize(train_vectors, dim=1)
    dev_vectors = torch.nn.functional.normalize(dev_vectors, dim=1)
    train_targets = torch.tensor([lookup[str(row["label_id"])] for row in train_exec])
    similarities = dev_vectors @ train_vectors.T
    centroid_rows = []
    maximum_rows = []
    top3_rows = []
    for label_index in range(len(labels)):
        columns = train_targets.eq(label_index)
        vectors = train_vectors[columns]
        centroid_rows.append(torch.nn.functional.normalize(vectors.mean(0), dim=0))
        values = similarities[:, columns]
        maximum_rows.append(values.max(1).values)
        top3_rows.append(values.topk(min(3, values.shape[1]), dim=1).values.mean(1))
    centroid_scores = dev_vectors @ torch.stack(centroid_rows).T
    prototype_scores = {
        "centroid": centroid_scores,
        "maximum": torch.stack(maximum_rows, dim=1),
        "top3_mean": torch.stack(top3_rows, dim=1),
    }
    targets, _ = build_targets(dev_exec, labels)
    scene_rows = torch.tensor([str(row["label_id"]).startswith("scene:") for row in dev_exec])
    trials = []
    for method, scores in prototype_scores.items():
        trials.append({"method": method, "weight": None, "mode": "prototype_only", **_metrics(scores, targets, scene_rows)})
        for weight in weights:
            fused = rerank_finite_scores(final, scores, weight=weight)
            trials.append({"method": method, "weight": weight, "mode": "finite_rerank", **_metrics(fused, targets, scene_rows)})
    best = max(trials, key=lambda row: (min(row["scene"], row["object"]), row["overall"]))
    return {
        "checkpoint": str(checkpoint),
        "baseline": _metrics(final, targets, scene_rows),
        "best_balanced": best,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--weights", nargs="+", type=float, default=[0.5, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--output", type=Path, default=Path("outputs/detailed_action_bge/prototype_rerank_diagnostics.json"))
    args = parser.parse_args()
    result = diagnose(args.checkpoint, args.dataset_dir, args.encoder_dir, args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
