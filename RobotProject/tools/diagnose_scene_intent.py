"""Train/dev-only diagnostics for the BGE scene intent head."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.detailed_action_bge_model import (
    DetailedActionBgeClassifier,
    combine_scene_object_logits,
    load_local_bge_encoder,
    rerank_topk_scores,
)
from tools.detailed_action_runtime import _cosine, _text_features
from train_detailed_action_bge import _encode_records, build_label_order, build_targets, load_training_splits


def _accuracy_at(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    return float(logits.topk(k, dim=1).indices.eq(targets.unsqueeze(1)).any(1).float().mean())


def diagnose(checkpoint: Path, dataset_dir: Path, encoder_dir: Path) -> dict:
    train, dev = load_training_splits(dataset_dir)
    labels = build_label_order(train)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("encoder_state"):
        encoder.load_state_dict(payload["encoder_state"])
        encoder.eval()
    pooling = str(payload.get("pooling", "mean"))
    vectors = _encode_records(tokenizer, encoder, dev, device, 64, pooling=pooling).cpu()
    model = DetailedActionBgeClassifier(vectors.shape[1], len(labels))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        primary = model(vectors)["intent_logits"]
    _, targets = None, None
    intent_targets, _ = build_targets(dev, labels)

    label_texts = defaultdict(list)
    for row in train:
        if row.get("expected_executable") is True:
            label_texts[str(row["label_id"])].append(
                "；".join([str(row["text"]), *row.get("object_mentions", ()), *row.get("operation_evidence", ())])
            )
    label_features = {label: _text_features("".join(label_texts[label])) for label in labels}
    lexical = torch.tensor([
        [_cosine(_text_features(str(row["text"])), label_features[label]) for label in labels]
        for row in dev
    ])
    rerank = payload["intent_rerank"]
    primary_reranked = rerank_topk_scores(
        primary, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"])
    )
    specialist = payload["scene_specialist"]
    state = specialist["intent_head_state"]
    scene = torch.nn.functional.linear(vectors, state["weight"], state["bias"])
    scene_reranked = rerank_topk_scores(
        scene, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"])
    )
    scene_mask = torch.tensor([label.startswith("scene:") for label in labels])
    final = combine_scene_object_logits(
        primary_reranked, scene_reranked, scene_mask,
        scene_bias=float(specialist["scene_bias"]),
    )

    indices = torch.tensor([
        index for index, row in enumerate(dev)
        if row.get("expected_executable") is True and str(row.get("label_id", "")).startswith("scene:")
    ])
    target = intent_targets[indices]
    stages = {"primary": primary[indices], "primary_reranked": primary_reranked[indices], "final": final[indices]}
    errors = []
    prediction_types = Counter()
    confusion = Counter()
    final_predictions = stages["final"].argmax(1)
    for dev_index, gold_index, predicted_index in zip(indices, target, final_predictions):
        gold = labels[int(gold_index)]
        predicted = labels[int(predicted_index)]
        prediction_types["scene" if predicted.startswith("scene:") else "object"] += 1
        if gold != predicted:
            confusion[(gold, predicted)] += 1
            ranking = stages["final"][len(errors) if False else 0]
            errors.append({
                "sample_id": dev[int(dev_index)]["sample_id"],
                "gold": gold,
                "predicted": predicted,
                "gold_primary_rank": int((primary[int(dev_index)] > primary[int(dev_index), int(gold_index)]).sum()) + 1,
                "gold_scene_rank": int((scene[int(dev_index)] > scene[int(dev_index), int(gold_index)]).sum()) + 1,
            })
    return {
        "checkpoint": str(checkpoint),
        "scene_records": len(indices),
        "stages": {
            name: {"top1": _accuracy_at(logits, target, 1), "top3": _accuracy_at(logits, target, 3), "top5": _accuracy_at(logits, target, 5)}
            for name, logits in stages.items()
        },
        "final_prediction_types": dict(prediction_types),
        "top_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(20)
        ],
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--output", type=Path, default=Path("outputs/detailed_action_bge/scene_intent_diagnostics.json"))
    args = parser.parse_args()
    result = diagnose(args.checkpoint, args.dataset_dir, args.encoder_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scene_records", "stages", "final_prediction_types", "top_confusions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
