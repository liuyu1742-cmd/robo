"""Train a character n-gram scene/object router and calibrate downstream intent."""

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
    load_local_bge_encoder,
    rerank_topk_scores,
    route_scene_object_logits,
)
from tools.detailed_action_runtime import _cosine, _text_features
from tools.semantic_action_model import (
    CharacterNgramRelationClassifier,
    build_character_ngram_vocabulary,
    vectorize_batch,
)
from tools.semantic_text_normalization import normalize_for_semantic_model
from train_detailed_action_bge import _encode_records, build_label_order, build_targets, load_training_splits


def _thresholds(scores: torch.Tensor) -> list[float]:
    ordered = sorted(float(value) for value in torch.unique(scores).tolist())
    return [ordered[0] - 1e-6, *[(a + b) / 2 for a, b in zip(ordered, ordered[1:])], ordered[-1] + 1e-6]


def _lexical(train: list[dict], rows: list[dict], labels: list[str]) -> torch.Tensor:
    texts: dict[str, list[str]] = defaultdict(list)
    for row in train:
        if row.get("expected_executable") is True:
            texts[str(row["label_id"])].append(
                "；".join([str(row["text"]), *row.get("object_mentions", ()), *row.get("operation_evidence", ())])
            )
    features = {label: _text_features("".join(texts[label])) for label in labels}
    return torch.tensor([
        [_cosine(_text_features(str(row["text"])), features[label]) for label in labels]
        for row in rows
    ])


def _metrics(logits: torch.Tensor, targets: torch.Tensor, scene_rows: torch.Tensor) -> dict[str, float]:
    correct = logits.argmax(1).eq(targets)
    return {
        "scene": float(correct[scene_rows].float().mean()),
        "object": float(correct[~scene_rows].float().mean()),
        "overall": float(correct.float().mean()),
    }


def train_router(
    checkpoint: Path,
    output: Path,
    dataset_dir: Path,
    encoder_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    learning_rate: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, dev = load_training_splits(dataset_dir)
    train_exec = [row for row in train if row.get("expected_executable") is True]
    dev_exec = [row for row in dev if row.get("expected_executable") is True]
    train_texts = [normalize_for_semantic_model(str(row["text"])) for row in train_exec]
    dev_texts = [normalize_for_semantic_model(str(row["text"])) for row in dev_exec]
    train_targets = torch.tensor([int(str(row["label_id"]).startswith("scene:")) for row in train_exec])
    dev_scene = torch.tensor([bool(str(row["label_id"]).startswith("scene:")) for row in dev_exec])
    vocabulary = build_character_ngram_vocabulary(train_texts)
    head = CharacterNgramRelationClassifier(len(vocabulary), 2, embedding_dim).to(device)
    counts = torch.bincount(train_targets, minlength=2).float()
    weights = (counts.sum() / counts.clamp_min(1)).to(device)
    weights /= weights.mean()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best = {"balanced": -1.0, "epoch": 0, "state": None, "scores": None}
    dev_values, dev_offsets = vectorize_batch(dev_texts, vocabulary, device=device)
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_exec), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            values, offsets = vectorize_batch(
                [train_texts[int(index)] for index in indices], vocabulary, device=device
            )
            logits = head(values, offsets)
            loss = torch.nn.functional.cross_entropy(logits, train_targets[indices].to(device), weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        head.eval()
        with torch.no_grad():
            logits = head(dev_values, dev_offsets).cpu()
        scores = logits[:, 1] - logits[:, 0]
        prediction = scores >= 0
        scene_recall = float(prediction[dev_scene].float().mean())
        object_recall = float((~prediction[~dev_scene]).float().mean())
        balanced = (scene_recall + object_recall) / 2
        if balanced > float(best["balanced"]):
            best = {
                "balanced": balanced,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                "scores": scores.clone(),
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps({
                "epoch": epoch,
                "loss": total_loss / len(train_exec),
                "scene_type_recall": scene_recall,
                "object_type_recall": object_recall,
                "balanced": balanced,
            }, ensure_ascii=False), flush=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = build_label_order(train)
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, str(device))
    if payload.get("encoder_state"):
        encoder.load_state_dict(payload["encoder_state"])
    encoder.eval()
    vectors = _encode_records(tokenizer, encoder, dev_exec, str(device), 128, pooling=str(payload.get("pooling", "mean"))).cpu()
    model = DetailedActionBgeClassifier(vectors.shape[1], len(labels))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        primary = model(vectors)["intent_logits"]
    rerank = payload["intent_rerank"]
    lexical = _lexical(train, dev_exec, labels)
    primary = rerank_topk_scores(primary, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"]))
    specialist = payload["scene_specialist"]
    state = specialist["intent_head_state"]
    scene = torch.nn.functional.linear(vectors, state["weight"], state["bias"])
    scene = rerank_topk_scores(scene, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"]))
    scene_mask = torch.tensor([label.startswith("scene:") for label in labels])
    intent_targets, _ = build_targets(dev_exec, labels)
    trials = []
    for threshold in _thresholds(best["scores"]):
        routed = route_scene_object_logits(
            primary,
            scene,
            scene_mask,
            scene_type_scores=best["scores"],
            threshold=threshold,
        )
        row = {"threshold": threshold, **_metrics(routed, intent_targets, dev_scene)}
        row["scene_type_recall"] = float((best["scores"][dev_scene] >= threshold).float().mean())
        row["object_type_recall"] = float((best["scores"][~dev_scene] < threshold).float().mean())
        trials.append(row)
    selected = max(trials, key=lambda row: (min(row["scene"], row["object"]), row["overall"]))
    payload["scene_type_router"] = {
        "format": "character_ngram_scene_type_router_v1",
        "vocabulary": vocabulary,
        "embedding_dim": embedding_dim,
        "head_state": best["state"],
        "threshold": float(selected["threshold"]),
        "normalization": "label_free_zh_v1",
        "training": {"split": "train", "records": len(train_exec), "epochs": epochs, "selected_epoch": best["epoch"], "seed": seed},
        "calibration": {"split": "dev", "records": len(dev_exec), "objective": "maximize_min_scene_object_then_overall"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    result = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "best_training_epoch": best["epoch"],
        "best_type_balanced_at_zero": best["balanced"],
        "selected": selected,
        "acceptance_met": selected["scene"] >= 0.95 and selected["object"] >= 0.95 and selected["overall"] >= 0.95,
        "top_trials": sorted(trials, key=lambda row: (min(row["scene"], row["object"]), row["overall"]), reverse=True)[:20],
    }
    output.with_suffix(".training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    result = train_router(
        args.checkpoint,
        args.output,
        args.dataset_dir,
        args.encoder_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
