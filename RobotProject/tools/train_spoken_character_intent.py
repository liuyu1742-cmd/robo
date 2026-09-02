"""Train a 254-label character intent head on spoken train/dev only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.semantic_action_model import (
    CharacterNgramRelationClassifier,
    build_character_ngram_vocabulary,
    vectorize_batch,
)
from tools.semantic_text_normalization import normalize_for_semantic_model
from train_detailed_action_bge import build_catalog_anchor_rows, build_label_order, load_training_splits


def _metrics(logits: torch.Tensor, targets: torch.Tensor, scene: torch.Tensor) -> dict[str, float]:
    correct = logits.argmax(1).eq(targets)
    return {
        "scene": float(correct[scene].float().mean()),
        "object": float(correct[~scene].float().mean()),
        "overall": float(correct.float().mean()),
    }


def train(
    dataset_dir: Path,
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    learning_rate: float,
    anchor_repeats: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, dev_rows = load_training_splits(dataset_dir)
    labels = build_label_order(train_rows)
    lookup = {label: index for index, label in enumerate(labels)}
    train_rows = [row for row in train_rows if row.get("expected_executable") is True]
    train_rows.extend(build_catalog_anchor_rows(labels, anchor_repeats))
    dev_rows = [row for row in dev_rows if row.get("expected_executable") is True]
    train_texts = [normalize_for_semantic_model(str(row["text"])) for row in train_rows]
    dev_texts = [normalize_for_semantic_model(str(row["text"])) for row in dev_rows]
    vocabulary = build_character_ngram_vocabulary(train_texts)
    train_targets = torch.tensor([lookup[str(row["label_id"])] for row in train_rows])
    dev_targets = torch.tensor([lookup[str(row["label_id"])] for row in dev_rows])
    dev_scene = torch.tensor([str(row["label_id"]).startswith("scene:") for row in dev_rows])
    model = CharacterNgramRelationClassifier(
        len(vocabulary), len(labels), embedding_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best = {"overall": -1.0, "epoch": 0, "state": None, "metrics": None}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            texts = [train_texts[int(index)] for index in indices]
            values, offsets = vectorize_batch(texts, vocabulary, device=device)
            logits = model(values, offsets)
            loss = torch.nn.functional.cross_entropy(logits, train_targets[indices].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        model.eval()
        values, offsets = vectorize_batch(dev_texts, vocabulary, device=device)
        with torch.no_grad():
            dev_logits = model(values, offsets).cpu()
        metrics = _metrics(dev_logits, dev_targets, dev_scene)
        if metrics["overall"] > float(best["overall"]):
            best = {
                "overall": metrics["overall"],
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "metrics": metrics,
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps({"epoch": epoch, "loss": total_loss / len(train_rows), **metrics}, ensure_ascii=False), flush=True)
    payload = {
        "format": "spoken_character_intent_v1",
        "labels": labels,
        "vocabulary": vocabulary,
        "embedding_dim": embedding_dim,
        "state_dict": best["state"],
        "normalization": "label_free_zh_v1",
        "training": {
            "split": "train",
            "records": len(train_rows),
            "anchor_repeats": anchor_repeats,
            "epochs": epochs,
            "selected_epoch": best["epoch"],
            "seed": seed,
        },
        "dev_metrics": best["metrics"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {key: value for key, value in payload.items() if key not in ("state_dict", "vocabulary")}
    report["vocabulary_size"] = len(vocabulary)
    output.with_suffix(".training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--output", type=Path, default=Path("models/detailed_action_bge/candidate_spoken_character_intent.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--anchor-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    print(json.dumps(train(
        args.dataset_dir,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        anchor_repeats=args.anchor_repeats,
        seed=args.seed,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
