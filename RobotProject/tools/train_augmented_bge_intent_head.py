"""Train the BGE intent head with symmetric Qwen train-only augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.detailed_action_bge_model import DetailedActionBgeClassifier, load_local_bge_encoder
from train_detailed_action_bge import (
    _encode_records,
    build_catalog_anchor_rows,
    build_label_order,
    build_targets,
    load_training_splits,
    shuffled_batch_indices,
)


def _metrics(logits: torch.Tensor, targets: torch.Tensor, scene: torch.Tensor) -> dict[str, float]:
    correct = logits.argmax(1).eq(targets)
    return {
        "scene": float(correct[scene].float().mean()),
        "object": float(correct[~scene].float().mean()),
        "overall": float(correct.float().mean()),
    }


def train(
    checkpoint: Path,
    output: Path,
    dataset_dir: Path,
    encoder_dir: Path,
    augmentations: list[Path],
    *,
    anchor_repeats: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, dev_rows = load_training_splits(dataset_dir)
    labels = build_label_order(train_rows)
    rows = [row for row in train_rows if row.get("expected_executable") is True]
    for path in augmentations:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    rows.extend(build_catalog_anchor_rows(labels, anchor_repeats))
    dev_exec = [row for row in dev_rows if row.get("expected_executable") is True]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    if payload.get("encoder_state"):
        encoder.load_state_dict(payload["encoder_state"])
    encoder.eval()
    pooling = str(payload.get("pooling", "mean"))
    train_vectors = _encode_records(tokenizer, encoder, rows, device, batch_size, pooling=pooling).cpu()
    dev_vectors = _encode_records(tokenizer, encoder, dev_exec, device, batch_size, pooling=pooling).cpu()
    targets, _ = build_targets(rows, labels)
    dev_targets, _ = build_targets(dev_exec, labels)
    dev_scene = torch.tensor([str(row["label_id"]).startswith("scene:") for row in dev_exec])
    model = DetailedActionBgeClassifier(train_vectors.shape[1], len(labels)).to(device)
    model.load_state_dict(payload["state_dict"])
    for parameter in model.gate_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.intent_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    best = {"overall": -1.0, "epoch": 0, "state": None, "metrics": None}
    for epoch in range(0, epochs + 1):
        if epoch:
            model.train()
            total_loss = 0.0
            for indices in shuffled_batch_indices(len(rows), batch_size, seed=seed + epoch):
                logits = model(train_vectors[indices].to(device))["intent_logits"]
                loss = torch.nn.functional.cross_entropy(logits, targets[indices].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
        else:
            total_loss = 0.0
        model.eval()
        with torch.no_grad():
            logits = model(dev_vectors.to(device))["intent_logits"].cpu()
        metrics = _metrics(logits, dev_targets, dev_scene)
        if metrics["overall"] > float(best["overall"]):
            best = {
                "overall": metrics["overall"],
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.intent_head.state_dict().items()},
                "metrics": metrics,
            }
        if epoch == 0 or epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps({"epoch": epoch, "loss": total_loss / len(rows) if epoch else None, **metrics}, ensure_ascii=False), flush=True)
    state = dict(payload["state_dict"])
    state["intent_head.weight"] = best["state"]["weight"]
    state["intent_head.bias"] = best["state"]["bias"]
    payload["state_dict"] = state
    payload["augmented_intent_head"] = {
        "format": "qwen_symmetric_train_only_v1",
        "augmentations": [str(path) for path in augmentations],
        "anchor_repeats": anchor_repeats,
        "records": len(rows),
        "epochs": epochs,
        "selected_epoch": best["epoch"],
        "learning_rate": learning_rate,
        "seed": seed,
        "dev_raw_metrics": best["metrics"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    result = {"checkpoint": str(checkpoint), "output": str(output), **payload["augmented_intent_head"]}
    output.with_suffix(".training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--augmentation", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--anchor-repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    print(json.dumps(train(
        args.checkpoint, args.output, args.dataset_dir, args.encoder_dir, args.augmentation,
        anchor_repeats=args.anchor_repeats, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
