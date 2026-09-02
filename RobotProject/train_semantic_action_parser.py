"""Train the relation classifier for the hybrid semantic action parser.

It reads generated train/dev files only.  It does not open any human blind
evaluation bundle.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from tools.semantic_action_model import (
    CharacterNgramRelationClassifier,
    build_vocabulary,
    make_checkpoint,
    predict_relations,
    vectorize_batch,
)


ROOT = Path(__file__).resolve().parent


def _read_examples(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected non-empty example list: {path}")
    required = {"instruction", "relation_key"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"malformed semantic examples: {path}")
    return rows


def _accuracy(
    model: CharacterNgramRelationClassifier,
    rows: list[dict],
    vocabulary: dict[str, int],
    label_to_index: dict[str, int],
    labels: list[str],
    device: torch.device,
) -> float:
    predictions = predict_relations(
        model, vocabulary, labels, [row["instruction"] for row in rows], device=device
    )
    return sum(
        prediction["relation_key"] == row["relation_key"]
        for prediction, row in zip(predictions, rows)
    ) / len(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "datasets" / "semantic_action_train.json")
    parser.add_argument("--dev", type=Path, default=ROOT / "datasets" / "semantic_action_dev.json")
    parser.add_argument("--model-output", type=Path, default=ROOT / "models" / "semantic_action" / "best.pt")
    parser.add_argument("--report-output", type=Path, default=ROOT / "models" / "semantic_action" / "dev_report.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    train = _read_examples(args.train)
    dev = _read_examples(args.dev)
    labels = sorted({row["relation_key"] for row in train})
    label_to_index = {label: index for index, label in enumerate(labels)}
    if set(row["relation_key"] for row in dev) != set(labels):
        raise ValueError("development data must contain one example for every relation")

    vocabulary = build_vocabulary(row["instruction"] for row in train)
    model = CharacterNgramRelationClassifier(
        len(vocabulary), len(labels), args.embedding_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = [train[index] for index in order[start : start + args.batch_size]]
            values, offsets = vectorize_batch(
                [row["instruction"] for row in batch], vocabulary, device=device
            )
            target = torch.tensor(
                [label_to_index[row["relation_key"]] for row in batch],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(values, offsets), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch)

        model.eval()
        dev_accuracy = _accuracy(model, dev, vocabulary, label_to_index, labels, device)
        if dev_accuracy > best_accuracy:
            best_accuracy, best_epoch = dev_accuracy, epoch
            args.model_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                make_checkpoint(
                    model, vocabulary, labels, embedding_dim=args.embedding_dim
                ),
                args.model_output,
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={total_loss / len(train):.4f} "
                f"dev_relation_accuracy={dev_accuracy:.2%}"
            )

    report = {
        "format": "semantic_action_training_report_v1",
        "device": str(device),
        "seed": args.seed,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "relations": len(labels),
        "vocabulary_size": len(vocabulary),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_dev_relation_accuracy": best_accuracy,
        "model": str(args.model_output),
        "blind_final_data_read": False,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
