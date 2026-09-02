"""Train a local character n-gram classifier for all detailed action labels."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from tools.detailed_action_catalog import CATALOGUE_PATH, load_detailed_action_catalog
from tools.detailed_action_semantic_data import (
    DEV_PATH,
    TRAIN_PATH,
    catalog_sha256,
    write_detailed_action_splits,
)
from tools.semantic_action_model import (
    build_character_ngram_vocabulary,
    make_tfidf_centroid_checkpoint,
    predict_relations,
)
from tools.semantic_text_normalization import normalize_for_semantic_model
from tools.detailed_action_semantic_evaluation import evaluate_semantic_checkpoint


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "detailed_action_semantic" / "best.pt"
REPORT_PATH = ROOT / "models" / "detailed_action_semantic" / "training_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOGUE_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--dev", type=Path, default=DEV_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


@torch.inference_mode()
def _accuracy(
    model: torch.nn.Module,
    rows: list[dict],
    vocabulary: dict[str, int],
    labels: list[str],
    device: torch.device,
) -> float:
    predictions = predict_relations(
        model, vocabulary, labels, [row["text"] for row in rows], device=device
    )
    correct = sum(
        prediction["relation_key"] == row["label_id"]
        for prediction, row in zip(predictions, rows)
    )
    return correct / len(rows)


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def configure_deterministic_training(seed: int) -> dict[str, bool]:
    """Seed all RNGs and require deterministic PyTorch/cuDNN execution."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    deterministic_settings = configure_deterministic_training(args.seed)
    device = _select_device(args.device)

    catalog = load_detailed_action_catalog(args.catalog)
    train, dev = write_detailed_action_splits(
        catalog, train_path=args.train, dev_path=args.dev
    )
    labels = sorted(entry["label"] for entry in catalog["entries"])
    train_labels = {row["label_id"] for row in train}
    if train_labels != set(labels):
        raise ValueError("training data label set does not match the catalogue")

    started = time.perf_counter()
    vocabulary = build_character_ngram_vocabulary(
        normalize_for_semantic_model(row["text"]) for row in train
    )
    checkpoint = make_tfidf_centroid_checkpoint(
        train, vocabulary, labels, text_normalization="label_free_zh_v1"
    )
    checkpoint.update(
        {
            "catalog_sha256": catalog_sha256(args.catalog),
            "train_examples": len(train),
            "dev_examples": len(dev),
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": 1,
            "deterministic_settings": deterministic_settings,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_output)
    metrics = evaluate_semantic_checkpoint(
        args.model_output, args.dev, device=device, top_k=3
    )
    checkpoint["best_dev_accuracy"] = metrics["top1_accuracy"]
    checkpoint["dev_metrics"] = metrics
    torch.save(checkpoint, args.model_output)
    print(
        f"fit=tfidf_centroid ngram=1-4 dev_accuracy={metrics['top1_accuracy']:.2%} "
        f"dev_top3_accuracy={metrics['top3_accuracy']:.2%}"
    )

    elapsed_seconds = time.perf_counter() - started
    report = {
        "format": "detailed_action_semantic_training_report_v1",
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": 1,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "labels": len(labels),
        "vocabulary_size": len(vocabulary),
        "best_dev_accuracy": metrics["top1_accuracy"],
        "top3_accuracy": metrics["top3_accuracy"],
        "scene_top1_accuracy": metrics["scene_top1_accuracy"],
        "scene_top3_accuracy": metrics["scene_top3_accuracy"],
        "object_top1_accuracy": metrics["object_top1_accuracy"],
        "object_top3_accuracy": metrics["object_top3_accuracy"],
        "major_confusions": metrics["major_confusions"],
        "model_format": checkpoint["format"],
        "feature_ngram_range": checkpoint["feature_ngram_range"],
        "scoring": checkpoint["scoring"],
        "text_normalization": checkpoint["text_normalization"],
        "deterministic_settings": deterministic_settings,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "elapsed_seconds": elapsed_seconds,
        "catalog_sha256": catalog_sha256(args.catalog),
        "model": str(args.model_output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
