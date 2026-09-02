"""Train and dev-calibrate a scene/object router without opening independent test."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

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
from train_detailed_action_bge import (
    _encode_records,
    build_label_order,
    build_targets,
    load_training_splits,
)


def select_router_candidate(trials: list[dict], *, minimum_object: float) -> dict | None:
    """Maximize scene accuracy, then overall, without crossing the object floor."""
    safe = [row for row in trials if float(row["object"]) >= float(minimum_object)]
    if not safe:
        return None
    return max(
        safe,
        key=lambda row: (float(row["scene"]), float(row["overall"]), float(row["object"])),
    )


def _lexical_scores(train: list[dict], records: list[dict], labels: list[str]) -> torch.Tensor:
    label_texts: dict[str, list[str]] = defaultdict(list)
    for row in train:
        if row.get("expected_executable") is True:
            label_texts[str(row["label_id"])].append(
                "；".join(
                    [str(row["text"]), *row.get("object_mentions", ()), *row.get("operation_evidence", ())]
                )
            )
    label_features = {label: _text_features("".join(label_texts[label])) for label in labels}
    return torch.tensor(
        [
            [_cosine(_text_features(str(row["text"])), label_features[label]) for label in labels]
            for row in records
        ]
    )


def _thresholds(scores: torch.Tensor) -> list[float]:
    ordered = sorted(float(value) for value in torch.unique(scores).tolist())
    if not ordered:
        raise ValueError("router calibration scores are empty")
    result = [ordered[0] - 1e-6]
    result.extend((left + right) / 2.0 for left, right in zip(ordered, ordered[1:]))
    result.append(ordered[-1] + 1e-6)
    return result


def _intent_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    scene_targets: torch.Tensor,
) -> dict[str, float]:
    prediction = logits.argmax(1)
    correct = prediction.eq(targets)
    return {
        "scene": float(correct[scene_targets].float().mean()),
        "object": float(correct[~scene_targets].float().mean()),
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
    learning_rate: float,
    minimum_object: float,
    seed: int,
    augmentation: list[Path] | None = None,
    balance_classes: bool = True,
    architecture: str = "linear",
) -> dict:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = load_training_splits(dataset_dir)
    train_executable = [row for row in train if row.get("expected_executable") is True]
    for augmentation_path in augmentation or ():
        augmented_rows = json.loads(augmentation_path.read_text(encoding="utf-8"))
        train_executable.extend(row for row in augmented_rows if row.get("expected_executable") is True)
    dev_executable = [row for row in dev if row.get("expected_executable") is True]
    labels = build_label_order(train)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if list(payload.get("labels", labels)) != labels:
        raise ValueError("checkpoint labels do not match train labels")
    specialist = payload.get("scene_specialist")
    rerank = payload.get("intent_rerank")
    if not isinstance(specialist, dict) or not isinstance(rerank, dict):
        raise ValueError("router training requires scene specialist and intent rerank")

    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    if payload.get("encoder_state"):
        encoder.load_state_dict(payload["encoder_state"])
    encoder.eval()
    pooling = str(payload.get("pooling", "mean"))
    train_vectors = _encode_records(
        tokenizer, encoder, train_executable, device, batch_size, pooling=pooling
    ).cpu()
    dev_vectors = _encode_records(
        tokenizer, encoder, dev_executable, device, batch_size, pooling=pooling
    ).cpu()
    train_type = torch.tensor(
        [int(str(row["label_id"]).startswith("scene:")) for row in train_executable]
    )
    dev_type = torch.tensor(
        [bool(str(row["label_id"]).startswith("scene:")) for row in dev_executable]
    )

    if architecture == "linear":
        head = nn.Linear(train_vectors.shape[1], 2).to(device)
    elif architecture == "mlp":
        head = nn.Sequential(
            nn.Linear(train_vectors.shape[1], 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 2),
        ).to(device)
    else:
        raise ValueError("architecture must be linear or mlp")
    counts = torch.bincount(train_type, minlength=2).float()
    weights = None
    if balance_classes:
        weights = (counts.sum() / counts.clamp_min(1)).to(device)
        weights = weights / weights.mean()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best = {"balanced_accuracy": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_vectors), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            logits = head(train_vectors[indices].to(device))
            loss = nn.functional.cross_entropy(logits, train_type[indices].to(device), weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        head.eval()
        with torch.no_grad():
            dev_prediction = head(dev_vectors.to(device)).argmax(1).cpu().bool()
        scene_recall = float(dev_prediction[dev_type].float().mean())
        object_recall = float((~dev_prediction[~dev_type]).float().mean())
        balanced = (scene_recall + object_recall) / 2.0
        if balanced > float(best["balanced_accuracy"]):
            best = {
                "balanced_accuracy": balanced,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": total_loss / len(train_vectors),
                        "scene_type_recall": scene_recall,
                        "object_type_recall": object_recall,
                        "balanced_accuracy": balanced,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    head.load_state_dict(best["state"])
    head.eval()

    classifier = DetailedActionBgeClassifier(dev_vectors.shape[1], len(labels))
    classifier.load_state_dict(payload["state_dict"])
    classifier.eval()
    with torch.no_grad():
        primary = classifier(dev_vectors)["intent_logits"]
        type_logits = head(dev_vectors.to(device)).cpu()
    lexical = _lexical_scores(train, dev_executable, labels)
    primary = rerank_topk_scores(
        primary, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"])
    )
    scene_state = specialist["intent_head_state"]
    scene_logits = nn.functional.linear(dev_vectors, scene_state["weight"], scene_state["bias"])
    scene_logits = rerank_topk_scores(
        scene_logits, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"])
    )
    scene_mask = torch.tensor([label.startswith("scene:") for label in labels])
    intent_targets, _ = build_targets(dev_executable, labels)
    type_scores = type_logits[:, 1] - type_logits[:, 0]
    trials = []
    for threshold in _thresholds(type_scores):
        routed = route_scene_object_logits(
            primary,
            scene_logits,
            scene_mask,
            scene_type_scores=type_scores,
            threshold=threshold,
        )
        trials.append(
            {
                "threshold": threshold,
                **_intent_metrics(routed, intent_targets, dev_type),
                "scene_type_recall": float((type_scores[dev_type] >= threshold).float().mean()),
                "object_type_recall": float((type_scores[~dev_type] < threshold).float().mean()),
            }
        )
    selected = select_router_candidate(trials, minimum_object=minimum_object)
    baseline = _intent_metrics(
        route_scene_object_logits(
            primary,
            scene_logits,
            scene_mask,
            scene_type_scores=torch.full_like(type_scores, -1.0),
            threshold=0.0,
        ),
        intent_targets,
        dev_type,
    )
    accepted = selected is not None and float(selected["scene"]) > 0.8027210884353742
    if accepted:
        payload["scene_type_router"] = {
            "format": f"{architecture}_scene_type_router_v1",
            "head_state": best["state"],
            "threshold": float(selected["threshold"]),
            "training": {
                "split": "train",
                "records": len(train_executable),
                "class_counts": {"object": int(counts[0]), "scene": int(counts[1])},
                "epochs": epochs,
                "selected_epoch": int(best["epoch"]),
                "learning_rate": learning_rate,
                "seed": seed,
                "augmentation": [str(path) for path in augmentation or ()],
                "balance_classes": balance_classes,
                "architecture": architecture,
            },
            "calibration": {
                "split": "dev",
                "records": len(dev_executable),
                "minimum_object_accuracy": minimum_object,
                "objective": "maximize_scene_then_overall_without_object_regression",
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)

    sorted_trials = sorted(
        trials,
        key=lambda row: (float(row["scene"]), float(row["overall"]), float(row["object"])),
        reverse=True,
    )
    result = {
        "checkpoint": str(checkpoint),
        "output": str(output) if accepted else None,
        "device": device,
        "best_training_epoch": best["epoch"],
        "best_training_balanced_accuracy": best["balanced_accuracy"],
        "minimum_object": minimum_object,
        "baseline_restricted_primary": baseline,
        "selected": selected,
        "accepted": accepted,
        "top_trials": sorted_trials[:20],
    }
    report_path = output.with_suffix(".training.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-object", type=float, default=0.9149606299212598)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--augmentation", nargs="+", type=Path)
    parser.add_argument("--no-balance-classes", action="store_true")
    parser.add_argument("--architecture", choices=("linear", "mlp"), default="linear")
    args = parser.parse_args()
    result = train_router(
        args.checkpoint,
        args.output,
        args.dataset_dir,
        args.encoder_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        minimum_object=args.minimum_object,
        seed=args.seed,
        augmentation=args.augmentation,
        balance_classes=not args.no_balance_classes,
        architecture=args.architecture,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
