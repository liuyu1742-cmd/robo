"""Train a train/ontology-only object-operation specialist above frozen BGE."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.detailed_action_bge_model import load_local_bge_encoder
from tools.object_operation_augmentation import build_ontology_augmentation_rows
from tools.semantic_action_model import (
    CharacterNgramRelationClassifier,
    build_character_ngram_vocabulary,
    vectorize_batch,
)
from train_detailed_action_bge import _encode_records


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _train_linear_batches(model, inputs, targets, *, epochs, batch_size, lr, weights, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(20260827)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(targets), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            logits = inputs(model, indices)
            loss = nn.functional.cross_entropy(logits, targets[indices].to(device), weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        print(f"epoch={epoch} loss={total / len(targets):.6f}", flush=True)
    model.eval()


def train_specialist(
    *, base_checkpoint: Path, dataset_dir: Path, ontology_path: Path,
    encoder_dir: Path, output: Path, templates_per_phrase: int = 2,
) -> dict:
    torch.manual_seed(20260827)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = _load_json(dataset_dir / "train.json")
    ontology = _load_json(ontology_path)
    augmented = build_ontology_augmentation_rows(
        ontology, templates_per_phrase=templates_per_phrase
    )
    original = [
        row for row in train
        if row.get("expected_executable") is True
        or row.get("safety_class") == "object_operation_incompatibility"
    ]
    rows = [*original, *augmented]
    texts = [str(row["text"]) for row in rows]
    targets = torch.tensor([int(row["expected_executable"] is True) for row in rows])
    counts = torch.bincount(targets, minlength=2).float()
    weights = (counts.sum() / counts.clamp_min(1)).to(device)
    weights /= weights.mean()

    vocabulary = build_character_ngram_vocabulary(texts, min_n=1, max_n=4)
    values, offsets = vectorize_batch(texts, vocabulary)
    character = CharacterNgramRelationClassifier(len(vocabulary), 2, 128).to(device)

    def char_inputs(model, indices):
        selected = [texts[int(index)] for index in indices]
        batch_values, batch_offsets = vectorize_batch(selected, vocabulary, device=torch.device(device))
        return model(batch_values, batch_offsets)

    print(
        f"rows={len(rows)} original={len(original)} augmented={len(augmented)} "
        f"negative={int(counts[0])} positive={int(counts[1])} vocab={len(vocabulary)} device={device}",
        flush=True,
    )
    _train_linear_batches(
        character, char_inputs, targets, epochs=12, batch_size=128, lr=2e-3,
        weights=weights, device=device,
    )

    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    vectors = _encode_records(tokenizer, encoder, rows, device, 128, pooling="cls").cpu()
    bge_head = nn.Linear(vectors.shape[1], 2).to(device)

    def bge_inputs(model, indices):
        return model(vectors[indices].to(device))

    _train_linear_batches(
        bge_head, bge_inputs, targets, epochs=24, batch_size=256, lr=2e-3,
        weights=weights, device=device,
    )

    original_count = len(original)
    original_texts = texts[:original_count]
    original_values, original_offsets = vectorize_batch(
        original_texts, vocabulary, device=torch.device(device)
    )
    with torch.no_grad():
        char_logits = character(original_values, original_offsets).cpu()
        bge_logits = bge_head(vectors[:original_count].to(device)).cpu()
    char_scores = char_logits[:, 1] - char_logits[:, 0]
    bge_scores = bge_logits[:, 1] - bge_logits[:, 0]

    payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    prior = payload["object_operation_specialist"]
    specialist = copy.deepcopy(prior)
    specialist["format"] = "object_operation_specialist_v2_ontology_augmented"
    specialist["character"] = {
        "vocabulary": vocabulary,
        "embedding_dim": 128,
        "state_dict": {key: value.detach().cpu() for key, value in character.state_dict().items()},
    }
    specialist["bge_head_state"] = {
        key: value.detach().cpu() for key, value in bge_head.state_dict().items()
    }
    specialist["config"] = {
        **dict(prior["config"]),
        "character_mean": float(char_scores.mean()),
        "character_std": float(char_scores.std().clamp_min(1e-6)),
        "bge_mean": float(bge_scores.mean()),
        "bge_std": float(bge_scores.std().clamp_min(1e-6)),
        # Start disabled; dev-only threshold calibration decides whether the
        # augmented expert provides a safe improvement.
        "fusion_threshold": -100.0,
    }
    specialist["augmentation"] = {
        "source": "train.json+object_capability_ontology.json",
        "templates_per_phrase": templates_per_phrase,
        "rows": len(augmented),
        "negative_rows": sum(not row["expected_executable"] for row in augmented),
        "positive_rows": sum(bool(row["expected_executable"]) for row in augmented),
        "ontology_sha256": hashlib.sha256(ontology_path.read_bytes()).hexdigest(),
    }
    payload["object_operation_specialist"] = specialist
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "output": str(output),
        "device": device,
        "training_rows": len(rows),
        "class_counts": {"incompatible": int(counts[0]), "executable": int(counts[1])},
        "augmentation": specialist["augmentation"],
        "score_stats": specialist["config"],
    }
    output.with_suffix(".training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=Path("models/detailed_action_bge/best_hybrid_dual_intent_object_operation_v5.pt"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--ontology", type=Path, default=Path("datasets/detailed_action_spoken_v2/object_capability_ontology.json"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--output", type=Path, default=Path("models/detailed_action_bge/candidate_object_operation_ontology_augmented.pt"))
    parser.add_argument("--templates-per-phrase", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(train_specialist(
        base_checkpoint=args.base_checkpoint,
        dataset_dir=args.dataset_dir,
        ontology_path=args.ontology,
        encoder_dir=args.encoder_dir,
        output=args.output,
        templates_per_phrase=args.templates_per_phrase,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
