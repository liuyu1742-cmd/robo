"""Diagnostic-only train/dev learning curves for the frozen BGE dual heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from tools.detailed_action_bge_model import DetailedActionBgeClassifier, load_local_bge_encoder
from train_detailed_action_bge import _encode_records, build_label_order, build_targets, load_training_splits


def trace_frozen_training(
    dataset_dir: str | Path,
    encoder_dir: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    shuffle: bool = False,
    train_text_mode: str = "raw",
) -> list[dict]:
    torch.manual_seed(20260814)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = load_training_splits(dataset_dir)
    if train_text_mode in {"evidence", "hybrid"}:
        evidence_rows = [
            {**row, "text": "；".join([*row.get("object_mentions", ()), *row.get("operation_evidence", ())])}
            for row in train
        ]
        train = evidence_rows if train_text_mode == "evidence" else [*train, *evidence_rows]
    elif train_text_mode != "raw":
        raise ValueError("train_text_mode must be raw, evidence, or hybrid")
    labels = build_label_order(train)
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    train_vectors = _encode_records(tokenizer, encoder, train, device, batch_size).cpu()
    dev_vectors = _encode_records(tokenizer, encoder, dev, device, batch_size).cpu()
    train_intent, train_gate = build_targets(train, labels)
    dev_intent, dev_gate = build_targets(dev, labels)
    model = DetailedActionBgeClassifier(train_vectors.shape[1], len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train)) if shuffle else torch.arange(len(train))
        for start in range(0, len(train), batch_size):
            indices = order[start:start + batch_size]
            output = model(train_vectors[indices].to(device))
            loss = nn.functional.cross_entropy(
                output["intent_logits"], train_intent[indices].to(device), ignore_index=-100
            ) + nn.functional.cross_entropy(output["gate_logits"], train_gate[indices].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(dev_vectors.to(device))
        valid = dev_intent.ne(-100)
        intent = float((output["intent_logits"].cpu().argmax(1)[valid] == dev_intent[valid]).float().mean())
        gate = float((output["gate_logits"].cpu().argmax(1) == dev_gate).float().mean())
        joint = float(
            ((output["gate_logits"].cpu().argmax(1) == dev_gate)
             & (~dev_gate.bool() | (output["intent_logits"].cpu().argmax(1) == dev_intent))).float().mean()
        )
        history.append({"epoch": epoch, "intent_accuracy": intent, "gate_accuracy": gate, "joint_accuracy": joint})
    return history


def trace_split_input_heads(
    dataset_dir: str | Path,
    encoder_dir: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> list[dict]:
    """Train intent on concise evidence and gate on raw text, then trace dev."""
    torch.manual_seed(20260814)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = load_training_splits(dataset_dir)
    intent_rows = [
        {**row, "text": "；".join([*row.get("object_mentions", ()), *row.get("operation_evidence", ())])}
        for row in train
    ]
    labels = build_label_order(train)
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    intent_vectors = _encode_records(tokenizer, encoder, intent_rows, device, batch_size).cpu()
    gate_vectors = _encode_records(tokenizer, encoder, train, device, batch_size).cpu()
    dev_vectors = _encode_records(tokenizer, encoder, dev, device, batch_size).cpu()
    train_intent, train_gate = build_targets(train, labels)
    dev_intent, dev_gate = build_targets(dev, labels)
    model = DetailedActionBgeClassifier(intent_vectors.shape[1], len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for start in range(0, len(train), batch_size):
            end = start + batch_size
            intent_logits = model(intent_vectors[start:end].to(device))["intent_logits"]
            gate_logits = model(gate_vectors[start:end].to(device))["gate_logits"]
            loss = nn.functional.cross_entropy(intent_logits, train_intent[start:end].to(device), ignore_index=-100)
            loss += nn.functional.cross_entropy(gate_logits, train_gate[start:end].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(dev_vectors.to(device))
        valid = dev_intent.ne(-100)
        intent = float((output["intent_logits"].cpu().argmax(1)[valid] == dev_intent[valid]).float().mean())
        gate = float((output["gate_logits"].cpu().argmax(1) == dev_gate).float().mean())
        joint = float(
            ((output["gate_logits"].cpu().argmax(1) == dev_gate)
             & (~dev_gate.bool() | (output["intent_logits"].cpu().argmax(1) == dev_intent))).float().mean()
        )
        history.append({"epoch": epoch, "intent_accuracy": intent, "gate_accuracy": gate, "joint_accuracy": joint})
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--train-text-mode", choices=("raw", "evidence", "hybrid"), default="raw")
    parser.add_argument("--split-head-inputs", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/detailed_action_bge/frozen_training_trace.json"))
    args = parser.parse_args()
    if args.split_head_inputs:
        result = trace_split_input_heads("datasets/detailed_action_spoken_v2", "models/pretrained/BAAI_bge-small-zh-v1.5", epochs=args.epochs)
    else:
        result = trace_frozen_training("datasets/detailed_action_spoken_v2", "models/pretrained/BAAI_bge-small-zh-v1.5", epochs=args.epochs, shuffle=args.shuffle, train_text_mode=args.train_text_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_intent": max(result, key=lambda row: row["intent_accuracy"]), "best_gate": max(result, key=lambda row: row["gate_accuracy"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
