"""Train the detailed-action BGE heads using frozen train/dev data only."""

from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn

from tools.detailed_action_bge_model import (
    DetailedActionBgeClassifier,
    cls_pool_embeddings,
    encode_bge_texts,
    load_local_bge_encoder,
    mean_pool_embeddings,
    supervised_contrastive_loss,
)
from tools.detailed_action_catalog import CATALOGUE_PATH, load_detailed_action_catalog
from tools.detailed_action_semantic_data import catalog_sha256


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(dataset_dir: str | Path, encoder_dir: str | Path) -> dict[str, str]:
    encoder_manifest = Path(encoder_dir) / "manifest.json"
    dataset_manifest = Path(dataset_dir) / "manifest.json"
    encoder_info = json.loads(encoder_manifest.read_text(encoding="utf-8")) if encoder_manifest.exists() else {}
    return {
        "catalog_sha256": catalog_sha256(CATALOGUE_PATH),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest) if dataset_manifest.exists() else "",
        "encoder_revision": str(encoder_info.get("revision", "")),
        "encoder_manifest_sha256": _sha256_file(encoder_manifest) if encoder_manifest.exists() else "",
    }


def load_training_splits(dataset_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Load the two permitted splits without opening independent_test.json."""
    root = Path(dataset_dir)
    train = json.loads((root / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((root / "dev.json").read_text(encoding="utf-8"))
    return train, dev


def build_label_order(records: Iterable[Mapping[str, object]]) -> list[str]:
    """Return the stable executable-label vocabulary learned from train only."""
    return sorted(
        {
            label
            for record in records
            if record.get("expected_executable") is True
            and isinstance((label := record.get("label_id")), str)
        }
    )


def build_targets(records: Iterable[Mapping[str, object]], labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create intent targets (ignored for rejections) and executable gate targets."""
    lookup = {label: index for index, label in enumerate(labels)}
    items = list(records)
    return (
        torch.tensor([lookup.get(item.get("label_id"), -100) for item in items]),
        torch.tensor([int(item.get("expected_executable") is True) for item in items]),
    )


def build_training_text(record: Mapping[str, object], mode: str = "raw") -> str:
    """Return an allowed train-side representation without using dev/test metadata."""
    if mode == "raw":
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("record text must be a string")
        return text
    if mode != "evidence":
        raise ValueError("training text mode must be raw or evidence")
    parts: list[str] = []
    for key in ("object_mentions", "operation_evidence"):
        values = record.get(key, ())
        if isinstance(values, (list, tuple)):
            parts.extend(str(value) for value in values if isinstance(value, str) and value.strip())
    if not parts:
        if record.get("expected_executable") is False or record.get("catalog_anchor") is True:
            text = record.get("text")
            if isinstance(text, str):
                return text
        raise ValueError("evidence mode requires object_mentions or operation_evidence")
    return "；".join(parts)


def build_catalog_anchor_text(entry: Mapping[str, object]) -> str:
    """Create a concise, auditable label-definition anchor from the reviewed catalog."""
    if entry.get("entry_type") == "object":
        actions = entry.get("detailed_actions", ())
        if not isinstance(actions, list) or not actions:
            raise ValueError("object catalog entry requires detailed_actions")
        return f"{entry['object_name_zh']}；{actions[0]}"
    if entry.get("entry_type") == "scene":
        objects = entry.get("objects", ())
        names = [str(item["object_name_zh"]) for item in objects if isinstance(item, Mapping)]
        return "；".join([str(entry["command"]), *names])
    raise ValueError("catalog anchor requires object or scene entry")


def build_catalog_anchor_rows(labels: list[str], repeats: int) -> list[dict]:
    if repeats < 0:
        raise ValueError("catalog anchor repeats must be non-negative")
    entries = {entry["label"]: entry for entry in load_detailed_action_catalog(CATALOGUE_PATH)["entries"]}
    rows: list[dict] = []
    for label in labels:
        if label not in entries:
            raise ValueError(f"catalog is missing training label: {label}")
        rows.extend({"text": build_catalog_anchor_text(entries[label]), "label_id": label, "expected_executable": True, "catalog_anchor": True} for _ in range(repeats))
    return rows


def calibrate_gate_threshold(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Choose a deterministic binary threshold using dev scores only."""
    scores = scores.detach().float().cpu().flatten()
    targets = targets.detach().long().cpu().flatten()
    if scores.numel() != targets.numel() or scores.numel() == 0:
        raise ValueError("scores and targets must be non-empty and have equal length")
    candidates = torch.unique(scores).tolist()
    candidates += [float(scores.min()) - 1e-6, float(scores.max()) + 1e-6]
    best = (-1.0, 0.0)
    for threshold in sorted(candidates):
        accuracy = float(((scores >= threshold).long() == targets).float().mean())
        if accuracy > best[0] or (accuracy == best[0] and threshold < best[1]):
            best = (accuracy, threshold)
    return {"threshold": float(best[1]), "accuracy": float(best[0])}


def merge_dual_head_states(intent_state: Mapping[str, torch.Tensor], gate_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Combine independently selected intent and executable-gate head weights."""
    result = {key: value.detach().cpu().clone() for key, value in intent_state.items()}
    for key, value in gate_state.items():
        if key.startswith("gate_head."):
            result[key] = value.detach().cpu().clone()
    return result


def output_to_cpu(output: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach both model heads before CPU-only dev metric calculation."""
    return {key: value.detach().cpu() for key, value in output.items()}


def shuffled_batch_indices(length: int, batch_size: int, *, seed: int) -> list[torch.Tensor]:
    """Return one deterministic shuffled epoch covering every row exactly once."""
    if length <= 0 or batch_size <= 0:
        raise ValueError("length and batch_size must be positive")
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed))
    return [order[start : start + batch_size] for start in range(0, length, batch_size)]


def pool_training_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool trainable encoder outputs exactly as the resumed checkpoint expects."""
    if pooling == "mean":
        return mean_pool_embeddings(hidden, attention_mask)
    if pooling == "cls":
        return cls_pool_embeddings(hidden)
    raise ValueError("unsupported pooling mode")


def load_frozen_head_checkpoint(
    model: DetailedActionBgeClassifier,
    payload: Mapping[str, object],
    labels: list[str],
) -> None:
    """Restore the selected frozen heads only when their label columns match."""
    if list(payload.get("labels", ())) != labels:
        raise ValueError("resume checkpoint label order does not match train labels")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint does not contain classifier state_dict")
    model.load_state_dict(state)


def _encode_records(
    tokenizer,
    encoder,
    records: list[dict],
    device: str,
    batch_size: int,
    pooling: str = "mean",
    text_mode: str = "raw",
) -> torch.Tensor:
    return torch.cat([
        encode_bge_texts(
            tokenizer,
            encoder,
            [build_training_text(row, text_mode) for row in records[index:index + batch_size]],
            device,
            pooling=pooling,
        )
        for index in range(0, len(records), batch_size)
    ])


def train_frozen_heads(dataset_dir: str | Path, encoder_dir: str | Path, output: str | Path,
                       epochs: int = 5, batch_size: int = 32, lr: float = 1e-3,
                       pooling: str = "mean", balance_gate: bool = False,
                       intent_text_mode: str = "raw", gate_text_mode: str = "raw",
                       catalog_anchor_repeats: int = 0) -> dict:
    """Train BGE heads and retain each head at its own dev-optimal epoch."""
    torch.manual_seed(20260814)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = load_training_splits(dataset_dir)
    labels = build_label_order(train)
    train = [*train, *build_catalog_anchor_rows(labels, catalog_anchor_repeats)]
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    intent_train_vectors = _encode_records(
        tokenizer, encoder, train, device, batch_size, pooling=pooling, text_mode=intent_text_mode,
    ).cpu()
    gate_train_vectors = _encode_records(
        tokenizer, encoder, train, device, batch_size, pooling=pooling, text_mode=gate_text_mode,
    ).cpu()
    dev_vectors = _encode_records(tokenizer, encoder, dev, device, batch_size, pooling=pooling).cpu()
    train_intent, train_gate = build_targets(train, labels)
    dev_intent, dev_gate = build_targets(dev, labels)
    model = DetailedActionBgeClassifier(intent_train_vectors.shape[1], len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    gate_weight = None
    if balance_gate:
        gate_counts = torch.bincount(train_gate, minlength=2).float()
        gate_weight = (gate_counts.sum() / gate_counts.clamp_min(1)).to(device)
        gate_weight = gate_weight / gate_weight.mean()
    best_intent = {"intent_accuracy": -1.0, "state": None, "epoch": 0}
    best_gate = {"gate_calibrated_accuracy": -1.0, "state": None, "epoch": 0, "threshold": 0.0}
    for epoch in range(1, epochs + 1):
        model.train()
        for start in range(0, len(train), batch_size):
            end = start + batch_size
            intent_logits = model(intent_train_vectors[start:end].to(device))["intent_logits"]
            gate_logits = model(gate_train_vectors[start:end].to(device))["gate_logits"]
            intent_loss = nn.functional.cross_entropy(intent_logits, train_intent[start:end].to(device), ignore_index=-100)
            gate_loss = nn.functional.cross_entropy(gate_logits, train_gate[start:end].to(device), weight=gate_weight)
            optimizer.zero_grad(); (intent_loss + gate_loss).backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_output = model(dev_vectors.to(device))
            gate_accuracy = (dev_output["gate_logits"].argmax(1).cpu() == dev_gate).float().mean().item()
            gate_scores = dev_output["gate_logits"][:, 1].cpu() - dev_output["gate_logits"][:, 0].cpu()
            gate_calibration = calibrate_gate_threshold(gate_scores, dev_gate)
            gate_calibrated_accuracy = float(
                ((gate_scores >= gate_calibration["threshold"]).long() == dev_gate).float().mean()
            )
            valid = dev_intent.ne(-100)
            intent_accuracy = (dev_output["intent_logits"].argmax(1).cpu()[valid] == dev_intent[valid]).float().mean().item()
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if intent_accuracy > best_intent["intent_accuracy"]:
            best_intent = {"intent_accuracy": intent_accuracy, "state": state, "epoch": epoch}
        if gate_calibrated_accuracy > best_gate["gate_calibrated_accuracy"]:
            best_gate = {
                "gate_accuracy": gate_accuracy,
                "gate_calibrated_accuracy": gate_calibrated_accuracy,
                "threshold": gate_calibration["threshold"],
                "state": state,
                "epoch": epoch,
            }
    merged_state = merge_dual_head_states(best_intent["state"], best_gate["state"])
    model.load_state_dict(merged_state)
    model.eval()
    with torch.no_grad():
        merged_output = output_to_cpu(model(dev_vectors.to(device)))
    merged_gate_scores = merged_output["gate_logits"][:, 1] - merged_output["gate_logits"][:, 0]
    merged_gate = (merged_gate_scores >= best_gate["threshold"]).long()
    merged_intent = merged_output["intent_logits"].argmax(1)
    valid = dev_intent.ne(-100)
    best = {
        "intent_accuracy": float((merged_intent[valid] == dev_intent[valid]).float().mean()),
        "gate_accuracy": float((merged_output["gate_logits"].argmax(1) == dev_gate).float().mean()),
        "gate_calibrated_accuracy": float((merged_gate == dev_gate).float().mean()),
        "gate_threshold": float(best_gate["threshold"]),
        "joint_accuracy": float(((merged_gate == dev_gate) & (~dev_gate.bool() | (merged_intent == dev_intent))).float().mean()),
        "intent_epoch": int(best_intent["epoch"]),
        "gate_epoch": int(best_gate["epoch"]),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "detailed_action_bge_v1", "labels": labels, "metrics": best,
                "pooling": pooling, "intent_text_mode": intent_text_mode, "gate_text_mode": gate_text_mode,
                **_checkpoint_metadata(dataset_dir, encoder_dir),
                "state_dict": merged_state}, output)
    return best


def train_unfreeze_last_layer(dataset_dir: str | Path, encoder_dir: str | Path, output: str | Path,
                              epochs: int = 2, batch_size: int = 16, lr: float = 2e-5,
                              gate_loss_weight: float = 0.2, contrastive_loss_weight: float = 0.1,
                              evidence_loss_weight: float = 0.25,
                              resume: str | Path | None = None) -> dict:
    """Fine-tune BGE's final two layers from a selected frozen-head checkpoint."""
    if resume is None:
        raise ValueError("unfreeze-last-2 requires --resume frozen checkpoint")
    torch.manual_seed(20260814)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = load_training_splits(dataset_dir)
    labels = build_label_order(train)
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad_(name.startswith("encoder.layer.2") or name.startswith("encoder.layer.3") or name.startswith("pooler."))
    model = DetailedActionBgeClassifier(encoder.config.hidden_size, len(labels)).to(device)
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    load_frozen_head_checkpoint(model, payload, labels)
    pooling = str(payload.get("pooling", "mean"))
    optimizer = torch.optim.AdamW([*filter(lambda p: p.requires_grad, encoder.parameters()), *model.parameters()], lr=lr)
    train_intent, train_gate = build_targets(train, labels)
    dev_intent, dev_gate = build_targets(dev, labels)
    gate_counts = torch.bincount(train_gate, minlength=2).float()
    gate_weight = (gate_counts.sum() / gate_counts.clamp_min(1)).to(device)
    gate_weight = gate_weight / gate_weight.mean()
    best = {"joint_accuracy": -1.0}

    def evaluate(epoch: int) -> tuple[dict, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        encoder.eval(); model.eval(); intent_logits = []; gate_logits = []
        with torch.no_grad():
            for start in range(0, len(dev), batch_size):
                rows = dev[start:start + batch_size]
                encoded = tokenizer([row["text"] for row in rows], padding=True, truncation=True, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                pooled = pool_training_hidden(encoder(**encoded).last_hidden_state, encoded["attention_mask"], pooling)
                out = model(pooled)
                intent_logits.append(out["intent_logits"].detach().cpu())
                gate_logits.append(out["gate_logits"].detach().cpu())
        intent_all = torch.cat(intent_logits); gate_all = torch.cat(gate_logits)
        gate_scores = gate_all[:, 1] - gate_all[:, 0]
        calibration = calibrate_gate_threshold(gate_scores, dev_gate)
        gate_pred = (gate_scores >= calibration["threshold"]).long()
        intent_pred = intent_all.argmax(1); valid = dev_intent.ne(-100)
        metrics = {
            "epoch": epoch,
            "intent_accuracy": float((intent_pred[valid] == dev_intent[valid]).float().mean()),
            "gate_accuracy": float((gate_all.argmax(1) == dev_gate).float().mean()),
            "gate_calibrated_accuracy": float((gate_pred == dev_gate).float().mean()),
            "gate_threshold": float(calibration["threshold"]),
            "joint_accuracy": float(((gate_pred == dev_gate) & (~dev_gate.bool() | (intent_pred == dev_intent))).float().mean()),
        }
        return metrics, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}

    for epoch in range(0, epochs + 1):
        if epoch == 0:
            metrics, model_state, encoder_state = evaluate(epoch)
        else:
            encoder.train(); model.train()
            for indices in shuffled_batch_indices(
                len(train), batch_size, seed=20260814 + epoch
            ):
                rows = [train[int(index)] for index in indices]
                raw_encoded = tokenizer([row["text"] for row in rows], padding=True, truncation=True, return_tensors="pt")
                evidence_encoded = tokenizer([build_training_text(row, "evidence") for row in rows], padding=True, truncation=True, return_tensors="pt")
                raw_encoded = {key: value.to(device) for key, value in raw_encoded.items()}
                evidence_encoded = {key: value.to(device) for key, value in evidence_encoded.items()}
                raw_pooled = pool_training_hidden(encoder(**raw_encoded).last_hidden_state, raw_encoded["attention_mask"], pooling)
                evidence_pooled = pool_training_hidden(encoder(**evidence_encoded).last_hidden_state, evidence_encoded["attention_mask"], pooling)
                intent_targets = train_intent[indices].to(device)
                output_intent = model(raw_pooled)
                output_evidence = model(evidence_pooled)
                output_gate = model(raw_pooled)
                executable = intent_targets.ne(-100)
                if executable.any():
                    intent_loss = nn.functional.cross_entropy(
                        output_intent["intent_logits"], intent_targets, ignore_index=-100
                    )
                    evidence_loss = nn.functional.cross_entropy(
                        output_evidence["intent_logits"], intent_targets, ignore_index=-100
                    )
                else:
                    intent_loss = output_intent["intent_logits"].sum() * 0.0
                    evidence_loss = output_evidence["intent_logits"].sum() * 0.0
                gate_loss = nn.functional.cross_entropy(output_gate["gate_logits"], train_gate[indices].to(device), weight=gate_weight)
                contrastive_loss = supervised_contrastive_loss(
                    torch.cat([raw_pooled[executable], evidence_pooled[executable]]),
                    torch.cat([intent_targets[executable], intent_targets[executable]]),
                )
                optimizer.zero_grad()
                (
                    intent_loss
                    + evidence_loss_weight * evidence_loss
                    + gate_loss_weight * gate_loss
                    + contrastive_loss_weight * contrastive_loss
                ).backward()
                optimizer.step()
            metrics, model_state, encoder_state = evaluate(epoch)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if metrics["joint_accuracy"] > best["joint_accuracy"]:
            best = metrics
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format":"detailed_action_bge_v1","labels":labels,"metrics":best,"pooling":pooling, **_checkpoint_metadata(dataset_dir, encoder_dir), "encoder_state":encoder_state,"state_dict":model_state},output)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("frozen", "unfreeze-last-2"), default="frozen")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--output", type=Path, default=Path("models/detailed_action_bge/best.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-loss-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--balance-gate", action="store_true")
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--intent-text-mode", choices=("raw", "evidence"), default="raw")
    parser.add_argument("--gate-text-mode", choices=("raw", "evidence"), default="raw")
    parser.add_argument("--catalog-anchor-repeats", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "frozen":
        metrics = train_frozen_heads(
            args.dataset_dir, args.encoder, args.output,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, pooling=args.pooling,
            balance_gate=args.balance_gate, intent_text_mode=args.intent_text_mode,
            gate_text_mode=args.gate_text_mode, catalog_anchor_repeats=args.catalog_anchor_repeats,
        )
    else:
        metrics = train_unfreeze_last_layer(
            args.dataset_dir, args.encoder, args.output,
            epochs=args.epochs, batch_size=min(args.batch_size, 16), lr=min(args.lr, 2e-5),
            gate_loss_weight=args.gate_loss_weight, resume=args.resume,
            contrastive_loss_weight=args.contrastive_loss_weight,
            evidence_loss_weight=args.evidence_loss_weight,
        )
    report = {
        "stage": args.stage,
        "dataset_dir": str(args.dataset_dir),
        "encoder_dir": str(args.encoder),
        "output": str(args.output),
        "train_records": len(load_training_splits(args.dataset_dir)[0]),
        "dev_records": len(load_training_splits(args.dataset_dir)[1]),
        "independent_test_policy": "frozen and not read by training",
        "metrics": metrics,
    }
    report_path = args.output.parent / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
