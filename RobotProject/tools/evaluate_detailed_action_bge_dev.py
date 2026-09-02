"""Evaluate a BGE checkpoint on the permitted train/dev calibration split."""

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
    apply_scene_rescue,
    apply_scene_text_rescue,
    apply_object_text_rescue,
    apply_object_surface_override,
    apply_short_object_margin_gate,
    apply_object_operation_specialist_gate,
    combine_scene_object_logits,
    combine_scene_object_logits_conditionally,
    load_character_gate,
    load_local_bge_encoder,
    load_object_operation_specialist,
    rerank_topk_scores,
)
from tools.detailed_action_runtime import (
    _cosine,
    _has_cancelling_negation,
    _has_object_attribute_mismatch,
    _has_unresolved_instance_reference,
    _has_unresolved_choice,
    _is_information_request,
    _is_out_of_domain_request,
    _is_completed_statement,
    _is_restrictive_attribute_command,
    _semantic_text,
    _text_features,
)
from tools.semantic_action_model import vectorize_batch
from train_detailed_action_bge import (
    _encode_records,
    build_label_order,
    build_targets,
    calibrate_gate_threshold,
    load_training_splits,
)


def _object_surface_candidate_mask(
    train: list[dict], rows: list[dict], labels: list[str]
) -> torch.Tensor:
    label_index = {label: index for index, label in enumerate(labels)}
    surfaces: dict[str, set[int]] = defaultdict(set)
    for row in train:
        label = str(row.get("label_id") or "")
        if row.get("expected_executable") is True and label.startswith("object:"):
            for surface in row.get("object_mentions", ()):
                surface = str(surface).strip()
                if len(surface) >= 2:
                    surfaces[surface].add(label_index[label])
    catalog_path = PROJECT_ROOT / "meta" / "detailed_action_entry_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    for entry in catalog["entries"]:
        if entry.get("entry_type") == "object":
            surfaces[str(entry["object_name_zh"])].add(label_index[str(entry["label"])])
    mask = torch.zeros((len(rows), len(labels)), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        text = str(row["text"])
        matches = [surface for surface in surfaces if surface in text]
        maximal = [
            surface for surface in matches
            if not any(surface != other and surface in other for other in matches)
        ]
        for surface in maximal:
            mask[row_index, list(surfaces[surface])] = True
    return mask


def evaluate_dev(checkpoint: str | Path, dataset_dir: str | Path, encoder_dir: str | Path) -> dict:
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
    intent_targets, gate_targets = build_targets(dev, labels)
    with torch.no_grad():
        output = model(vectors)
    rerank = payload.get("intent_rerank")
    if isinstance(rerank, dict) and rerank.get("method") == "train_lexical_prototype":
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
        output["intent_logits"] = rerank_topk_scores(
            output["intent_logits"], lexical,
            top_k=int(rerank["top_k"]), weight=float(rerank["weight"]),
        )
        specialist = payload.get("scene_specialist")
        if isinstance(specialist, dict):
            primary_logits = output["intent_logits"]
            state = specialist["intent_head_state"]
            scene_logits = torch.nn.functional.linear(vectors, state["weight"], state["bias"])
            scene_logits = rerank_topk_scores(
                scene_logits, lexical,
                top_k=int(rerank["top_k"]), weight=float(rerank["weight"]),
            )
            scene_mask = torch.tensor([label.startswith("scene:") for label in labels])
            if "primary_scene_override_margin" in specialist:
                output["intent_logits"] = combine_scene_object_logits_conditionally(
                    output["intent_logits"], scene_logits, scene_mask,
                    scene_bias=float(specialist["scene_bias"]),
                    primary_scene_override_margin=float(specialist["primary_scene_override_margin"]),
                )
            else:
                output["intent_logits"] = combine_scene_object_logits(
                    output["intent_logits"], scene_logits, scene_mask,
                    scene_bias=float(specialist["scene_bias"]),
                )
            if isinstance(payload.get("object_surface_override"), dict):
                surface_config = payload["object_surface_override"]
                surface_candidate_mask = _object_surface_candidate_mask(train, dev, labels)
                output["intent_logits"] = apply_object_surface_override(
                    output["intent_logits"],
                    primary_logits,
                    scene_mask,
                    surface_candidate_mask,
                    scene_guard_logits=(
                        scene_logits if "scene_guard_margin" in surface_config else None
                    ),
                    scene_guard_margin=surface_config.get("scene_guard_margin"),
                )
            rescue = payload.get("scene_rescue")
            if isinstance(rescue, dict) and rescue.get("method") in ("specialist_gap", "lexical_gap"):
                output["intent_logits"] = apply_scene_rescue(
                    output["intent_logits"],
                    primary_logits,
                    scene_logits,
                    scene_mask,
                    threshold=float(rescue["threshold"]),
                    method=str(rescue["method"]),
                    auxiliary_scores=lexical,
                )
            text_rescue = payload.get("scene_text_rescue")
            if isinstance(text_rescue, dict):
                output["intent_logits"] = apply_scene_text_rescue(
                    output["intent_logits"],
                    scene_logits,
                    scene_mask,
                    [str(row["text"]) for row in dev],
                    [str(pattern) for pattern in text_rescue.get("patterns", ())],
                )
            object_text_rescue = payload.get("object_text_rescue")
            if isinstance(object_text_rescue, dict):
                if "surface_candidate_mask" not in locals():
                    surface_candidate_mask = _object_surface_candidate_mask(train, dev, labels)
                output["intent_logits"] = apply_object_text_rescue(
                    output["intent_logits"],
                    primary_logits,
                    scene_mask,
                    surface_candidate_mask,
                    [str(row["text"]) for row in dev],
                    [str(pattern) for pattern in object_text_rescue.get("patterns", ())],
                )
    if payload.get("character_gate"):
        character_gate, gate_vocabulary, stored_threshold = load_character_gate(payload)
        values, offsets = vectorize_batch([str(row["text"]) for row in dev], gate_vocabulary)
        with torch.no_grad():
            character_logits = character_gate(values, offsets)
        gate_scores = character_logits[:, 1] - character_logits[:, 0]
        gate_argmax = character_logits.argmax(1)
        calibrated = {"threshold": stored_threshold, "accuracy": float(((gate_scores >= stored_threshold).long() == gate_targets).float().mean())}
        gate_backend = "character_ngram"
    else:
        gate_scores = output["gate_logits"][:, 1] - output["gate_logits"][:, 0]
        gate_argmax = output["gate_logits"].argmax(1)
        calibrated = calibrate_gate_threshold(gate_scores, gate_targets)
        gate_backend = "bge"
    gate_calibrated = (gate_scores >= calibrated["threshold"]).long()
    gate_thresholds = torch.full_like(gate_scores, float(calibrated["threshold"]))
    specialist_report = None
    known_object_mask = torch.zeros(len(dev), dtype=torch.bool)
    if payload.get("object_operation_specialist"):
        (
            specialist_character,
            specialist_vocabulary,
            specialist_bge_head,
            capability_labels,
            capability_vectors,
            specialist_config,
        ) = load_object_operation_specialist(payload)
        specialist_texts = [
            str(row["text"]).strip()
            if str(row["text"]).strip().endswith(("。", "！", "？", "!", "?"))
            else str(row["text"]).strip() + "。"
            for row in dev
        ]
        specialist_values, specialist_offsets = vectorize_batch(
            specialist_texts, specialist_vocabulary
        )
        with torch.no_grad():
            specialist_character_logits = specialist_character(
                specialist_values, specialist_offsets
            )
            specialist_bge_logits = specialist_bge_head(vectors)
        specialist_character_scores = (
            specialist_character_logits[:, 1] - specialist_character_logits[:, 0]
        )
        specialist_bge_scores = specialist_bge_logits[:, 1] - specialist_bge_logits[:, 0]

        masked_records = []
        for row, text in zip(dev, specialist_texts):
            for mention in sorted(row.get("object_mentions", ()), key=len, reverse=True):
                text = text.replace(str(mention), "[对象]")
            masked_records.append({"text": text})
        operation_vectors = _encode_records(
            tokenizer, encoder, masked_records, device, 64, pooling=pooling
        ).cpu()
        operation_vectors = torch.nn.functional.normalize(operation_vectors, dim=1)
        capability_vectors = torch.nn.functional.normalize(capability_vectors.cpu(), dim=1)
        similarities = operation_vectors @ capability_vectors.T
        global_capability_scores = similarities.max(1).values
        own_capability_scores = torch.full((len(dev),), -1.0)
        capability_index = {label: index for index, label in enumerate(capability_labels)}
        surfaces: dict[str, set[str]] = defaultdict(set)
        for row in train:
            label = str(row.get("label_id") or "")
            if row.get("expected_executable") is True and label in capability_index:
                for mention in row.get("object_mentions", ()):
                    surfaces[str(mention)].add(label)
        catalog_path = Path(__file__).resolve().parents[1] / "meta" / "detailed_action_entry_catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for entry in catalog["entries"]:
            label = str(entry.get("label") or "")
            if entry.get("entry_type") == "object" and label in capability_index:
                surfaces[str(entry["object_name_zh"])].add(label)
        for index, row in enumerate(dev):
            candidates = {
                label
                for mention in row.get("object_mentions", ())
                for label in surfaces.get(str(mention), ())
            }
            columns = [capability_index[label] for label in candidates]
            if columns:
                own_capability_scores[index] = similarities[index, columns].max()
        single_object_mask = torch.tensor([
            len(row.get("object_mentions", ())) == 1 for row in dev
        ])
        known_object_mask = own_capability_scores >= 0
        if "known_object_gate_threshold" in payload:
            gate_thresholds = torch.where(
                known_object_mask,
                torch.full_like(gate_scores, float(payload["known_object_gate_threshold"])),
                gate_thresholds,
            )
        gate_calibrated = apply_object_operation_specialist_gate(
            base_scores=gate_scores,
            base_threshold=gate_thresholds,
            character_scores=specialist_character_scores,
            bge_scores=specialist_bge_scores,
            own_capability_scores=own_capability_scores,
            global_capability_scores=global_capability_scores,
            single_object_mask=single_object_mask,
            known_object_mask=known_object_mask,
            config=specialist_config,
        ).long()
        gate_backend += "+object_operation_specialist"
        specialist_report = {
            "format": payload["object_operation_specialist"].get("format"),
            "ontology_sha256": payload["object_operation_specialist"].get("ontology_sha256"),
            "config": specialist_config,
        }
    short_object_config = payload.get("short_object_gate")
    if isinstance(short_object_config, dict):
        text_lengths = torch.tensor([len(_semantic_text(str(row["text"]))) for row in dev])
        gate_calibrated = apply_short_object_margin_gate(
            predictions=gate_calibrated,
            scores=gate_scores,
            threshold=gate_thresholds,
            text_lengths=text_lengths,
            known_object_mask=known_object_mask,
            config=short_object_config,
        ).long()
        gate_backend += "+short_object_margin"
    restrictive_attribute_mask = torch.tensor([
        _is_restrictive_attribute_command(str(row["text"])) for row in dev
    ])
    if restrictive_attribute_mask.any():
        gate_calibrated = gate_calibrated.bool() | restrictive_attribute_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+restrictive_attribute_grammar"
    unresolved_choice_mask = torch.tensor([
        _has_unresolved_choice(str(row["text"])) for row in dev
    ])
    if unresolved_choice_mask.any():
        gate_calibrated = gate_calibrated.bool() & ~unresolved_choice_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+unresolved_choice_grammar"
    cancellation_mask = torch.tensor([
        _has_cancelling_negation(str(row["text"])) for row in dev
    ])
    if cancellation_mask.any():
        gate_calibrated = gate_calibrated.bool() & ~cancellation_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+cancellation_grammar"
    information_mask = torch.tensor([
        _is_information_request(str(row["text"])) for row in dev
    ])
    if information_mask.any():
        gate_calibrated = gate_calibrated.bool() & ~information_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+information_grammar"
    completed_mask = torch.tensor([
        _is_completed_statement(str(row["text"])) for row in dev
    ])
    if completed_mask.any():
        gate_calibrated = gate_calibrated.bool() & ~completed_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+completed_statement_grammar"
    instance_ambiguity_mask = torch.tensor([
        _has_unresolved_instance_reference(str(row["text"])) for row in dev
    ])
    if instance_ambiguity_mask.any():
        gate_calibrated = gate_calibrated.bool() & ~instance_ambiguity_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+instance_ambiguity_grammar"
    attribute_mismatch_mask = torch.tensor([
        _has_object_attribute_mismatch(str(row["text"])) for row in dev
    ])
    if attribute_mismatch_mask.any():
        gate_calibrated = gate_calibrated.bool() & ~attribute_mismatch_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+object_attribute_compatibility_grammar"
    out_of_domain_mask = torch.tensor([
        _is_out_of_domain_request(str(row["text"])) for row in dev
    ])
    if out_of_domain_mask.any():
        gate_calibrated = gate_calibrated.bool() & ~out_of_domain_mask
        gate_calibrated = gate_calibrated.long()
        gate_backend += "+out_of_domain_grammar"
    valid = intent_targets.ne(-100)
    intent_pred = output["intent_logits"].argmax(1)
    intent_accuracy = float((intent_pred[valid] == intent_targets[valid]).float().mean())
    joint_accuracy = float(((gate_calibrated == gate_targets) & (~gate_targets.bool() | (intent_pred == intent_targets))).float().mean())
    strata: dict[str, list[bool]] = defaultdict(list)
    for index, row in enumerate(dev):
        if row.get("expected_executable") is True:
            strata[str(row["label_id"]).split(":", 1)[0]].append(bool(intent_pred[index] == intent_targets[index]))
    safety: dict[str, list[bool]] = defaultdict(list)
    for index, row in enumerate(dev):
        safety[str(row.get("safety_class") or "core_executable")].append(bool(gate_calibrated[index] == gate_targets[index]))
    gate_errors = [
        {
            "sample_id": str(row.get("sample_id", "")),
            "text": str(row["text"]),
            "safety_class": str(row.get("safety_class") or "core_executable"),
            "expected_executable": bool(row.get("expected_executable")),
            "predicted_executable": bool(gate_calibrated[index]),
            "gate_score": float(gate_scores[index]),
        }
        for index, row in enumerate(dev)
        if bool(gate_calibrated[index] != gate_targets[index])
    ]
    intent_errors = [
        {
            "sample_id": str(row.get("sample_id", "")),
            "text": str(row["text"]),
            "target": labels[int(intent_targets[index])],
            "predicted": labels[int(intent_pred[index])],
        }
        for index, row in enumerate(dev)
        if int(intent_targets[index]) >= 0 and bool(intent_pred[index] != intent_targets[index])
    ]
    return {
        "checkpoint": str(checkpoint),
        "pooling": pooling,
        "intent_rerank": rerank,
        "dev_records": len(dev),
        "intent_accuracy": intent_accuracy,
        "gate_argmax_accuracy": float((gate_argmax == gate_targets).float().mean()),
        "gate_backend": gate_backend,
        "gate_calibrated_accuracy": float((gate_calibrated == gate_targets).float().mean()),
        "gate_threshold": calibrated["threshold"],
        "known_object_gate_threshold": payload.get("known_object_gate_threshold"),
        "object_operation_specialist": specialist_report,
        "joint_accuracy": joint_accuracy,
        "intent_by_label_type": {
            key: {"accuracy": sum(values) / len(values), "records": len(values)}
            for key, values in sorted(strata.items())
        },
        "gate_by_safety_class": {
            key: {"accuracy": sum(values) / len(values), "records": len(values)}
            for key, values in sorted(safety.items())
        },
        "gate_errors": gate_errors,
        "intent_errors": intent_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_dev(args.checkpoint, args.dataset_dir, args.encoder_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
