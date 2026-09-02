"""Search train/dev-only high-confidence rules that rescue scene-to-object errors."""

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

from tools.detailed_action_catalog import load_detailed_action_catalog
from tools.detailed_action_bge_model import (
    DetailedActionBgeClassifier,
    apply_object_surface_override,
    apply_scene_rescue,
    combine_scene_object_logits,
    load_local_bge_encoder,
    rerank_topk_scores,
)
from tools.evaluate_detailed_action_bge_dev import _object_surface_candidate_mask
from tools.detailed_action_runtime import _cosine, _text_features
from train_detailed_action_bge import _encode_records, build_label_order, build_targets, load_training_splits


def diagnose(checkpoint: Path, dataset_dir: Path, encoder_dir: Path) -> dict:
    train, dev = load_training_splits(dataset_dir)
    executable = [row for row in dev if row.get("expected_executable") is True]
    labels = build_label_order(train)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device)
    if payload.get("encoder_state"):
        encoder.load_state_dict(payload["encoder_state"])
    encoder.eval()
    vectors = _encode_records(
        tokenizer, encoder, executable, device, 128, pooling=str(payload.get("pooling", "mean"))
    ).cpu()
    model = DetailedActionBgeClassifier(vectors.shape[1], len(labels))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        primary_raw = model(vectors)["intent_logits"]

    label_texts: dict[str, list[str]] = defaultdict(list)
    for row in train:
        if row.get("expected_executable") is True:
            label_texts[str(row["label_id"])].append(
                "；".join([str(row["text"]), *row.get("object_mentions", ()), *row.get("operation_evidence", ())])
            )
    features = {label: _text_features("".join(label_texts[label])) for label in labels}
    lexical = torch.tensor([
        [_cosine(_text_features(str(row["text"])), features[label]) for label in labels]
        for row in executable
    ])
    rerank = payload["intent_rerank"]
    primary = rerank_topk_scores(
        primary_raw, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"])
    )
    specialist = payload["scene_specialist"]
    state = specialist["intent_head_state"]
    scene_raw = torch.nn.functional.linear(vectors, state["weight"], state["bias"])
    scene = rerank_topk_scores(
        scene_raw, lexical, top_k=int(rerank["top_k"]), weight=float(rerank["weight"])
    )
    scene_mask = torch.tensor([label.startswith("scene:") for label in labels])
    combined = combine_scene_object_logits(
        primary, scene, scene_mask, scene_bias=float(specialist["scene_bias"])
    )
    final = combined
    surface_candidate_mask = _object_surface_candidate_mask(train, executable, labels)
    if isinstance(payload.get("object_surface_override"), dict):
        surface_config = payload["object_surface_override"]
        final = apply_object_surface_override(
            final,
            primary,
            scene_mask,
            surface_candidate_mask,
            scene_guard_logits=(scene if "scene_guard_margin" in surface_config else None),
            scene_guard_margin=surface_config.get("scene_guard_margin"),
        )
    configured_rescue = payload.get("scene_rescue")
    if isinstance(configured_rescue, dict) and configured_rescue.get("method") in (
        "specialist_gap",
        "lexical_gap",
    ):
        final = apply_scene_rescue(
            final,
            primary,
            scene,
            scene_mask,
            threshold=float(configured_rescue["threshold"]),
            method=str(configured_rescue["method"]),
            auxiliary_scores=lexical,
        )
    targets, _ = build_targets(executable, labels)
    target_scene = torch.tensor([str(row["label_id"]).startswith("scene:") for row in executable])
    baseline_prediction = final.argmax(1)
    baseline_object_rows = ~scene_mask[baseline_prediction]
    primary_scene_values, primary_scene_local = primary[:, scene_mask].max(1)
    primary_object_values = primary[:, ~scene_mask].max(1).values
    scene_values, scene_local = scene[:, scene_mask].max(1)
    lexical_scene_values = lexical[:, scene_mask].max(1).values
    lexical_object_values = lexical[:, ~scene_mask].max(1).values
    scene_columns = torch.where(scene_mask)[0]
    proposed = scene_columns[scene_local]
    scene_guard_values = scene_values - primary_object_values
    ordered_guard_values = sorted(float(value) for value in torch.unique(scene_guard_values).tolist())
    guard_thresholds = [
        ordered_guard_values[0] - 1e-6,
        *[(left + right) / 2 for left, right in zip(ordered_guard_values, ordered_guard_values[1:])],
        ordered_guard_values[-1] + 1e-6,
    ] if ordered_guard_values else []
    scene_guard_trials = []
    for guard_margin in guard_thresholds:
        guarded = apply_object_surface_override(
            combined,
            primary,
            scene_mask,
            surface_candidate_mask,
            scene_guard_logits=scene,
            scene_guard_margin=guard_margin,
        )
        if isinstance(configured_rescue, dict) and configured_rescue.get("method") in (
            "specialist_gap",
            "lexical_gap",
        ):
            guarded = apply_scene_rescue(
                guarded,
                primary,
                scene,
                scene_mask,
                threshold=float(configured_rescue["threshold"]),
                method=str(configured_rescue["method"]),
                auxiliary_scores=lexical,
            )
        guard_prediction = guarded.argmax(1)
        guard_correct = guard_prediction.eq(targets)
        scene_guard_trials.append({
            "scene_guard_margin": guard_margin,
            "scene": float(guard_correct[target_scene].float().mean()),
            "object": float(guard_correct[~target_scene].float().mean()),
            "overall": float(guard_correct.float().mean()),
        })
    scene_errors = []
    for index in torch.where(target_scene & baseline_prediction.ne(targets))[0].tolist():
        predicted_index = int(baseline_prediction[index])
        target_index = int(targets[index])
        scene_errors.append({
            "sample_id": str(executable[index].get("sample_id", "")),
            "text": str(executable[index]["text"]),
            "target": labels[target_index],
            "predicted": labels[predicted_index],
            "predicted_type": "scene" if labels[predicted_index].startswith("scene:") else "object",
            "forced_scene": labels[int(proposed[index])],
            "forced_scene_correct": int(proposed[index]) == target_index,
            "scene_guard_value": float(scene_guard_values[index]),
            "margin": float(final[index, predicted_index] - final[index, target_index]),
        })
    agreement = primary_scene_local.eq(scene_local)
    scores = {
        "primary_gap": primary_scene_values - primary_object_values,
        "specialist_gap": scene_values - primary_object_values,
        "lexical_gap": lexical_scene_values - lexical_object_values,
        "primary_plus_lexical": primary_scene_values - primary_object_values + 4.0 * (lexical_scene_values - lexical_object_values),
    }
    trials = []
    for feature_name, values in scores.items():
        for require_agreement in (False, True):
            eligible = baseline_object_rows & (agreement if require_agreement else torch.ones_like(agreement))
            ordered = sorted(float(value) for value in torch.unique(values[eligible]).tolist())
            thresholds = [ordered[0] - 1e-6, *[(a + b) / 2 for a, b in zip(ordered, ordered[1:])], ordered[-1] + 1e-6] if ordered else []
            for threshold in thresholds:
                rescue = eligible & (values >= threshold)
                prediction = baseline_prediction.clone()
                prediction[rescue] = proposed[rescue]
                correct = prediction.eq(targets)
                trials.append({
                    "feature": feature_name,
                    "require_agreement": require_agreement,
                    "threshold": threshold,
                    "rescued_rows": int(rescue.sum()),
                    "scene": float(correct[target_scene].float().mean()),
                    "object": float(correct[~target_scene].float().mean()),
                    "overall": float(correct.float().mean()),
                })
    minimum_object = 0.9149606299212598
    safe = [row for row in trials if row["object"] + 1e-7 >= minimum_object]
    best = max(
        safe,
        key=lambda row: (row["scene"], row["overall"], row["object"]),
        default=None,
    )
    baseline_correct = baseline_prediction.eq(targets)
    surface_labels: dict[str, set[int]] = defaultdict(set)
    for row in train:
        label = str(row.get("label_id") or "")
        if row.get("expected_executable") is True and label.startswith("object:"):
            for surface in row.get("object_mentions", ()):
                if len(str(surface).strip()) >= 2:
                    surface_labels[str(surface).strip()].add(labels.index(label))
    for entry in load_detailed_action_catalog()["entries"]:
        if entry.get("entry_type") == "object":
            surface_labels[str(entry["object_name_zh"])].add(labels.index(str(entry["label"])))
    candidate_columns: list[list[int]] = []
    for row in executable:
        text = str(row["text"])
        matches = [surface for surface in surface_labels if surface in text]
        maximal = [
            surface for surface in matches
            if not any(surface != other and surface in other for other in matches)
        ]
        candidate_columns.append(sorted({column for surface in maximal for column in surface_labels[surface]}))
    surface_trials = []
    margin_trials = []
    primary_object_gap = (
        primary[:, ~scene_mask].max(1).values - primary[:, scene_mask].max(1).values
    )
    rescue_config = payload.get("scene_rescue")
    lexical_gap = lexical[:, scene_mask].max(1).values - lexical[:, ~scene_mask].max(1).values
    for bias in (-1.0, -0.5, 0.0, 0.5, 1.0):
        biased = combine_scene_object_logits(primary, scene, scene_mask, scene_bias=bias)
        for rule in ("none", "primary_object_only", "final_object_only", "always"):
            prediction = biased.argmax(1)
            primary_prediction = primary.argmax(1)
            replaced = 0
            for index, columns in enumerate(candidate_columns):
                if not columns or rule == "none":
                    continue
                eligible = (
                    rule == "always"
                    or (rule == "primary_object_only" and not bool(scene_mask[primary_prediction[index]]))
                    or (rule == "final_object_only" and not bool(scene_mask[prediction[index]]))
                )
                if eligible:
                    prediction[index] = columns[int(primary[index, columns].argmax())]
                    replaced += 1
            correct = prediction.eq(targets)
            surface_trials.append({
                "scene_bias": bias,
                "rule": rule,
                "replaced": replaced,
                "scene": float(correct[target_scene].float().mean()),
                "object": float(correct[~target_scene].float().mean()),
                "overall": float(correct.float().mean()),
            })
        for margin in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0):
            prediction = biased.argmax(1)
            replaced = 0
            for index, columns in enumerate(candidate_columns):
                if columns and float(primary_object_gap[index]) >= margin:
                    prediction[index] = columns[int(primary[index, columns].argmax())]
                    replaced += 1
            rescued = 0
            if isinstance(rescue_config, dict) and rescue_config.get("method") == "lexical_gap":
                threshold = float(rescue_config["threshold"])
                for index in range(len(prediction)):
                    if not bool(scene_mask[prediction[index]]) and float(lexical_gap[index]) >= threshold:
                        prediction[index] = scene_columns[scene_local[index]]
                        rescued += 1
            correct = prediction.eq(targets)
            margin_trials.append({
                "scene_bias": bias,
                "object_margin": margin,
                "replaced": replaced,
                "rescued": rescued,
                "scene": float(correct[target_scene].float().mean()),
                "object": float(correct[~target_scene].float().mean()),
                "overall": float(correct.float().mean()),
            })
    return {
        "baseline": {
            "scene": float(baseline_correct[target_scene].float().mean()),
            "object": float(baseline_correct[~target_scene].float().mean()),
            "overall": float(baseline_correct.float().mean()),
        },
        "scene_error_count": len(scene_errors),
        "scene_errors": scene_errors,
        "best_scene_guard": max(
            scene_guard_trials,
            key=lambda row: (min(row["scene"], row["object"]), row["overall"]),
            default=None,
        ),
        "top_scene_guards": sorted(
            scene_guard_trials,
            key=lambda row: (min(row["scene"], row["object"]), row["overall"]),
            reverse=True,
        )[:20],
        "best_safe": best,
        "top_safe": sorted(safe, key=lambda row: (row["scene"], row["overall"], row["object"]), reverse=True)[:20],
        "best_surface_constraint": max(
            surface_trials,
            key=lambda row: (min(row["scene"], row["object"]), row["overall"]),
        ),
        "surface_constraint_trials": surface_trials,
        "best_safe_surface_margin": max(
            (row for row in margin_trials if row["object"] + 1e-7 >= 0.9149606299212598),
            key=lambda row: (row["scene"], row["overall"], row["object"]),
            default=None,
        ),
        "top_safe_surface_margins": sorted(
            (row for row in margin_trials if row["object"] + 1e-7 >= 0.9149606299212598),
            key=lambda row: (row["scene"], row["overall"], row["object"]),
            reverse=True,
        )[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("models/pretrained/BAAI_bge-small-zh-v1.5"))
    parser.add_argument("--output", type=Path, default=Path("outputs/detailed_action_bge/scene_rescue_diagnostics.json"))
    args = parser.parse_args()
    result = diagnose(args.checkpoint, args.dataset_dir, args.encoder_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
