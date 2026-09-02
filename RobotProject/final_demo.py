import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tools.project_encode import (
    ACTION_VOCAB,
    ARG_NONE_ID,
    ARG_PAD_ID,
    ARG_VOCAB,
    EOS_ID,
    ID_TO_ACTION,
    ID_TO_ARG,
    OBJECT_VOCAB,
    PAD_ID,
    TARGET_VOCAB,
    TASK_VOCAB,
    encode_object,
    encode_target,
    encode_task,
)
from train_project_transformer_v2 import (
    ProjectConditionedTransformer,
    SOURCE_VOCAB,
    VERB_VOCAB,
)
from train_epic_visual_conditions import VisualConditionClassifier


ROOT = Path(__file__).resolve().parent


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ProjectConditionedTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def encode_condition(source, verb, task, obj, target, device):
    return {
        "source": torch.tensor([SOURCE_VOCAB[source]], dtype=torch.long, device=device),
        "verb": torch.tensor([VERB_VOCAB[verb]], dtype=torch.long, device=device),
        "task": torch.tensor([encode_task(task)], dtype=torch.long, device=device),
        "object": torch.tensor([encode_object(obj)], dtype=torch.long, device=device),
        "target": torch.tensor([encode_target(target)], dtype=torch.long, device=device),
    }


def preferred_argument(action_id, obj, target):
    action = ID_TO_ACTION[action_id]
    if action in {"move", "insert"} and target != "none":
        return ARG_VOCAB.get(target, ARG_NONE_ID)
    return ARG_VOCAB.get(obj, ARG_NONE_ID)


@torch.no_grad()
def greedy_generate(
    model, source, verb, task, obj, target, device, max_actions=12,
    visual=None, visual_present=None,
):
    cond = encode_condition(source, verb, task, obj, target, device)
    action_ids = [ACTION_VOCAB["<bos>"]]
    arg_ids = [ARG_NONE_ID]
    for _ in range(max_actions):
        action_tensor = torch.tensor([action_ids], dtype=torch.long, device=device)
        arg_tensor = torch.tensor([arg_ids], dtype=torch.long, device=device)
        action_logits, argument_logits = model(
            cond["source"], cond["verb"], cond["task"], cond["object"],
            cond["target"], action_tensor, arg_tensor,
            visual, visual_present,
        )
        logits = action_logits[0, -1].clone()
        logits[PAD_ID] = -1e9
        logits[ACTION_VOCAB["<bos>"]] = -1e9
        next_action = int(logits.argmax().item())
        if next_action == EOS_ID:
            break
        # Use the model's argument if confident; fall back to schema grounding.
        arg_logits = argument_logits[0, -1].clone()
        arg_logits[ARG_PAD_ID] = -1e9
        next_arg = int(arg_logits.argmax().item())
        if next_arg in (ARG_NONE_ID, ARG_PAD_ID):
            next_arg = preferred_argument(next_action, obj, target)
        action_ids.append(next_action)
        arg_ids.append(next_arg)
    return [
        f"{ID_TO_ACTION[action_id]}({ID_TO_ARG[arg_id]})"
        for action_id, arg_id in zip(action_ids[1:], arg_ids[1:])
    ]


def load_sample(dataset_path, sample_id=None, seed=42):
    with dataset_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if sample_id:
        for item in data:
            if item["id"] == sample_id:
                return item
        raise ValueError(f"sample id not found: {sample_id}")
    rng = random.Random(seed)
    return rng.choice(data)


def load_precomputed_visual(path, sample_id, expected_dim, device):
    if expected_dim <= 0 or path is None or not path.is_file():
        return None, torch.tensor([False], dtype=torch.bool, device=device), None
    store = np.load(path, allow_pickle=False)
    ids = store["segment_ids"].tolist()
    try:
        index = ids.index(sample_id)
    except ValueError:
        return None, torch.tensor([False], dtype=torch.bool, device=device), None
    value = store["features"][index].astype(np.float32, copy=False)
    if value.shape != (expected_dim,):
        raise ValueError(f"visual feature dimension mismatch: {value.shape} != ({expected_dim},)")
    return (
        torch.from_numpy(value.copy()).unsqueeze(0).to(device),
        torch.tensor([True], dtype=torch.bool, device=device),
        "precomputed_segment_feature",
    )


def extract_clip_visual(path, start, stop, weights, expected_dim, device):
    if expected_dim != 512:
        raise ValueError(f"online extractor produces 512 features, model expects {expected_dim}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {path}")
    if stop is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1.0
        stop = frames / fps
    duration = max(float(stop) - float(start), 0.04)
    images = []
    try:
        for fraction in (0.2, 0.5, 0.8):
            cap.set(cv2.CAP_PROP_POS_MSEC, (float(start) + duration * fraction) * 1000.0)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"cannot decode {path} at fraction {fraction}")
            images.append(image)
    finally:
        cap.release()
    detector = YOLO(str(weights))
    embeddings = detector.predict(images, embed=[21], imgsz=640, device=str(device), verbose=False)
    values = torch.stack([item.detach().float().cpu() for item in embeddings]).numpy()
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    feature = np.concatenate([values.mean(axis=0), values.std(axis=0)]).astype(np.float32)
    return (
        torch.from_numpy(feature).unsqueeze(0).to(device),
        torch.tensor([True], dtype=torch.bool, device=device),
        "online_clip_feature",
    )


def infer_visual_conditions(checkpoint_path, visual, device):
    if visual is None:
        raise ValueError("visual condition inference requires a clip feature")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VisualConditionClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        verb_logits, task_logits, object_logits = model(visual)
    id_to_verb = {value: key for key, value in checkpoint["verb_vocab"].items()}
    id_to_task = {value: key for key, value in checkpoint["task_vocab"].items()}
    id_to_object = {value: key for key, value in checkpoint["object_vocab"].items()}

    def prediction(logits, labels):
        probabilities = logits.softmax(dim=1)
        confidence, index = probabilities.max(dim=1)
        return labels[int(index.item())], float(confidence.item())

    verb, verb_conf = prediction(verb_logits, id_to_verb)
    task, task_conf = prediction(task_logits, id_to_task)
    obj, object_conf = prediction(object_logits, id_to_object)
    return {
        "verb": verb,
        "task": task,
        "object": obj,
        "confidence": {"verb": verb_conf, "task": task_conf, "object": object_conf},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Final project action-sequence demo.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "action_transformer_project_visual.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "project_test_dataset_normalized.json")
    parser.add_argument("--sample-id", help="Use one normalized dataset sample by id.")
    parser.add_argument("--seed", type=int, default=42, help="Random sample seed when --sample-id is omitted.")
    parser.add_argument("--source", choices=sorted(SOURCE_VOCAB), help="Manual source, e.g. hoi4d or hico_det.")
    parser.add_argument("--verb", choices=sorted(VERB_VOCAB), help="Manual verb/intention.")
    parser.add_argument("--task", choices=sorted(TASK_VOCAB), help="Manual task.")
    parser.add_argument("--object", choices=sorted(OBJECT_VOCAB), help="Manual object.")
    parser.add_argument("--target", choices=sorted(TARGET_VOCAB), default="none", help="Manual target.")
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--visual-features", type=Path,
        default=ROOT / "datasets" / "epic_kitchens" / "processed" / "tier_a_visual_features.npz",
    )
    parser.add_argument("--clip", type=Path, help="Optional new video clip for online visual extraction.")
    parser.add_argument("--clip-start", type=float, default=0.0)
    parser.add_argument("--clip-stop", type=float, default=None)
    parser.add_argument(
        "--vision-weights", type=Path,
        default=ROOT / "models" / "coco_household" / "yolov8n_16cls_continue_e10" / "weights" / "best.pt",
    )
    parser.add_argument("--infer-conditions", action="store_true", help="Infer verb/task/object from the visual feature.")
    parser.add_argument(
        "--condition-checkpoint", type=Path,
        default=ROOT / "models" / "epic_visual_condition_classifier.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)

    manual = all([args.source, args.verb, args.task, args.object])
    if manual:
        sample = {
            "id": "manual_input",
            "source": args.source,
            "verb": args.verb,
            "task": args.task,
            "object": args.object,
            "target": args.target,
            "instruction": f"{args.verb} {args.object}",
            "actions": None,
        }
    else:
        sample = load_sample(args.dataset, args.sample_id, args.seed)

    visual_dim = int(checkpoint["model_config"].get("visual_dim", 0))
    if args.clip is not None:
        visual, visual_present, visual_source = extract_clip_visual(
            args.clip, args.clip_start, args.clip_stop, args.vision_weights, visual_dim, device
        )
    else:
        visual, visual_present, visual_source = load_precomputed_visual(
            args.visual_features, sample["id"], visual_dim, device
        )
    inferred_conditions = None
    if args.infer_conditions:
        if not bool(visual_present.item()):
            raise ValueError("--infer-conditions requires --clip or a sample with precomputed visual features")
        inferred_conditions = infer_visual_conditions(args.condition_checkpoint, visual, device)
        sample = {
            "id": sample["id"] if args.clip is None else "visual_clip_input",
            "source": "epic_kitchens_100",
            "verb": inferred_conditions["verb"],
            "task": inferred_conditions["task"],
            "object": inferred_conditions["object"],
            "target": "none",
            "instruction": f"{inferred_conditions['verb']} {inferred_conditions['object']}",
            "actions": None,
        }
    predicted = greedy_generate(
        model,
        sample["source"],
        sample["verb"],
        sample["task"],
        sample["object"],
        sample["target"],
        device,
        args.max_actions,
        visual,
        visual_present,
    )
    output = {
        "model": str(args.checkpoint),
        "sample_id": sample["id"],
        "instruction": sample.get("instruction"),
        "condition": {
            "source": sample["source"],
            "verb": sample["verb"],
            "task": sample["task"],
            "object": sample["object"],
            "target": sample["target"],
        },
        "predicted_actions": predicted,
        "visual": {
            "present": bool(visual_present.item()),
            "source": visual_source,
            "feature_dim": visual_dim,
        },
    }
    if inferred_conditions is not None:
        output["inferred_conditions"] = inferred_conditions
    if sample.get("actions") is not None:
        output["expected_actions"] = sample["actions"]
        output["exact_match"] = predicted == sample["actions"]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
