"""Generate train-only scene paraphrases with the pinned local Qwen teacher."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.detailed_action_catalog import load_detailed_action_catalog
from train_detailed_action_bge import load_training_splits


def _parse_array(text: str) -> list[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(value).strip() for value in values if isinstance(value, str) and str(value).strip()]


def generate(
    model_dir: Path,
    dataset_dir: Path,
    output: Path,
    *,
    per_label: int,
    seed: int,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    train, dev = load_training_splits(dataset_dir)
    forbidden = {str(row["text"]).strip() for row in [*train, *dev]}
    examples: dict[str, list[str]] = defaultdict(list)
    for row in train:
        label = str(row.get("label_id") or "")
        if row.get("expected_executable") is True and label.startswith("scene:"):
            examples[label].append(str(row["text"]))
    catalog = load_detailed_action_catalog()
    scenes = [entry for entry in catalog["entries"] if entry.get("entry_type") == "scene"]
    existing: dict[str, list[str]] = {}
    if output.exists():
        for row in json.loads(output.read_text(encoding="utf-8")):
            existing.setdefault(str(row["label_id"]), []).append(str(row["text"]))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    rows = [
        {
            "sample_id": f"qwen-scene-router-{label.split(':', 1)[1]}-{index + 1:02d}",
            "text": text,
            "label_id": label,
            "expected_executable": True,
            "annotation_source": "qwen_teacher_train_only_scene_router_v1",
        }
        for label, texts in existing.items()
        for index, text in enumerate(texts)
    ]
    completed = {row["label_id"] for row in rows if sum(item["label_id"] == row["label_id"] for item in rows) >= per_label}
    for scene_index, entry in enumerate(scenes, 1):
        label = str(entry["label"])
        if label in completed:
            continue
        existing_count = len(existing.get(label, ()))
        remaining = per_label - existing_count
        object_names = [str(item["object_name_zh"]) for item in entry.get("objects", ())]
        prompt = (
            f"你在扩充中文机器人指令训练集。场景命令是“{entry['command']}”，涉及对象："
            f"{'、'.join(object_names)}。参考口语：{json.dumps(examples[label][:3], ensure_ascii=False)}。"
            f"请写{remaining}句新的自然口语命令，必须表达完成整个场景/整体目的，需要协调多个操作；"
            "不要只命令一个物体，不要解释，不要编号，不要复制参考句。只输出JSON字符串数组。"
        )
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                encoded,
                max_new_tokens=384,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
            )
        values = _parse_array(tokenizer.decode(generated[0, encoded.shape[1] :], skip_special_tokens=True))
        unique = []
        seen = set(forbidden) | set(existing.get(label, ()))
        for value in values:
            if value not in seen:
                seen.add(value)
                unique.append(value)
            if len(unique) == remaining:
                break
        for index, text in enumerate(unique, existing_count + 1):
            rows.append({
                "sample_id": f"qwen-scene-router-{label.split(':', 1)[1]}-{index:02d}",
                "text": text,
                "label_id": label,
                "expected_executable": True,
                "annotation_source": "qwen_teacher_train_only_scene_router_v1",
            })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "scene_labels_processed": scene_index,
            "scene_labels_total": len(scenes),
            "label": label,
            "generated": len(unique),
            "records": len(rows),
        }, ensure_ascii=False), flush=True)
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["label_id"])] += 1
    return {
        "output": str(output),
        "records": len(rows),
        "labels": len(counts),
        "complete_labels": sum(count >= per_label for count in counts.values()),
        "minimum_per_label": min(counts.values()) if counts else 0,
        "target_per_label": per_label,
        "independent_test_policy": "not opened or used",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/pretrained/Qwen2.5-7B-Instruct"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--output", type=Path, default=Path("datasets/detailed_action_spoken_v2/train_scene_router_qwen_v1.json"))
    parser.add_argument("--per-label", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    print(json.dumps(generate(args.model_dir, args.dataset_dir, args.output, per_label=args.per_label, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
