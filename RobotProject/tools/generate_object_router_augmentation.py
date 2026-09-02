"""Generate symmetric train-only object paraphrases with the local Qwen teacher."""

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


def _parse_mapping(text: str) -> dict[str, list[str]]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(label): [str(item).strip() for item in items if isinstance(item, str) and str(item).strip()]
        for label, items in value.items()
        if isinstance(items, list)
    }


def generate(model_dir: Path, dataset_dir: Path, output: Path, *, per_label: int, batch_labels: int, seed: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    train, dev = load_training_splits(dataset_dir)
    forbidden = {str(row["text"]).strip() for row in [*train, *dev]}
    examples: dict[str, list[str]] = defaultdict(list)
    for row in train:
        label = str(row.get("label_id") or "")
        if row.get("expected_executable") is True and label.startswith("object:"):
            examples[label].append(str(row["text"]))
    objects = [entry for entry in load_detailed_action_catalog()["entries"] if entry.get("entry_type") == "object"]
    rows = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    counts = defaultdict(int)
    existing_texts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        counts[str(row["label_id"])] += 1
        existing_texts[str(row["label_id"])].add(str(row["text"]))
    pending = [entry for entry in objects if counts[str(entry["label"])] < per_label]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    for start in range(0, len(pending), batch_labels):
        group = pending[start : start + batch_labels]
        specifications = []
        for entry in group:
            label = str(entry["label"])
            remaining = per_label - counts[label]
            specifications.append({
                "label": label,
                "object": entry["object_name_zh"],
                "action": entry["detailed_actions"][0],
                "reference": examples[label][:1],
                "count": remaining,
            })
        prompt = (
            "你在扩充中文机器人单物体操作指令训练集。根据下面JSON规格，为每个label写指定数量的自然口语命令。"
            "每句只针对该物体及其合理操作，不要写全屋场景或多设备联动，不要复制参考句。"
            "只输出JSON对象，键必须是原label，值是字符串数组："
            + json.dumps(specifications, ensure_ascii=False)
        )
        encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(encoded, max_new_tokens=768, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.05)
        mapping = _parse_mapping(tokenizer.decode(generated[0, encoded.shape[1] :], skip_special_tokens=True))
        added = 0
        for entry in group:
            label = str(entry["label"])
            needed = per_label - counts[label]
            for text in mapping.get(label, ()):
                if text in forbidden or text in existing_texts[label]:
                    continue
                index = counts[label] + 1
                rows.append({
                    "sample_id": f"qwen-object-router-{label.replace(':', '-')}-{index:02d}",
                    "text": text,
                    "label_id": label,
                    "expected_executable": True,
                    "annotation_source": "qwen_teacher_train_only_object_router_v1",
                })
                counts[label] += 1
                existing_texts[label].add(text)
                added += 1
                if counts[label] >= per_label:
                    break
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"batch": start // batch_labels + 1, "batches": (len(pending) + batch_labels - 1) // batch_labels, "labels": len(group), "added": added, "records": len(rows)}, ensure_ascii=False), flush=True)
    return {
        "output": str(output),
        "records": len(rows),
        "labels": sum(count > 0 for count in counts.values()),
        "complete_labels": sum(count >= per_label for count in counts.values()),
        "target_labels": len(objects),
        "target_per_label": per_label,
        "independent_test_policy": "not opened or used",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/pretrained/Qwen2.5-7B-Instruct"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/detailed_action_spoken_v2"))
    parser.add_argument("--output", type=Path, default=Path("datasets/detailed_action_spoken_v2/train_object_router_qwen_v1.json"))
    parser.add_argument("--per-label", type=int, default=2)
    parser.add_argument("--batch-labels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    print(json.dumps(generate(args.model_dir, args.dataset_dir, args.output, per_label=args.per_label, batch_labels=args.batch_labels, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
