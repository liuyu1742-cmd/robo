import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EPIC_PATH = ROOT / "datasets" / "epic_kitchens" / "processed" / "household_segments.json"
EPIC_CLIP_INDEX_PATH = ROOT / "datasets" / "epic_kitchens" / "processed" / "project_tier_a_clip_index.json"
HICO_PATH = ROOT / "datasets" / "hico_20160224_det" / "processed" / "hico_household_subset.json"
HOI4D_PATH = ROOT / "datasets" / "hoi4d" / "processed" / "household_segments.json"
OUT = ROOT / "datasets" / "unified_actions"


def load(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def stable_validation(key, percent=10):
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return value < percent


def hico_actions(skill, obj):
    templates = {
        "grasp": [f"locate({obj})", f"grasp({obj})"],
        "carry": [f"locate({obj})", f"grasp({obj})", f"carry({obj})"],
        "open": [f"locate({obj})", f"grasp_handle({obj})", f"open({obj})"],
        "pour": [f"grasp({obj})", f"tilt({obj})", f"pour({obj})"],
        "wash": [f"locate({obj})", f"wash({obj})", f"rinse({obj})"],
        "inspect": [f"locate({obj})", f"inspect({obj})"],
    }
    return templates.get(skill, [f"locate({obj})", f"{skill}({obj})"])


def epic_records():
    clip_index = {}
    if EPIC_CLIP_INDEX_PATH.exists():
        clip_index = {item["segment_id"]: item for item in load(EPIC_CLIP_INDEX_PATH)}
    for item in load(EPIC_PATH):
        clip = clip_index.get(item["id"])
        if clip_index and clip is None:
            continue
        media = {
            "video_id": item["video_id"], "start_seconds": item["start_seconds"],
            "stop_seconds": item["stop_seconds"], "start_frame": item["start_frame"],
            "stop_frame": item["stop_frame"],
        }
        if clip is not None:
            media.update(
                {
                    "clip_path": clip["clip_path"],
                    "segment_start_in_clip": clip["segment_start_in_clip"],
                    "segment_stop_in_clip": clip["segment_stop_in_clip"],
                    "tier": clip["tier"],
                    "project_rank": clip["project_rank"],
                }
            )
        yield {
            "id": item["id"], "source": "epic_kitchens_100", "modality": "video_segment",
            "split": item["split"],
            "media": media,
            "instruction": item["narration"], "task": item["task"],
            "object": item["canonical_object"], "verb": item["verb"],
            "ordered_actions": item["ordered_actions"],
            "annotations": {
                "verb_class": item["verb_class"], "noun_class": item["noun_class"],
                "raw_verb": item["raw_verb"], "raw_noun": item["raw_noun"],
            },
        }


def hico_records():
    for sample in load(HICO_PATH):
        if sample["split"] == "test":
            split = "test"
        else:
            split = "validation" if stable_validation(sample["file_name"]) else "train"
        for interaction_index, interaction in enumerate(sample["interactions"]):
            obj, skill = interaction["canonical_object"], interaction["robot_skill"]
            yield {
                "id": f"hico_{sample['split']}_{sample['file_name']}_{interaction_index}",
                "source": "hico_det", "modality": "image_hoi", "split": split,
                "media": {
                    "image_path": f"datasets/hico_20160224_det/{sample['image_path']}",
                    "width": sample["width"], "height": sample["height"],
                },
                "instruction": f"{interaction['hico_verb']} {obj}",
                "task": interaction["requirement_task"], "object": obj,
                "verb": interaction["hico_verb"], "ordered_actions": hico_actions(skill, obj),
                "annotations": {
                    "hoi_id": interaction["hoi_id"], "robot_skill": skill,
                    "human_boxes": interaction["human_boxes"],
                    "object_boxes": interaction["object_boxes"],
                    "connections": interaction["connections"],
                },
            }


def hoi4d_records():
    for item in load(HOI4D_PATH):
        yield {
            "id": item["id"],
            "source": "hoi4d",
            "modality": "video_annotation",
            "split": item["split"],
            "media": {
                "annotation_path": item["annotation_path"],
                "file_path": item["file_path"],
                "duration_seconds": item["duration_seconds"],
                "start_seconds": item["start_seconds"],
                "stop_seconds": item["stop_seconds"],
            },
            "instruction": item["instruction"],
            "task": item["task"],
            "object": item["canonical_object"],
            "verb": item["verb"],
            "ordered_actions": item["ordered_actions"],
            "annotations": {
                "category_id": item["category_id"],
                "category_name_zh": item["category_name_zh"],
                "package_id": item["package_id"],
                "hand_id": item["hand_id"],
                "instance_id": item["instance_id"],
                "subject_id": item["subject_id"],
                "layout_id": item["layout_id"],
                "take_id": item["take_id"],
                "raw_events": item["raw_events"],
            },
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Build unified ordered-action JSONL datasets.")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory.")
    parser.add_argument("--include-hoi4d", action="store_true", help="Include processed HOI4D household segments.")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (out / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    counts, sources, tasks = Counter(), Counter(), Counter()
    identifiers = set()
    record_iterators = [epic_records(), hico_records()]
    if args.include_hoi4d:
        record_iterators.append(hoi4d_records())
    try:
        for records in record_iterators:
            for record in records:
                if record["id"] in identifiers:
                    raise ValueError(f"Duplicate id: {record['id']}")
                identifiers.add(record["id"])
                if not record["ordered_actions"]:
                    raise ValueError(f"Empty action sequence: {record['id']}")
                handles[record["split"]].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[record["split"]] += 1
                sources[record["source"]] += 1
                tasks[record["task"]] += 1
    finally:
        for handle in handles.values():
            handle.close()

    schema = {
        "format": "robot_household_ordered_actions_v1",
        "required_fields": [
            "id", "source", "modality", "split", "media", "instruction",
            "task", "object", "verb", "ordered_actions", "annotations",
        ],
        "split_policy": {
            "epic": "official train/validation",
            "hico": "official test retained; official train split by image SHA1 into 90% train/10% validation",
            "hoi4d": "stable SHA1 split by segment id into 80% train/10% validation/10% test when --include-hoi4d is used",
        },
    }
    summary = {
        "total_records": sum(counts.values()), "splits": dict(counts),
        "sources": dict(sources), "tasks": dict(tasks), "unique_ids": len(identifiers),
    }
    with (out / "schema.json").open("w", encoding="utf-8") as file:
        json.dump(schema, file, ensure_ascii=False, indent=2)
    with (out / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
