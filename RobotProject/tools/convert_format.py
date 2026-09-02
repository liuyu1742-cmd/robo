import json

TASK_MAP = {
    "cup": ("drink_task", ["locate", "grasp", "place"]),
    "bottle": ("drink_task", ["locate", "grasp", "place"]),
    "plate": ("cleaning_task", ["locate", "grasp", "wipe"]),
    "chair": ("organizing_task", ["locate", "push"]),
    "remote": ("appliance_task", ["locate", "press"])
}

INPUT_FILE = "datasets/coco/coco_filtered.json"
OUTPUT_FILE = "datasets/coco/coco_action_dataset.json"

def convert():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []

    for ann in data["annotations"]:
        obj_id = ann["category_id"]

        # COCO里通常需要name映射，这里简化
        for obj, (task, actions) in TASK_MAP.items():
            sample = {
                "task": task,
                "object": obj,
                "actions": [f"{a}({obj})" for a in actions]
            }
            result.append(sample)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("✔ Action dataset created:", len(result))

if __name__ == "__main__":
    convert()