import json
import random

# =========================
# 读取映射
# =========================
with open("meta/unified_object_map.json", "r", encoding="utf-8") as f:
    obj_map = json.load(f)

# =========================
# 输出数据
# =========================
dataset = []

SCENES = ["kitchen", "living_room", "bedroom"]

for obj, info in obj_map.items():

    task = info["task"]

    sample = {
        "task": task,
        "scene": random.choice(SCENES),
        "object": obj,
        "target": "table",
        "actions": [
            f"locate({obj})",
            f"grasp({obj})",
            f"move(table)",
            f"release({obj})"
        ]
    }

    dataset.append(sample)

# =========================
# 保存
# =========================
output_path = "datasets/final_train_dataset.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print("✔ FINAL DATASET CREATED:", len(dataset))