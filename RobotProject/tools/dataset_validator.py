import os
import json

FILES = [
    "datasets/coco/train2017",
    "datasets/coco/val2017",
    "datasets/coco/annotations",
    "datasets/coco/coco_filtered.json"
]

def check():
    print("\n=== DATASET VALIDATION ===\n")

    for f in FILES:
        if os.path.exists(f):
            print("✔ OK:", f)
        else:
            print("❌ MISSING:", f)

    # 检查JSON合法性
    json_files = [
        "datasets/coco/coco_filtered.json"
    ]

    for jf in json_files:
        if os.path.exists(jf):
            try:
                with open(jf, "r") as f:
                    json.load(f)
                print("✔ JSON valid:", jf)
            except:
                print("❌ JSON broken:", jf)

if __name__ == "__main__":
    check()