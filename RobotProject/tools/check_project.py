import os

paths = [
    "datasets/coco",
    "datasets/ycb_lite",
    "meta",
    "tools"
]

print("=== PROJECT CHECK ===")

for p in paths:
    print(p, "✔" if os.path.exists(p) else "❌")