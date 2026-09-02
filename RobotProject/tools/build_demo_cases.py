import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_demo import greedy_generate, load_model


OUT_JSON = ROOT / "datasets" / "demo_cases.json"
OUT_MD = ROOT / "datasets" / "demo_cases_report.md"


CASES = [
    {
        "id": "hoi4d_trash_bin",
        "title": "垃圾桶开合",
        "source": "hoi4d",
        "verb": "operate",
        "task": "waste_disposal",
        "object": "trash_bin",
        "target": "none",
        "instruction": "操作垃圾桶，完成开合动作",
    },
    {
        "id": "hoi4d_kettle",
        "title": "水壶取放",
        "source": "hoi4d",
        "verb": "operate",
        "task": "smart_cooking",
        "object": "kettle",
        "target": "none",
        "instruction": "取放水壶",
    },
    {
        "id": "hoi4d_knife",
        "title": "刀具切割",
        "source": "hoi4d",
        "verb": "operate",
        "task": "smart_cooking",
        "object": "knife",
        "target": "none",
        "instruction": "使用刀具执行切割",
    },
    {
        "id": "hoi4d_reading_lamp",
        "title": "台灯控制",
        "source": "hoi4d",
        "verb": "operate",
        "task": "appliance_management",
        "object": "reading_lamp",
        "target": "none",
        "instruction": "控制台灯",
    },
    {
        "id": "hoi4d_chair",
        "title": "椅子整理",
        "source": "hoi4d",
        "verb": "operate",
        "task": "organizing",
        "object": "chair",
        "target": "none",
        "instruction": "移动并整理椅子",
    },
    {
        "id": "hico_book",
        "title": "书本整理/阅读",
        "source": "hico_det",
        "verb": "read",
        "task": "object_fetching",
        "object": "book",
        "target": "none",
        "instruction": "定位并阅读书本",
    },
    {
        "id": "hico_remote",
        "title": "遥控器操作",
        "source": "hico_det",
        "verb": "operate",
        "task": "appliance_management",
        "object": "remote_control",
        "target": "none",
        "instruction": "操作遥控器",
    },
    {
        "id": "hico_cup_wash",
        "title": "杯子清洗",
        "source": "hico_det",
        "verb": "wash",
        "task": "dish_washing",
        "object": "cup",
        "target": "none",
        "instruction": "清洗杯子",
    },
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ROOT / "models" / "action_transformer_project_final.pt"
    model = load_model(model_path, device)
    outputs = []
    for case in CASES:
        predicted = greedy_generate(
            model,
            case["source"],
            case["verb"],
            case["task"],
            case["object"],
            case["target"],
            device,
            max_actions=12,
        )
        outputs.append({**case, "model": str(model_path.relative_to(ROOT)), "predicted_actions": predicted})

    OUT_JSON.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 最终演示样例集", ""]
    for item in outputs:
        lines.extend([
            f"## {item['title']}",
            "",
            f"- 样例 ID：`{item['id']}`",
            f"- 输入说明：{item['instruction']}",
            f"- 条件：source=`{item['source']}`，verb=`{item['verb']}`，task=`{item['task']}`，object=`{item['object']}`，target=`{item['target']}`",
            "- 预测动作序列：",
            "",
        ])
        lines.extend([f"  {index}. `{action}`" for index, action in enumerate(item["predicted_actions"], start=1)])
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "cases": len(outputs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
