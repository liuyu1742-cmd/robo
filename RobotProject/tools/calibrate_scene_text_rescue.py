"""Attach the train/dev-calibrated low-false-positive scene text rescue."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PATTERNS = [
    r"大扫除",
    r"整体(?:布置|整理|收拾)",
    r"统筹安排",
    r"收纳系统",
    r"分类整理",
    r"重新归类",
    r"按类别.{0,8}归位",
    r"作品展示",
    r"拍摄布景",
    r"朋友要来",
    r"今晚下厨",
    r"做饭前",
    r"早餐所需",
    r"午饭的",
    r"用品.{0,8}归位",
    r"设施清洗",
    r"(?:这片|整片|整个).{0,6}(?:区域|地面).{0,8}(?:清扫|拖洗|清洁)",
    r"拖洗.{0,8}地面|地面.{0,8}拖洗",
    r"(?:收纳|整理).{0,8}浴室用品|浴室用品.{0,8}(?:收纳|整理)",
    r"整套.{0,8}一起",
    r"(?:厨房用具|厨具).{0,10}(?:也归位|收好|备齐|备妥)",
    r"寒潮.{0,8}(?:来了|将至)|随后也.{0,8}(?:清洁|储存)",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload["scene_text_rescue"] = {
        "method": "reviewed_coordination_patterns",
        "patterns": PATTERNS,
        "calibration_split": "dev",
        "independent_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
