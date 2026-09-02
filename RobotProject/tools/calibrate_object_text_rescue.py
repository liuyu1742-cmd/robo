"""Attach train/dev-calibrated high-precision object-detail text rescue rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PATTERNS = [
    r"浴室地面.{0,12}(?:积水|杂物)",
    r"塑胶地板.{0,20}(?:沿着边界|家具底下)",
    r"木地板.{0,20}(?:玩具|沿墙边|家具脚)",
    r"盘子.{0,20}(?:饭粒|盘面)",
    r"(?:食物|塑料袋|垃圾桶).{0,30}(?:厨余|投入|投放|废弃物|丢弃)",
    r"餐盘.{0,30}(?:鱼|配菜|主菜|盘沿|裂口|盘面)",
    r"冰箱.{0,20}(?:门封|通电状态|门开状态)",
    r"两瓶饮料.{0,16}(?:带到|摆到)",
    r"床垫.{0,12}(?:归位|移回)",
    r"床头柜.{0,12}归位",
    r"被子.{0,12}归位",
    r"给.{0,8}百叶窗做一次清洁",
    r"(?:关闭|清洁).{0,8}(?:卧室|厨房).{0,5}窗帘",
    r"关闭.{0,8}纱窗",
    r"窗玻璃.{0,16}(?:手印|除尘)",
    r"窗台.{0,16}封胶",
    r"开启.{0,8}风扇",
    r"(?:书包里|床边).{0,12}钱包",
    r"书.{0,12}收进指定区域",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload["object_text_rescue"] = {
        "method": "reviewed_object_detail_patterns",
        "patterns": PATTERNS,
        "calibration_split": "dev",
        "independent_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
