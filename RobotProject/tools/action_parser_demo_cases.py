"""Reproducible colloquial test-case generation for the action parser demo."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from tools.action_parser_command_quality import command_quality_violations
from tools.manual_instruction_entry import OBJECT_SYNONYMS
from tools.vla_action_templates import (
    load_vla_action_templates,
    naturalize_vla_goal,
    template_actions,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CATALOGUE = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    source: Literal["legacy", "vla"]
    instruction: str
    expected_task: str
    expected_object: str
    expected_operation: str
    expected_target: str | None
    expected_actions: tuple[str, ...]
    template_id: str
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


TASK_OPERATION = {
    "cleaning": "clean",
    "organizing": "organize",
    "smart_cooking": "cook",
    "appliance_management": "turn_on",
    "security_monitoring": "inspect",
    "laundry": "wash",
    "waste_disposal": "dispose",
    "clothing_care": "organize",
    "window_care": "clean",
    "bedroom_service": "organize",
    "food_serving": "serve",
    "object_fetching": "fetch",
    "elderly_assistance": "assist",
    "maintenance_management": "maintain",
    "entertainment_service": "operate",
}

OPERATION_PHRASES = {
    "clean": ("清理一下{obj}", "把{obj}弄干净", "麻烦收拾一下{obj}"),
    "organize": ("把{obj}收拾整齐", "帮我归置一下{obj}", "把{obj}整理好"),
    "cook": ("用{obj}开始做饭", "把{obj}开起来烹调", "准备好{obj}做菜"),
    "turn_on": ("把{obj}打开", "帮我启动{obj}", "{obj}开一下"),
    "turn_off": ("把{obj}关掉", "帮我关闭{obj}", "{obj}不用了，关一下"),
    "inspect": ("检查一下{obj}", "看看{obj}是否正常", "帮我确认{obj}的状态"),
    "lock": ("把{obj}锁好", "确认{obj}已经上锁", "出门前检查并锁上{obj}"),
    "press": ("按一下{obj}", "帮我触发{obj}", "试一下{obj}是否有反应"),
    "wash": ("把{obj}洗干净", "帮我清洗一下{obj}", "{obj}需要洗一洗"),
    "dispose": ("把{obj}处理掉", "帮我扔掉{obj}", "将{obj}放进回收处"),
    "serve": ("把{obj}送过来", "麻烦端来{obj}", "把{obj}放到餐桌上"),
    "fetch": ("帮我拿来{obj}", "把{obj}递给我", "去取一下{obj}"),
    "assist": ("把{obj}拿来帮忙", "帮老人准备好{obj}", "需要用一下{obj}"),
    "maintain": ("给{obj}做个保养", "检查并维护一下{obj}", "帮我处理{obj}的维护"),
    "operate": ("帮我操作一下{obj}", "启动并使用{obj}", "把{obj}准备好使用"),
    "open": ("把{obj}打开", "帮我开启{obj}", "{obj}开一下"),
    "close": ("把{obj}关好", "帮我关闭{obj}", "确认{obj}已经关上"),
}

PREFIXES = ("", "麻烦", "现在", "等会儿", "顺手", "请", "可以帮我")
SUFFIXES = ("。", "吧。", "，谢谢。", "，弄好后确认一下。", "，尽快处理。")
LOCATIONS = ("客厅里", "厨房里", "卧室里", "书房里", "门口", "走廊上", "桌面上")
QUANTITIES = ("", "这个", "那个", "一个", "几个")

BEDDING_OBJECTS = {
    "bed", "mattress", "quilt", "blanket", "pillow", "pillowcase",
    "bed_sheet", "duvet_cover",
}

FLOOR_OBJECTS = {
    "carpet", "wood_floor", "tile_floor", "marble_floor",
    "laminate_floor", "vinyl_floor",
}

VLA_NATURAL_GOAL_OVERRIDES = {
    "vla_001": "把厨房垃圾桶放稳，再把客厅里的汽水罐扔进去",
    "vla_005": "把客厅里的几罐饮料拿到厨房，扔进垃圾桶",
    "vla_006": "用削皮刀把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
    "vla_007": "把卷心菜和辣椒切成丁，再放进炉灶上的平底锅里烹调",
    "vla_010": "把蔬菜按类别分别放进三个混合碗",
    "vla_011": "把机器人移动到厨房炉灶旁",
    "vla_012": "做完饭把烤箱门关好",
    "vla_013": "把烤面包机的下层烤盘完全推回去",
    "vla_014": "把烤箱上层的烹饪托盘完全拉出来",
    "vla_015": "打开橱柜和玻璃罐，把熟香肠放进罐里，再关好罐子放回橱柜",
    "vla_016": "在砧板上把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
    "vla_017": "把平底锅里的苹果盛到碗里",
    "vla_022": "用餐刀把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
    "vla_023": "把床上的书放到床头柜，再把两只凉鞋并排放在床边",
    "vla_027": "从厨房冰箱拿两瓶饮料放到客厅咖啡台，再把冰箱门关好",
    "vla_030": "把床上的书收到床头柜，再把凉鞋并排放在床边",
    "vla_031": "把台面上的盒装饮料收进橱柜里的收纳盒",
    "vla_032": "把书架上的六本书收到客厅地面的收纳箱里",
    "vla_033": "把卧室收纳篮里的物品分类放到浴室指定位置",
    "vla_034": "把书架上的六本书收到客厅地面的纸箱里",
    "vla_035": "把冰箱抽屉里的苹果放到冰箱层架上",
    "vla_043": "烤箱不用了，把烤箱门关好",
    "vla_045": "把海报挂到厨房的墙钉上",
    "vla_046": "把数码相机安装到卧室的三脚架上",
    "vla_047": "把厨房台面上的海报挂到墙钉上",
    "vla_048": "用相机三脚架接住并支撑好数码相机",
    "vla_058": "把平底锅里的苹果盛到盘子里",
    "vla_060": "把台面上的两顶棒球帽拿到洗衣机里清洗",
    "vla_061": "把毛巾分别放进两个篮子里",
    "vla_062": "启动洗衣机，把两顶棒球帽洗干净",
    "vla_063": "把走廊地面的两双运动鞋和两双凉鞋拿到鞋架，分类并排放好",
    "vla_064": "把鞋架整理好，运动鞋和凉鞋分开并排放置",
    "vla_069": "把铅笔和两支笔放进笔盒，笔盒留在桌面上",
    "vla_071": "把床上的笔记本电脑拿到书桌上并合好屏幕",
    "vla_076": "把卫生巾盒放到卫浴搁板上，并和其他浴室用品分区摆放",
    "vla_077": "把卧室收纳篮里的卫生巾盒拿到浴室搁板上",
    "vla_079": "把卧室收纳篮里的洗涤剂拿到洗手池下方并排放好",
}


def _source_phrase(value: str) -> str:
    value = value.strip()
    suffixes = {
        "台面": "台面上", "橱柜": "橱柜里", "床": "床上",
        "床上": "床上", "书柜": "书柜里", "书架": "书架上",
        "转椅": "转椅上", "冰箱抽屉": "冰箱抽屉里",
        "厨房冰箱": "厨房冰箱里", "走廊地面": "走廊地面上",
    }
    return suffixes.get(value, value)


def _destination_phrase(value: str) -> str:
    value = value.strip()
    suffixes = {
        "台面": "台面上", "橱柜": "橱柜里", "抽屉": "抽屉里",
        "书桌": "书桌上", "桌面": "桌面上", "笔盒": "笔盒里",
    }
    return suffixes.get(value, value)


def _naturalize_vla_goal(relation: dict) -> str:
    return naturalize_vla_goal(relation)


def _chinese_object_name(object_id: str) -> str:
    for alias in OBJECT_SYNONYMS.get(object_id, (object_id,)):
        if re.search(r"[\u4e00-\u9fff]", alias):
            return alias
    return object_id


def _legacy_operation(record: dict) -> str:
    task = record["acceptance_task"]
    actions = list(record["actions"])
    primitives = [action.split("(", 1)[0] for action in actions]
    if task == "appliance_management" and "turn_off" in primitives:
        return "turn_off"
    if task == "security_monitoring":
        for operation in ("lock", "press", "inspect"):
            if operation in primitives:
                return operation
    if task == "window_care":
        if "open" in primitives:
            return "open"
        if "close" in primitives:
            return "close"
    if task == "clothing_care":
        for operation in ("iron", "fold", "inspect"):
            if operation in primitives:
                return operation
    if task == "elderly_assistance" and record["object"] in {
        "blood_pressure_monitor", "thermometer"
    }:
        return "inspect"
    return TASK_OPERATION[task]


def _legacy_relations() -> list[dict]:
    records = json.loads(LEGACY_CATALOGUE.read_text(encoding="utf-8"))
    relations = []
    for record in records:
        operation = _legacy_operation(record)
        relations.append(
            {
                "source": "legacy",
                "task": record["acceptance_task"],
                "object": record["object"],
                "object_name": _chinese_object_name(record["object"]),
                "operation": operation,
                "target": record.get("target"),
                "actions": tuple(record["actions"]),
                "template_id": record["id"],
                "source_operation_label": operation,
            }
        )
    return relations


def _vla_relations() -> list[dict]:
    catalogue = load_vla_action_templates()
    relations = []
    for row in catalogue["candidates"]:
        candidate = next(
            item
            for item in load_vla_action_templates()["candidates"]
            if item["object_id"] == row["object_id"]
        )
        relations.append(
            {
                "source": "vla",
                "task": row["task_id"],
                "object": row["object_id"],
                "object_name": row["object_name_zh"],
                "operation": row["source_operation_label"],
                "target": None,
                "actions": tuple(f"template_step({step})" for step in candidate["steps"]),
                "template_id": f"{row['source_row']['sheet']}:{row['source_row']['row']}",
                "source_operation_label": row["source_operation_label"],
            }
        )
    return relations


def _stratified_order(relations: list[dict], rng: random.Random) -> list[dict]:
    by_task: dict[str, list[dict]] = {}
    for relation in relations:
        by_task.setdefault(relation["task"], []).append(relation)
    for rows in by_task.values():
        rng.shuffle(rows)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    ordered: list[dict] = []
    offset = 0
    while any(offset < len(by_task[task]) for task in tasks):
        for task in tasks:
            if offset < len(by_task[task]):
                ordered.append(by_task[task][offset])
        offset += 1
    return ordered


def _legacy_core(relation: dict, object_text: str, rng: random.Random) -> str:
    operation = relation["operation"]
    task = relation["task"]
    object_id = relation["object"]
    actions = tuple(relation["actions"])

    if task == "laundry":
        source = rng.choice(("卧室衣柜里", "换衣篮里", "床边的衣物篮里"))
        return f"从{source}把{object_text}拿到洗衣机里洗一下"
    if task == "bedroom_service":
        if object_id == "bed":
            return "把卧室里的床整理好"
        return f"把{object_text}整理好"
    if task == "smart_cooking":
        return f"在厨房用{object_text}做饭"
    if task == "cleaning":
        if object_id in FLOOR_OBJECTS:
            return f"把客厅地面的{object_text}清洁干净"
        if any(action.startswith("wash(") for action in actions):
            return f"把{object_text}洗干净"
        return f"把{object_text}清洁干净"
    if task == "clothing_care":
        if operation == "fold":
            return f"把{object_text}叠好"
        if operation == "iron":
            return f"把{object_text}熨烫平整"
        return f"检查一下{object_text}"

    safe_phrases = {
        "clean": "清洁{obj}", "organize": "整理好{obj}", "cook": "用{obj}做饭",
        "turn_on": "打开{obj}", "turn_off": "关闭{obj}", "inspect": "检查{obj}",
        "lock": "锁好{obj}", "press": "按下{obj}", "wash": "清洗{obj}",
        "dispose": "扔掉{obj}", "serve": "把{obj}送过来", "fetch": "把{obj}拿给我",
        "assist": "把{obj}准备给老人", "maintain": "维护保养{obj}",
        "operate": "使用{obj}", "open": "打开{obj}", "close": "关闭{obj}",
        "iron": "熨烫{obj}", "fold": "折叠好{obj}",
    }
    phrase = safe_phrases.get(operation, OPERATION_PHRASES["operate"][0])
    return phrase.format(obj=object_text)


def _legacy_instruction(relation: dict, variant: int, rng: random.Random) -> str:
    del variant
    object_name = relation["object_name"]
    object_text = f"{rng.choice(QUANTITIES)}{object_name}"
    core = _legacy_core(relation, object_text, rng)
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    return f"{prefix}{core}{suffix}"


def _vla_instruction(relation: dict, variant: int, rng: random.Random) -> str:
    del variant
    goal = _naturalize_vla_goal(relation)
    return f"{rng.choice(PREFIXES)}{goal}{rng.choice(SUFFIXES)}"


def generate_demo_cases(count: int = 500, seed: int = 20260813) -> list[DemoCase]:
    if count <= 0:
        raise ValueError("count must be greater than zero")
    rng = random.Random(seed)
    legacy = _stratified_order(_legacy_relations(), rng)
    vla = _stratified_order(_vla_relations(), rng)
    source_rows = {"legacy": legacy, "vla": vla}
    source_offsets = {"legacy": 0, "vla": 0}
    variants: dict[tuple[str, str], int] = {}
    seen: set[str] = set()
    cases: list[DemoCase] = []
    source_cycle = ["legacy", "vla"]
    rng.shuffle(source_cycle)
    attempts = 0
    max_attempts = max(1000, count * 100)

    while len(cases) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("unable to generate enough unique quality-approved commands")
        source = source_cycle[len(cases) % 2]
        rows = source_rows[source]
        offset = source_offsets[source]
        relation = rows[offset % len(rows)]
        source_offsets[source] += 1
        key = (source, relation["template_id"])
        variant = variants.get(key, 0)
        variants[key] = variant + 1
        instruction = (
            _legacy_instruction(relation, variant, rng)
            if source == "legacy"
            else _vla_instruction(relation, variant, rng)
        )
        if instruction in seen:
            continue
        if command_quality_violations(
            instruction,
            task=relation["task"],
            object_id=relation["object"],
            operation=relation["operation"],
        ):
            continue
        seen.add(instruction)
        cases.append(
            DemoCase(
                case_id=f"CASE-{len(cases) + 1:04d}",
                source=source,
                instruction=instruction,
                expected_task=relation["task"],
                expected_object=relation["object"],
                expected_operation=relation["operation"],
                expected_target=relation["target"],
                expected_actions=relation["actions"],
                template_id=relation["template_id"],
                seed=seed,
            )
        )
    return cases


__all__ = ["DemoCase", "generate_demo_cases"]
