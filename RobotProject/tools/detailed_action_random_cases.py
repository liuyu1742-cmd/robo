"""Runtime-only colloquial cases for the isolated detailed-action parser.

The generator deliberately consumes ontology/catalogue structure only.  It never
loads train, dev, or frozen independent-test utterances.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.detailed_action_catalog import load_detailed_action_catalog


@dataclass(frozen=True)
class RandomDetailedActionCase:
    case_id: str
    case_type: str
    instruction: str
    expected_label: str
    expected_name: str

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "instruction": self.instruction,
            "expected_label": self.expected_label,
            "expected_name": self.expected_name,
        }


_PREFIXES = (
    "麻烦你",
    "帮我",
    "请你",
    "劳驾",
    "现在帮我",
    "能不能帮我",
    "我想让你",
    "这会儿给我",
)
_SUFFIXES = ("一下", "吧", "，谢谢", "，现在就弄", "，辛苦了", "可以吗")
_TASK_VERBS: Mapping[str, tuple[str, str]] = {
    "cleaning": ("清洁一下", "清理干净"),
    "indoor_cleaning": ("清理一下", "收拾干净"),
    "organizing": ("整理一下", "整理好"),
    "organizing_storage": ("收纳一下", "归置好"),
    "object_fetching": ("拿一下", "拿过来"),
    "item_delivery": ("送一下", "送到位"),
    "cooking_heating": ("加热一下", "按步骤处理好"),
    "smart_cooking": ("烹饪一下", "按流程做好"),
    "food_serving": ("准备一下", "摆放好"),
    "tableware_placement": ("摆一下", "摆放好"),
    "storage_open_close": ("操作一下", "开合到位"),
    "laundry": ("洗护一下", "洗护好"),
    "clothing_care": ("护理一下", "护理好"),
    "clothing_entryway": ("收拾一下", "归置好"),
    "bathroom_personal_care": ("整理一下", "放置好"),
    "bedroom_service": ("整理一下", "布置好"),
    "window_care": ("维护一下", "处理好"),
    "security_monitoring": ("检查一下", "检查好"),
    "maintenance_management": ("维护一下", "维护好"),
    "appliance_management": ("操作一下", "设置好"),
    "workspace_service": ("布置一下", "准备好"),
    "entertainment_service": ("设置一下", "准备好"),
    "indoor_installation": ("安装一下", "安装好"),
    "waste_disposal": ("清理一下", "妥善处理"),
    "elderly_assistance": ("准备一下", "放到方便使用的位置"),
}


def _spoken_object(entry: Mapping[str, Any], rng: random.Random) -> str:
    name = str(entry["object_name_zh"])
    ambiguous_surfaces = (
        "杯子", "遥控器", "毛巾", "充电器", "钥匙", "剪刀", "台灯", "垃圾桶",
        "抹布", "拖鞋", "药盒", "水瓶", "眼镜", "花盆", "雨伞", "围裙",
        "笔记本", "洗手液", "靠垫", "门",
    )
    spoken_name = f"我的{name}" if any(surface in name for surface in ambiguous_surfaces) else name
    verb, verb_tail = _TASK_VERBS.get(str(entry.get("task", "")), ("处理一下", "处理好"))
    # Every form contains the catalogue-compatible action phrase.  Randomness
    # changes natural request style without pairing an object with a foreign
    # operation merely to increase surface diversity.
    forms = (
        f"请把{spoken_name}{verb_tail}",
        f"请{verb}{spoken_name}",
        f"帮我把{spoken_name}{verb_tail}",
        f"麻烦把{spoken_name}{verb_tail}",
    )
    suffix = rng.choice(("。", "！", "吧", "，谢谢", "，现在弄一下"))
    return rng.choice(forms) + suffix


def _spoken_scene(entry: Mapping[str, Any], rng: random.Random) -> str:
    # `command` is the formal ontology descriptor, not a stored evaluation
    # utterance.  Wrapping it through independently sampled spoken slots keeps
    # the label auditable while ensuring the emitted command is newly composed.
    command = str(entry["command"]).strip().rstrip("。！!，,")
    # The reviewed command is already short spoken Mandarin.  Runtime-selected
    # pauses and sentence mood create a fresh utterance while keeping the core
    # multi-device intent unambiguous and logically tied to its object set.
    wrappers = (
        ("", "。"), ("", "！"), ("", "……"), ("", "——！"),
        ("……", "。"), ("……", "！"), ("——", "！"), ("——", "……"),
        ("“", "”"), ("“", "！”"),
    )
    before, after = rng.choice(wrappers)
    positions = [index for index, char in enumerate(command) if char in "，、"]
    if positions and rng.random() < 0.5:
        position = rng.choice(positions)
        command = command[:position] + rng.choice(("，", "、", "——")) + command[position + 1:]
    return before + command + after


def _entry_case(entry: Mapping[str, Any], case_id: str, rng: random.Random) -> RandomDetailedActionCase:
    is_scene = entry["entry_type"] == "scene"
    return RandomDetailedActionCase(
        case_id=case_id,
        case_type="scene" if is_scene else "object",
        instruction=_spoken_scene(entry, rng) if is_scene else _spoken_object(entry, rng),
        expected_label=str(entry["label"]),
        expected_name=str(entry["command"] if is_scene else entry["object_name_zh"]),
    )


def generate_object_subcommand(
    object_label: str,
    seed: int,
    *,
    catalogue: Mapping[str, Any] | None = None,
) -> RandomDetailedActionCase:
    """Generate one reproducible spoken command for a requested object label."""
    catalog = catalogue if catalogue is not None else load_detailed_action_catalog()
    entries: Sequence[Mapping[str, Any]] = catalog["entries"]
    entry = next((item for item in entries if str(item.get("label")) == object_label), None)
    if entry is None:
        raise KeyError(f"目录中不存在标签: {object_label}")
    if entry.get("entry_type") != "object":
        raise ValueError(f"子操作标签必须是物体类型: {object_label}")
    return _entry_case(entry, f"CHILD-{seed}", random.Random(seed))


def generate_random_cases(
    count: int = 100,
    seed: int = 20260828,
    *,
    catalogue: Mapping[str, Any] | None = None,
) -> list[RandomDetailedActionCase]:
    """Generate a reproducible, balanced batch of new spoken commands."""
    if count <= 0:
        raise ValueError("测试数量必须大于 0")
    catalog = catalogue if catalogue is not None else load_detailed_action_catalog()
    entries: Sequence[Mapping[str, Any]] = catalog["entries"]
    objects = [entry for entry in entries if entry.get("entry_type") == "object"]
    scenes = [entry for entry in entries if entry.get("entry_type") == "scene"]
    if not objects or not scenes:
        raise ValueError("目录必须同时包含独立物体和多设备场景")

    rng = random.Random(seed)
    rng.shuffle(objects)
    rng.shuffle(scenes)
    pools = {"object": objects, "scene": scenes}
    offsets = {"object": 0, "scene": 0}
    seen: set[str] = set()
    rows: list[RandomDetailedActionCase] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 400:
            raise RuntimeError("运行时口语组合空间不足，无法生成指定数量的唯一命令")
        case_type = ("object", "scene")[len(rows) % 2] if count > 1 else rng.choice(("object", "scene"))
        pool = pools[case_type]
        entry = pool[offsets[case_type] % len(pool)]
        offsets[case_type] += 1
        candidate = _entry_case(entry, f"RND-{len(rows) + 1:05d}", rng)
        if candidate.instruction in seen:
            continue
        seen.add(candidate.instruction)
        rows.append(candidate)
    return rows
