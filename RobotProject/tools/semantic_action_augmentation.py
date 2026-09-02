"""Distinct daily-language phrase families for semantic-parser development."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tools.semantic_action_data import (
    _object_names,
    _operation_word,
    normalized_instruction_hash,
)


_TEMPLATES = {
    "train": (
        "请{action}", "麻烦{action}", "帮我{action}", "现在{action}",
        "有空时{action}", "请你{action}", "快帮我{action}", "这个事情请{action}",
        "我想让你{action}", "请立即{action}", "今天{action}", "尽快{action}",
        "这会儿{action}", "请协助{action}", "请负责{action}", "帮忙{action}",
        "请替我{action}", "想请你{action}", "麻烦及时{action}", "帮我完成{action}",
        "请立刻{action}", "劳烦你{action}", "请接手{action}", "这个需要{action}",
        "帮我先{action}", "请马上{action}", "我需要你{action}", "请安排{action}",
        "请为我{action}", "请尽快帮我{action}", "正好要{action}", "现在就{action}",
    ),
    "dev": (
        "睡前{action}吧", "方便的话{action}", "这个时候需要{action}", "我等着用，请{action}",
        "现在正好想{action}", "这件事请帮我{action}", "家里要{action}了", "有空请帮我{action}",
    ),
    "assistant_selftest": (
        "请把这件事办一下：{action}",
        "我需要你马上{action}",
        "为了家里方便，请{action}",
        "能帮我{action}吗", "这个情况请{action}", "请优先{action}",
    ),
}

_TASK_CONTEXT = {
    "cleaning": "家庭清洁",
    "bedroom_service": "卧室服务",
    "clothing_care": "衣物护理",
    "medical_service": "家庭医疗关怀",
    "security_monitoring": "安防检查",
    "maintenance_management": "养护管理",
    "entertainment_service": "娱乐服务",
    "appliance_management": "家电管理",
    "object_fetching": "物品取送",
    "kitchen_service": "厨房服务",
    "bathroom_cleaning": "卫浴清洁",
    "laundry": "衣物洗护",
    "home_organization": "家居整理",
    "elderly_assistance": "医疗关怀",
}


def _action_phrase(object_name: str, actions: Sequence[str]) -> tuple[str, str]:
    verb = _operation_word(actions)
    if verb == "浇水":
        return f"给{object_name}浇水", verb
    return f"{verb}{object_name}", verb


def _ambiguous_relation_pairs(records: Sequence[Mapping]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (str(record["object"]), _operation_word(list(record["actions"])))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_augmented_split(
    records: Sequence[Mapping], split: str, variants_per_relation: int
) -> list[dict]:
    """Generate one named phrase family without accessing any blind-test text."""
    templates = _TEMPLATES.get(split)
    if templates is None:
        raise ValueError(f"unknown split: {split}")
    if variants_per_relation <= 0 or variants_per_relation > len(templates):
        raise ValueError(f"{split} supports 1..{len(templates)} variants")

    rows: list[dict] = []
    seen: set[str] = set()
    pair_counts = _ambiguous_relation_pairs(records)
    for record in records:
        relation = str(record["relation_key"])
        task = str(record["acceptance_task"])
        object_id = str(record["object"])
        names = _object_names(object_id)
        actions = list(record["actions"])
        for index, template in enumerate(templates[:variants_per_relation], start=1):
            object_name = names[(index - 1) % len(names)]
            action, verb = _action_phrase(object_name, actions)
            phrase = template.format(action=action)
            if pair_counts[(object_id, verb)] > 1:
                phrase = f"{phrase}，这是{_TASK_CONTEXT.get(task, task)}的事"
            digest = normalized_instruction_hash(phrase)
            if digest in seen:
                raise ValueError(f"duplicate generated instruction: {phrase}")
            seen.add(digest)
            rows.append(
                {
                    "id": f"augmented_{relation.replace('::', '__')}__{split}__v{index}",
                    "relation_key": relation,
                    "acceptance_task": task,
                    "object": object_id,
                    "target": str(record["target"]),
                    "instruction": phrase,
                    "source": f"augmented_semantic_{split}_v1",
                    "split": split,
                    "variant": index,
                }
            )
    return rows


def assert_disjoint_splits(*splits: Sequence[Mapping]) -> None:
    """Reject any instruction reused between train, dev, or self-test."""
    seen: set[str] = set()
    for split in splits:
        for row in split:
            digest = normalized_instruction_hash(str(row["instruction"]))
            if digest in seen:
                raise ValueError("overlap between semantic data splits")
            seen.add(digest)
