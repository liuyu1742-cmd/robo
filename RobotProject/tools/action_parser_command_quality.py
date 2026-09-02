"""Quality gates for generated colloquial action-parser commands."""

from __future__ import annotations

import re


FORBIDDEN_PATTERNS = (
    re.compile(r"按.+的方式操作"),
    re.compile(r"按照.+方式"),
    re.compile(r"具体是"),
    re.compile(r"场景中"),
    re.compile(r"口语变体"),
    re.compile(r"第\d+种说法"),
)

BEDDING_OBJECTS = {
    "bed", "mattress", "quilt", "blanket", "pillow", "pillowcase",
    "bed_sheet", "duvet_cover",
}

INVALID_BEDDING_LOCATIONS = (
    "桌面", "书桌", "餐桌", "厨房", "门口", "走廊",
)

NON_WASHING_ROOMS = ("卧室", "书房", "客厅")
NON_COOKING_ROOMS = ("卧室", "书房", "客厅", "卫生间", "浴室")


def _room_is_execution_location(
    instruction: str, rooms: tuple[str, ...], action_words: tuple[str, ...]
) -> bool:
    room_pattern = "|".join(map(re.escape, rooms))
    action_pattern = "|".join(map(re.escape, action_words))
    patterns = (
        rf"(?:在|到|去)(?:{room_pattern})(?:里|内)?[^，。]{{0,8}}(?:{action_pattern})",
        rf"(?:{room_pattern})(?:里|内)[^，。]{{0,4}}(?:{action_pattern})",
    )
    return any(re.search(pattern, instruction) for pattern in patterns)


def command_quality_violations(
    instruction: str,
    *,
    task: str,
    object_id: str,
    operation: str,
) -> list[str]:
    """Return stable violation codes for an unsuitable generated command."""
    del operation
    violations: list[str] = []
    if any(pattern.search(instruction) for pattern in FORBIDDEN_PATTERNS):
        violations.append("mechanical_wording")

    if task == "laundry" and "洗衣机" not in instruction:
        if _room_is_execution_location(
            instruction, NON_WASHING_ROOMS, ("清洗", "洗一下", "洗一洗", "洗干净")
        ):
            violations.append("location_action_conflict")

    if task in {"smart_cooking", "cooking_heating"}:
        if _room_is_execution_location(
            instruction, NON_COOKING_ROOMS, ("做饭", "烹调", "烹饪", "加热", "切菜")
        ):
            violations.append("location_action_conflict")

    if object_id in BEDDING_OBJECTS and any(
        location in instruction for location in INVALID_BEDDING_LOCATIONS
    ):
        violations.append("object_location_conflict")

    return violations


def is_acceptable_command(
    instruction: str,
    *,
    task: str,
    object_id: str,
    operation: str,
) -> bool:
    return not command_quality_violations(
        instruction,
        task=task,
        object_id=object_id,
        operation=operation,
    )


__all__ = [
    "BEDDING_OBJECTS",
    "FORBIDDEN_PATTERNS",
    "command_quality_violations",
    "is_acceptable_command",
]
