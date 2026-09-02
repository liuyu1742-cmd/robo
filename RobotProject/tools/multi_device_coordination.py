"""Deterministic coordination plans for supported multi-device household scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.manual_instruction_entry import (
    find_expected_actions,
    load_benchmark_records,
    resolve_manual_instruction,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "meta" / "household_task_catalog.json"

BEDTIME_SCENARIO = "bedtime_environment"
BEDTIME_DEVICES = ("air_conditioner", "electric_curtain", "bedroom_lamp")
BEDTIME_TERMS = (
    "准备就寝环境",
    "就寝环境",
    "睡眠模式",
    "睡前环境",
    "我要睡觉了",
    "我想睡觉了",
    "准备睡觉",
    "该睡觉了",
)
BEDTIME_NEGATIVE_TERMS = (
    "不想睡觉",
    "不睡觉",
    "不用准备睡觉环境",
    "不要进入睡眠模式",
)
BEDTIME_SUBTASK_INSTRUCTIONS = (
    ("air_conditioner", "打开空调"),
    ("electric_curtain", "关闭电动窗帘"),
    ("bedroom_lamp", "打开卧室灯"),
)


def _compact(text: str) -> str:
    return "".join(str(text).lower().split()).replace("。", "").replace("，", ",")


def _catalog_objects() -> set[str]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {str(item) for item in catalog.get("objects", [])}


def is_multi_device_instruction(text: str) -> bool:
    """Return True only for a supported coordinated household scene."""
    compact = _compact(text)
    if any(_compact(term) in compact for term in BEDTIME_NEGATIVE_TERMS):
        return False
    return any(_compact(term) in compact for term in BEDTIME_TERMS)


def _temperature_mode(indoor_temperature_c: float) -> tuple[str, bool]:
    if indoor_temperature_c > 28.0:
        return "cool", True
    if indoor_temperature_c < 20.0:
        return "heat", True
    return "auto", False


def _find_conflicts(actions: list[str]) -> list[str]:
    mutually_exclusive = (
        ("turn_on(air_conditioner)", "turn_off(air_conditioner)"),
        ("open(electric_curtain)", "close(electric_curtain)"),
        ("turn_on(bedroom_lamp)", "turn_off(bedroom_lamp)"),
    )
    action_set = set(actions)
    return [
        f"{left} <-> {right}"
        for left, right in mutually_exclusive
        if left in action_set and right in action_set
    ]


def plan_multi_device_instruction(
    instruction: str,
    *,
    environment: dict[str, Any] | None = None,
    benchmark_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one conflict-checked plan for the supported bedtime scene."""
    if not is_multi_device_instruction(instruction):
        raise ValueError("当前指令不是已支持的多设备协同场景")

    available_objects = _catalog_objects()
    missing_devices = [
        device for device in BEDTIME_DEVICES if device not in available_objects
    ]
    if missing_devices:
        raise ValueError("项目设备目录缺少：" + ", ".join(missing_devices))

    environment = dict(environment or {})
    indoor_temperature_c = float(environment.get("indoor_temperature_c", 24.0))
    mode, temperature_triggered = _temperature_mode(indoor_temperature_c)
    change_reason = str(environment.get("change_reason") or "").strip()
    replanning_triggered = temperature_triggered or bool(change_reason)

    records = benchmark_records or load_benchmark_records()
    parsed_subtasks = []
    for device, subtask_instruction in BEDTIME_SUBTASK_INSTRUCTIONS:
        resolved = resolve_manual_instruction(
            subtask_instruction, benchmark_records=records
        )
        if resolved.object != device:
            raise ValueError(
                f"协同子任务解析错误：{subtask_instruction} -> {resolved.object}"
            )
        parsed_subtasks.append(
            {
                "subtask_id": f"bedtime_{device}",
                "instruction": subtask_instruction,
                "planner": "existing_manual_instruction_pipeline",
                "resolved": {
                    "task": resolved.task,
                    "object": resolved.object,
                    "target": resolved.target,
                    "operation": resolved.operation,
                },
                "base_actions": find_expected_actions(resolved, records),
            }
        )
    parsed_by_device = {
        subtask["resolved"]["object"]: subtask for subtask in parsed_subtasks
    }

    device_plans = [
        {
            "device": "air_conditioner",
            "target_state": {
                "power": "on",
                "mode": mode,
                "temperature_c": 24,
            },
            "base_actions": parsed_by_device["air_conditioner"]["base_actions"],
            "actions": [
                *parsed_by_device["air_conditioner"]["base_actions"][:-1],
                f"set_mode(air_conditioner,{mode})",
                "set_temperature(air_conditioner,24C)",
                parsed_by_device["air_conditioner"]["base_actions"][-1],
            ],
        },
        {
            "device": "electric_curtain",
            "target_state": {"position": "closed"},
            "base_actions": parsed_by_device["electric_curtain"]["base_actions"],
            "actions": parsed_by_device["electric_curtain"]["base_actions"],
        },
        {
            "device": "bedroom_lamp",
            "target_state": {"power": "on", "brightness_percent": 20},
            "base_actions": parsed_by_device["bedroom_lamp"]["base_actions"],
            "actions": [
                *parsed_by_device["bedroom_lamp"]["base_actions"][:-1],
                "set_brightness(bedroom_lamp,20%)",
                parsed_by_device["bedroom_lamp"]["base_actions"][-1],
            ],
        },
    ]
    execution_groups = [
        {
            "group": 1,
            "mode": "parallel",
            "purpose": "定位并检查设备当前状态",
            "actions": [
                "locate(air_conditioner)",
                "inspect(air_conditioner)",
                "locate(electric_curtain)",
                "inspect(electric_curtain)",
                "locate(bedroom_lamp)",
                "inspect(bedroom_lamp)",
            ],
        },
        {
            "group": 2,
            "mode": "parallel",
            "purpose": "设置就寝目标状态",
            "actions": [
                "turn_on(air_conditioner)",
                f"set_mode(air_conditioner,{mode})",
                "set_temperature(air_conditioner,24C)",
                "close(electric_curtain)",
                "turn_on(bedroom_lamp)",
                "set_brightness(bedroom_lamp,20%)",
            ],
        },
        {
            "group": 3,
            "mode": "parallel",
            "purpose": "确认各设备最终状态",
            "actions": [
                "confirm_state(air_conditioner)",
                "confirm_state(electric_curtain)",
                "confirm_state(bedroom_lamp)",
            ],
        },
    ]
    final_actions = [
        action for group in execution_groups for action in group["actions"]
    ]
    conflicts = _find_conflicts(final_actions)

    return {
        "result_type": "multi_device_coordination",
        "instruction": instruction,
        "scenario": BEDTIME_SCENARIO,
        "devices": list(BEDTIME_DEVICES),
        "environment": {
            "indoor_temperature_c": indoor_temperature_c,
            "change_reason": change_reason or None,
        },
        "device_plans": device_plans,
        "subtasks": parsed_subtasks,
        "execution_groups": execution_groups,
        "final_actions": final_actions,
        "conflict_check": {
            "passed": not conflicts,
            "conflicts": conflicts,
            "rule": "同一设备不得同时包含互斥目标动作",
        },
        "replanning": {
            "triggered": replanning_triggered,
            "reason": change_reason
            or (
                f"室温{indoor_temperature_c:g}℃超出20℃至28℃舒适范围"
                if temperature_triggered
                else "环境状态稳定，无需重新规划"
            ),
            "air_conditioner_mode": mode,
        },
        "completed": not conflicts,
        "completion_status": "通过" if not conflicts else "未通过",
    }


def format_multi_device_result(result: dict[str, Any]) -> str:
    """Render the coordinated result without claiming physical actuation."""
    replanning = result["replanning"]
    lines = [
        f"你的指令：{result['instruction']}",
        "多设备协同方案：就寝环境",
        "协调设备：空调、电动窗帘、卧室灯",
        "空调目标温度：24℃",
        f"空调运行模式：{replanning['air_conditioner_mode']}",
        "电动窗帘：关闭",
        "卧室灯亮度：20%",
        f"重新规划：{'是' if replanning['triggered'] else '否'}",
        f"重新规划说明：{replanning['reason']}",
        f"冲突检查：{'通过' if result['conflict_check']['passed'] else '未通过'}",
        "",
        "既有动作解析子任务：",
    ]
    for index, subtask in enumerate(result["subtasks"], start=1):
        resolved = subtask["resolved"]
        lines.append(
            f"子任务{index}：{resolved['task']} / {resolved['object']} / "
            f"{resolved['operation']}；基础动作："
            + " → ".join(subtask["base_actions"])
        )
    lines.extend(
        [
        "",
        "执行分组：",
        ]
    )
    for group in result["execution_groups"]:
        lines.append(
            f"{group['group']}. {group['purpose']}（{group['mode']}）："
            + " → ".join(group["actions"])
        )
    lines.extend(
        [
            "",
            f"完成情况：{result['completion_status']}",
            "边界说明：本结果为软件决策与设备调度方案，不代表实体家电已实际执行。",
        ]
    )
    return "\n".join(lines)
