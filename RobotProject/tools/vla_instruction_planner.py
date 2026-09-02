"""Shared planning and presentation for approved VLA instruction templates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.vla_action_templates import (
    select_vla_action_template,
    template_actions,
)


def plan_vla_instruction(instruction: str) -> dict[str, Any] | None:
    """Return one approved VLA decision, or ``None`` when no template matches."""
    template = select_vla_action_template(instruction)
    if template is None:
        return None

    actions = template_actions(template)
    template_selection = {
        "task_id": template.task_id,
        "object_id": template.object_id,
        "object_name_zh": template.object_name_zh,
        "source_operation_label": template.source_operation_label,
        "steps": list(template.steps),
    }
    output_text = (
        f"VLA任务识别：{template.task_id}；"
        f"物体：{template.object_name_zh}（{template.object_id}）；"
        f"原始操作：{template.source_operation_label}；"
        "动作模板："
        + " → ".join(template.steps)
    )
    return {
        "result_type": "vla_task_decision",
        "instruction": instruction,
        "resolved": {
            "task": template.task_id,
            "object": template.object_id,
            "object_name_zh": template.object_name_zh,
            "operation": template.source_operation_label,
        },
        "final_actions": actions,
        "template_steps": list(template.steps),
        "template_selection": template_selection,
        "completed": True,
        "completion_status": "通过",
        "completion_explanation": "已匹配审核后的VLA动作模板",
        "real_output": output_text,
    }


def format_vla_instruction_result(result: Mapping[str, Any]) -> str:
    """Format a VLA decision for the popup and command-line presentation."""
    resolved = result["resolved"]
    template_steps = list(result.get("template_steps") or [])
    if not template_steps:
        template_selection = result.get("template_selection") or {}
        template_steps = list(template_selection.get("steps") or [])

    lines = [
        f"你的指令：{result['instruction']}",
        (
            "我理解为：执行VLA任务，"
            f"操作{resolved['object_name_zh']}（{resolved['object']}）。"
        ),
        "预设动作模板：" + " → ".join(template_steps),
        "建议步骤：" + " → ".join(template_steps),
        f"完成情况：{result.get('completion_status', '通过')}",
        f"说明：{result.get('completion_explanation', 'VLA动作模板匹配完成')}",
    ]
    lines.extend(
        [
            "",
            "技术详情（供检查）：",
            f"匹配任务：{resolved['task']}",
            f"匹配物体：{resolved['object']}",
            f"匹配操作：{resolved['operation']}",
        ]
    )
    return "\n".join(lines)
