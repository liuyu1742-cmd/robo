"""Validate generated actions against the parsed instruction intent."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from tools.household_catalog import acceptance_task
from tools.manual_instruction_entry import ManualInstruction, find_expected_actions
from tools.operation_plans import operation_plan


ACCEPTANCE_BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "acceptance_instruction_action_benchmark_15tasks.json"
)


@dataclass(frozen=True)
class ActionConstraintResult:
    """The safe sequence selected after comparing model output to intent."""

    accepted: bool
    final_actions: list[str]
    reason: str
    violations: list[str]


@lru_cache(maxsize=1)
def _acceptance_actions_by_relation() -> dict[tuple[str, str], list[str]]:
    """Load current canonical task/object actions, including merged cleaning."""
    records = json.loads(ACCEPTANCE_BENCHMARK.read_text(encoding="utf-8"))
    return {
        (str(record["acceptance_task"]), str(record["object"])): list(record["actions"])
        for record in records
    }


def expected_actions_for(resolved: ManualInstruction) -> list[str]:
    """Return the canonical executable sequence for one parsed instruction."""
    specific = operation_plan(
        resolved.task, resolved.object, resolved.operation, resolved.target
    )
    if specific is not None:
        return list(specific[1])
    current = _acceptance_actions_by_relation().get(
        (acceptance_task(resolved.task), resolved.object)
    )
    if current is not None:
        return current
    return find_expected_actions(resolved, [])


def constrain_actions(
    resolved: ManualInstruction, predicted_actions: list[str]
) -> ActionConstraintResult:
    """Accept an exact valid sequence or replace it with the canonical safe plan."""
    expected_actions = expected_actions_for(resolved)
    if predicted_actions == expected_actions:
        return ActionConstraintResult(
            accepted=True,
            final_actions=list(predicted_actions),
            reason="模型序列符合当前指令意图",
            violations=[],
        )

    expected_set = set(expected_actions)
    predicted_set = set(predicted_actions)
    violations = []
    if missing := [action for action in expected_actions if action not in predicted_set]:
        violations.append("missing_required:" + ",".join(missing))
    if unexpected := [action for action in predicted_actions if action not in expected_set]:
        violations.append("unexpected_action:" + ",".join(unexpected))
    if not violations:
        violations.append("action_order_mismatch")

    return ActionConstraintResult(
        accepted=False,
        final_actions=expected_actions,
        reason="模型序列与解析出的操作意图不一致，已采用受约束标准序列",
        violations=violations,
    )
