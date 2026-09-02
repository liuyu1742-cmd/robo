"""Execute generated detailed-action cases with auditable per-case timing."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.detailed_action_random_cases import RandomDetailedActionCase, generate_object_subcommand


_EXPECTED_RESULT_TYPE = {
    "object": "object_detailed_action",
    "scene": "coordination_detailed_action",
}


@dataclass(frozen=True)
class ChildOperationResult:
    index: int
    instruction: str
    expected_label: str
    actual_result_type: str
    actual_label: str
    actual_name: str
    detailed_actions: tuple[str, ...]
    passed: bool
    failure_reason: str
    error: str
    started_perf_counter_ns: int
    ended_perf_counter_ns: int
    elapsed_ns: int
    resolution_backend: str = ""
    clarification_reason: str = ""

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ns / 1_000_000_000

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["detailed_actions"] = list(self.detailed_actions)
        row["elapsed_ms"] = round(self.elapsed_ms, 6)
        row["elapsed_seconds"] = round(self.elapsed_seconds, 9)
        return row


@dataclass(frozen=True)
class RandomDetailedActionResult:
    case_id: str
    case_type: str
    instruction: str
    expected_label: str
    expected_name: str
    actual_result_type: str
    actual_label: str
    actual_name: str
    detailed_actions: tuple[str, ...]
    passed: bool
    failure_reason: str
    error: str
    started_perf_counter_ns: int
    ended_perf_counter_ns: int
    elapsed_ns: int
    resolution_backend: str = ""
    clarification_reason: str = ""
    scene_parse_elapsed_ns: int = 0
    child_operation_count: int = 0
    child_operation_total_elapsed_ns: int = 0
    child_operations: tuple[ChildOperationResult, ...] = ()

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ns / 1_000_000_000

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["detailed_actions"] = list(self.detailed_actions)
        row["child_operations"] = [child.to_dict() for child in self.child_operations]
        row["elapsed_ms"] = round(self.elapsed_ms, 6)
        row["elapsed_seconds"] = round(self.elapsed_seconds, 9)
        row["scene_parse_elapsed_ms"] = round(self.scene_parse_elapsed_ns / 1_000_000, 6)
        row["child_operation_total_elapsed_ms"] = round(
            self.child_operation_total_elapsed_ns / 1_000_000, 6
        )
        return row


def _actual_name(result: Mapping[str, Any]) -> str:
    if result.get("result_type") == "coordination_detailed_action":
        return str(result.get("scene", result.get("scenario_id", "")))
    return str(result.get("object_name_zh", result.get("object_id", "")))


def _actions(result: Mapping[str, Any]) -> tuple[str, ...]:
    if result.get("result_type") == "coordination_detailed_action":
        return tuple(
            f"{item.get('object_name_zh', item.get('object_label', ''))}: "
            + "；".join(str(action) for action in item.get("detailed_actions", ()))
            for item in result.get("objects", ())
        )
    return tuple(str(action) for action in result.get("detailed_actions", ()))


def _judge(case: RandomDetailedActionCase, result: Mapping[str, Any]) -> tuple[bool, str]:
    if result.get("result_type") != _EXPECTED_RESULT_TYPE[case.case_type]:
        return False, "result_type_mismatch"
    if result.get("label") != case.expected_label:
        return False, "label_mismatch"
    return True, ""


def _stable_child_seed(case_id: str, index: int, object_label: str) -> int:
    digest = hashlib.sha256(f"{case_id}:{index}:{object_label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _run_child_operation(
    parser: Any,
    child_case: RandomDetailedActionCase,
    clock_ns: Callable[[], int],
    index: int,
) -> ChildOperationResult:
    started = clock_ns()
    raw: Mapping[str, Any] = {}
    error = ""
    try:
        raw = parser.resolve(child_case.instruction)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = clock_ns()
    if error:
        passed, failure = False, "parser_error"
    else:
        passed, failure = _judge(child_case, raw)
    return ChildOperationResult(
        index=index,
        instruction=child_case.instruction,
        expected_label=child_case.expected_label,
        actual_result_type=str(raw.get("result_type", "")),
        actual_label=str(raw.get("label", "")),
        actual_name=_actual_name(raw),
        detailed_actions=_actions(raw),
        passed=passed,
        failure_reason=failure,
        error=error,
        started_perf_counter_ns=started,
        ended_perf_counter_ns=ended,
        elapsed_ns=ended - started,
        resolution_backend=str(raw.get("resolution_backend", "")),
        clarification_reason=str(raw.get("reason", "")),
    )


def run_random_cases(
    parser: Any,
    cases: Sequence[RandomDetailedActionCase],
    *,
    on_result: Callable[[RandomDetailedActionResult, int], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    child_case_factory: Callable[[str, int], RandomDetailedActionCase] = generate_object_subcommand,
) -> list[RandomDetailedActionResult]:
    rows: list[RandomDetailedActionResult] = []
    for case in cases:
        if stop_requested is not None and stop_requested():
            break
        started = clock_ns()
        raw: Mapping[str, Any] = {}
        error = ""
        try:
            raw = parser.resolve(case.instruction)
        except Exception as exc:  # keep the batch auditable and running
            error = f"{type(exc).__name__}: {exc}"
        parent_ended = clock_ns()
        if error:
            passed, failure = False, "parser_error"
        else:
            passed, failure = _judge(case, raw)
        scene_parse_elapsed_ns = parent_ended - started if case.case_type == "scene" else 0
        child_operations: list[ChildOperationResult] = []
        if case.case_type == "scene" and passed:
            objects = list(raw.get("objects", ()))
            if len(objects) < 2:
                passed, failure = False, "coordination_requires_multiple_objects"
            else:
                for index, item in enumerate(objects):
                    object_label = str(item.get("object_label", ""))
                    seed = _stable_child_seed(case.case_id, index, object_label)
                    try:
                        child_case = child_case_factory(object_label, seed)
                        child = _run_child_operation(parser, child_case, clock_ns, index + 1)
                    except Exception as exc:
                        child = ChildOperationResult(
                            index=index + 1,
                            instruction="",
                            expected_label=object_label,
                            actual_result_type="",
                            actual_label="",
                            actual_name="",
                            detailed_actions=(),
                            passed=False,
                            failure_reason="child_command_error",
                            error=f"{type(exc).__name__}: {exc}",
                            started_perf_counter_ns=parent_ended,
                            ended_perf_counter_ns=parent_ended,
                            elapsed_ns=0,
                        )
                    child_operations.append(child)
                if not all(child.passed for child in child_operations):
                    passed, failure = False, "child_operation_failed"
        child_elapsed_ns = sum(child.elapsed_ns for child in child_operations)
        total_elapsed_ns = (parent_ended - started) + child_elapsed_ns
        ended = parent_ended if not child_operations else child_operations[-1].ended_perf_counter_ns
        row = RandomDetailedActionResult(
            case_id=case.case_id,
            case_type=case.case_type,
            instruction=case.instruction,
            expected_label=case.expected_label,
            expected_name=case.expected_name,
            actual_result_type=str(raw.get("result_type", "")),
            actual_label=str(raw.get("label", "")),
            actual_name=_actual_name(raw),
            detailed_actions=_actions(raw),
            passed=passed,
            failure_reason=failure,
            error=error,
            started_perf_counter_ns=started,
            ended_perf_counter_ns=ended,
            elapsed_ns=total_elapsed_ns,
            resolution_backend=str(raw.get("resolution_backend", "")),
            clarification_reason=str(raw.get("reason", "")),
            scene_parse_elapsed_ns=scene_parse_elapsed_ns,
            child_operation_count=len(child_operations),
            child_operation_total_elapsed_ns=child_elapsed_ns,
            child_operations=tuple(child_operations),
        )
        rows.append(row)
        if on_result is not None:
            on_result(row, len(rows))
    return rows


def _category_summary(rows: Iterable[RandomDetailedActionResult]) -> dict[str, Any]:
    selected = list(rows)
    total = len(selected)
    passed = sum(row.passed for row in selected)
    elapsed = sum(row.elapsed_ns for row in selected)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy_percent": round(passed / total * 100.0, 2) if total else None,
        "total_elapsed_ns": elapsed,
        "average_elapsed_ns": round(elapsed / total) if total else None,
        "average_elapsed_ms": round(elapsed / total / 1_000_000, 6) if total else None,
    }


def summarize_random_results(
    rows: Sequence[RandomDetailedActionResult], *, requested: int, seed: int
) -> dict[str, Any]:
    overall = _category_summary(rows)
    accuracy = overall["accuracy_percent"] or 0.0
    scene_rows = [row for row in rows if row.case_type == "scene"]
    scene_summary = _category_summary(scene_rows)
    scene_parse_total = sum(row.scene_parse_elapsed_ns for row in scene_rows)
    child_total = sum(row.child_operation_total_elapsed_ns for row in scene_rows)
    child_count = sum(row.child_operation_count for row in scene_rows)
    scene_summary.update({
        "average_scene_parse_elapsed_ms": round(scene_parse_total / len(scene_rows) / 1_000_000, 6)
        if scene_rows else None,
        "average_child_operation_elapsed_ms": round(child_total / child_count / 1_000_000, 6)
        if child_count else None,
        "average_child_total_elapsed_ms": round(child_total / len(scene_rows) / 1_000_000, 6)
        if scene_rows else None,
    })
    return {
        "format": "detailed_action_random_visual_summary_v1",
        "requested": requested,
        "completed": len(rows),
        "seed": seed,
        "object": _category_summary(row for row in rows if row.case_type == "object"),
        "scene": scene_summary,
        "overall": overall,
        "acceptance": {
            "threshold_percent_exclusive": 80.0,
            "passed": len(rows) == requested and accuracy > 80.0,
            "status": "达标" if len(rows) == requested and accuracy > 80.0 else "未达标",
        },
    }
