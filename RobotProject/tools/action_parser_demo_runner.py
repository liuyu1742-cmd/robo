"""Execute reproducible action-parser cases through the real project chain."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from tools.action_parser_demo_cases import DemoCase
from tools.manual_instruction_entry import load_benchmark_records, resolve_manual_instruction
from tools.project_runtime import generate_for_instruction, load_project_v2_model
from tools.vla_instruction_planner import plan_vla_instruction


FAILURE_CATEGORIES = (
    "generation_error",
    "runtime_error",
    "model_generation_error",
    "object_not_found",
    "match_failure",
    "task_mismatch",
    "object_mismatch",
    "operation_mismatch",
    "target_mismatch",
    "action_sequence_mismatch",
    "other_failure",
)


def _normal_target(value: Any) -> str | None:
    if value in (None, "", "none", "None"):
        return None
    return str(value)


def classify_failure(
    case: DemoCase,
    actual: dict[str, Any] | None,
    *,
    error_kind: str | None = None,
) -> str | None:
    """Return the first failed stage, or ``None`` when the case passes."""
    if error_kind in FAILURE_CATEGORIES:
        return error_kind
    if actual is None:
        return "other_failure"
    if actual.get("task") != case.expected_task:
        return "task_mismatch"
    if actual.get("object") != case.expected_object:
        return "object_mismatch"
    if actual.get("operation") != case.expected_operation:
        return "operation_mismatch"
    if (
        _normal_target(case.expected_target) is not None
        and _normal_target(actual.get("target")) != _normal_target(case.expected_target)
    ):
        return "target_mismatch"
    if actual.get("prediction_error"):
        return "model_generation_error"
    if tuple(actual.get("actions") or ()) != tuple(case.expected_actions):
        return "action_sequence_mismatch"
    return None


@dataclass(frozen=True)
class DemoResult:
    case_id: str
    source: str
    instruction: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    passed: bool
    failure_category: str | None
    error: str | None
    elapsed_seconds: float

    @classmethod
    def from_outcome(
        cls,
        case: DemoCase,
        *,
        actual: dict[str, Any] | None,
        passed: bool,
        elapsed_seconds: float,
        failure_category: str | None = None,
        error: str | None = None,
    ) -> "DemoResult":
        return cls(
            case_id=case.case_id,
            source=case.source,
            instruction=case.instruction,
            expected={
                "task": case.expected_task,
                "object": case.expected_object,
                "operation": case.expected_operation,
                "target": case.expected_target,
                "actions": list(case.expected_actions),
                "template_id": case.template_id,
            },
            actual=dict(actual or {}),
            passed=passed,
            failure_category=failure_category,
            error=error,
            elapsed_seconds=round(elapsed_seconds, 6),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DemoBatchRunner:
    """A model-reusing runner that follows the production VLA/legacy routing order."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str | None = None,
        beam_width: int = 3,
        max_actions: int = 12,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beam_width = beam_width
        self.max_actions = max_actions
        self._model = None
        self._checkpoint_metadata: dict[str, Any] | None = None
        self._benchmark_records = load_benchmark_records()

    def prepare(self) -> None:
        if self._model is None:
            self._model, self._checkpoint_metadata = load_project_v2_model(
                self.checkpoint, self.device
            )

    def _execute(self, case: DemoCase) -> dict[str, Any]:
        # This order deliberately mirrors manual_instruction_test.py.
        vla = plan_vla_instruction(case.instruction)
        if vla is not None:
            resolved = vla["resolved"]
            return {
                "route": "vla",
                "task": resolved["task"],
                "object": resolved["object"],
                "operation": resolved["operation"],
                "target": None,
                "actions": list(vla.get("final_actions") or []),
                "prediction_error": None,
            }

        try:
            resolved = resolve_manual_instruction(
                case.instruction, benchmark_records=self._benchmark_records
            )
        except ValueError as exc:
            message = str(exc)
            lowered = message.lower()
            kind = (
                "object_not_found"
                if any(term in lowered for term in ("object", "物体", "具体", "requires"))
                else "match_failure"
            )
            raise DemoExecutionError(kind, message) from exc

        self.prepare()
        generation = generate_for_instruction(
            self._model,
            resolved,
            device=self.device,
            beam_width=self.beam_width,
            max_actions=self.max_actions,
        )
        prediction_error = None
        if generation.invalid:
            prediction_error = "invalid action token"
        elif generation.truncated or not generation.terminated:
            prediction_error = "generation did not terminate normally"
        return {
            "route": "legacy",
            "task": resolved.task,
            "object": resolved.object,
            "operation": resolved.operation,
            "target": resolved.target,
            "actions": list(generation.actions),
            "prediction_error": prediction_error,
            "generation": {
                "score": generation.score,
                "source": generation.source,
                "verb": generation.verb,
                "terminated": generation.terminated,
                "truncated": generation.truncated,
                "invalid": generation.invalid,
            },
        }

    def run_case(self, case: DemoCase) -> DemoResult:
        started = time.perf_counter()
        try:
            actual = self._execute(case)
            category = classify_failure(case, actual)
            return DemoResult.from_outcome(
                case,
                actual=actual,
                passed=category is None,
                failure_category=category,
                elapsed_seconds=time.perf_counter() - started,
            )
        except DemoExecutionError as exc:
            return DemoResult.from_outcome(
                case,
                actual=None,
                passed=False,
                failure_category=exc.kind,
                error=str(exc),
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # keep a 500-case run alive and preserve evidence
            return DemoResult.from_outcome(
                case,
                actual=None,
                passed=False,
                failure_category="runtime_error",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.perf_counter() - started,
            )

    def run(
        self,
        cases: Iterable[DemoCase],
        *,
        on_result: Callable[[DemoResult, int], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[DemoResult]:
        results: list[DemoResult] = []
        for case in cases:
            if stop_requested and stop_requested():
                break
            result = self.run_case(case)
            results.append(result)
            if on_result:
                on_result(result, len(results))
        return results


class DemoExecutionError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def summarize_results(
    results: Iterable[DemoResult], *, requested: int, seed: int
) -> dict[str, Any]:
    rows = list(results)
    passed = sum(row.passed for row in rows)
    failed = len(rows) - passed
    categories = {category: 0 for category in FAILURE_CATEGORIES}
    sources: dict[str, dict[str, int]] = {}
    for row in rows:
        source = sources.setdefault(row.source, {"total": 0, "passed": 0, "failed": 0})
        source["total"] += 1
        source["passed" if row.passed else "failed"] += 1
        if row.failure_category:
            categories[row.failure_category] = categories.get(row.failure_category, 0) + 1
    completed = len(rows)
    return {
        "seed": seed,
        "requested": requested,
        "completed": completed,
        "stopped_early": completed < requested,
        "status": "stopped" if completed < requested else "completed",
        "passed": passed,
        "failed": failed,
        "accuracy_percent": round((passed / completed * 100.0) if completed else 0.0, 2),
        "failure_categories": categories,
        "sources": sources,
        "total_elapsed_seconds": round(sum(row.elapsed_seconds for row in rows), 3),
    }


__all__ = [
    "DemoBatchRunner",
    "DemoResult",
    "FAILURE_CATEGORIES",
    "classify_failure",
    "summarize_results",
]
