"""Run the reproducible decision-planning benchmark used by the DOCX report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.manual_instruction_entry import (
    describe_actions,
    find_expected_actions,
    load_benchmark_records,
    resolve_manual_instruction,
)
from tools.multi_device_coordination import (
    format_multi_device_result,
    is_multi_device_instruction,
    plan_multi_device_instruction,
)
from tools.vla_action_templates import (
    TASKS,
    load_vla_action_templates,
)
from tools.vla_instruction_planner import plan_vla_instruction


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "decision_planning_benchmark_20260731.json"
DEFAULT_EVIDENCE_DIRECTORY = (
    ROOT / "outputs" / "decision_planning_evidence_20260731"
)
DEFAULT_RANDOM_SEED = 20260731
BEDTIME_INSTRUCTION = (
    "准备就寝环境：空调设为24℃、关闭电动窗帘、"
    "将卧室灯调至低亮度，并检查各设备状态。"
)
TASK_NAMES_ZH = dict(TASKS)


def select_test_cases(random_seed: int = DEFAULT_RANDOM_SEED) -> list[dict[str, Any]]:
    """Select two VLA objects from different VLA tasks plus coordination."""
    rng = random.Random(random_seed)
    candidates = load_vla_action_templates()["candidates"]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_task.setdefault(str(candidate["task_id"]), []).append(candidate)
    selected_tasks = rng.sample(sorted(by_task), 2)

    cases: list[dict[str, Any]] = []
    for index, task_id in enumerate(selected_tasks, start=1):
        candidate = rng.choice(by_task[task_id])
        instruction = (
            f"执行VLA任务：{candidate['source_operation_label']}"
            f"（对象：{candidate['object_name_zh']}）"
        )
        cases.append(
            {
                "case_id": f"DEC-{index:03d}",
                "source": "VLA",
                "task": task_id,
                "object_id": candidate["object_id"],
                "object_name_zh": candidate["object_name_zh"],
                "source_operation_label": candidate["source_operation_label"],
                "instruction": instruction,
                "scene": (
                    f"VLA {TASK_NAMES_ZH[task_id]}任务，"
                    f"操作对象：{candidate['object_name_zh']}"
                ),
                "threshold_seconds": 3.0,
            }
        )
    cases.append(
        {
            "case_id": "DEC-003",
            "source": "动作解析项目",
            "task": "multi_device_coordination",
            "instruction": BEDTIME_INSTRUCTION,
            "scene": "多设备协同就寝综合场景",
            "threshold_seconds": 5.0,
        }
    )
    return cases


def _plan_single_instruction(
    instruction: str, benchmark_records: list[dict[str, Any]]
) -> dict[str, Any]:
    resolved = resolve_manual_instruction(
        instruction, benchmark_records=benchmark_records
    )
    actions = find_expected_actions(resolved, benchmark_records)
    if not actions:
        raise ValueError(f"未生成动作计划：{instruction}")
    readable_actions = describe_actions(actions)
    output_text = (
        f"任务识别：{resolved.task}；物体：{resolved.object}；"
        f"操作：{resolved.operation}；动作序列："
        + " → ".join(actions)
        + "；中文步骤："
        + " → ".join(readable_actions)
    )
    return {
        "result_type": "single_task_decision",
        "instruction": instruction,
        "resolved": {
            "task": resolved.task,
            "object": resolved.object,
            "target": resolved.target,
            "operation": resolved.operation,
        },
        "final_actions": actions,
        "readable_actions": readable_actions,
        "completed": True,
        "real_output": output_text,
    }


def plan_decision_instruction(
    instruction: str,
    *,
    benchmark_records: list[dict[str, Any]] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete software decision output for one instruction."""
    records = benchmark_records or load_benchmark_records()
    if is_multi_device_instruction(instruction):
        result = plan_multi_device_instruction(
            instruction,
            environment=environment,
            benchmark_records=records,
        )
        result["real_output"] = format_multi_device_result(result)
        return result
    if vla_result := plan_vla_instruction(instruction):
        return vla_result
    return _plan_single_instruction(instruction, records)


def _timed_plan(
    instruction: str,
    *,
    benchmark_records: list[dict[str, Any]],
    environment: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = datetime.now().astimezone().isoformat(timespec="microseconds")
    started_ns = time.perf_counter_ns()
    result = plan_decision_instruction(
        instruction,
        benchmark_records=benchmark_records,
        environment=environment,
    )
    serialized = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    ended_ns = time.perf_counter_ns()
    ended_at = datetime.now().astimezone().isoformat(timespec="microseconds")
    elapsed_ns = ended_ns - started_ns
    timing = {
        "started_at": started_at,
        "ended_at": ended_at,
        "started_perf_counter_ns": started_ns,
        "ended_perf_counter_ns": ended_ns,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_ns / 1_000_000_000,
        "output_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
    return result, timing


def _write_evidence(
    *,
    evidence_directory: Path,
    case_id: str,
    run_index: int | str,
    timing: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    run_label = f"{run_index:02d}" if isinstance(run_index, int) else str(run_index)
    evidence_id = f"{case_id}-R{run_label}" if isinstance(run_index, int) else f"{case_id}-{run_label.upper()}"
    filename = (
        f"{case_id}_run_{run_label}.json"
        if isinstance(run_index, int)
        else f"{case_id}_{run_label}.json"
    )
    record = {
        "evidence_id": evidence_id,
        "case_id": case_id,
        "run_index": run_index,
        **timing,
        "structured_output": result,
    }
    evidence_directory.mkdir(parents=True, exist_ok=True)
    (evidence_directory / filename).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**timing, "evidence_id": evidence_id, "evidence_file": filename}


def run_benchmark(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    evidence_directory: Path = DEFAULT_EVIDENCE_DIRECTORY,
    repetitions: int = 10,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Run all cases and persist one independently inspectable file per run."""
    if repetitions < 1:
        raise ValueError("repetitions 必须大于等于 1")

    records = load_benchmark_records()
    cases = select_test_cases(random_seed)
    report_cases: list[dict[str, Any]] = []
    for case in cases:
        runs: list[dict[str, Any]] = []
        last_result: dict[str, Any] | None = None
        for run_index in range(1, repetitions + 1):
            last_result, timing = _timed_plan(
                case["instruction"], benchmark_records=records
            )
            run = _write_evidence(
                evidence_directory=evidence_directory,
                case_id=case["case_id"],
                run_index=run_index,
                timing=timing,
                result=last_result,
            )
            runs.append({"run_index": run_index, **run})
        assert last_result is not None
        elapsed_values = [run["elapsed_seconds"] for run in runs]
        average_seconds = sum(elapsed_values) / len(elapsed_values)
        content_passed = bool(last_result.get("completed"))
        if case["task"] == "multi_device_coordination":
            content_passed = (
                content_passed
                and bool(last_result["conflict_check"]["passed"])
                and len(last_result["subtasks"]) == 3
            )
        qualified = content_passed and average_seconds <= case["threshold_seconds"]
        case_report: dict[str, Any] = {
            **case,
            "runs": runs,
            "elapsed_seconds": elapsed_values,
            "average_seconds": average_seconds,
            "real_output": last_result["real_output"],
            "structured_output": last_result,
            "content_check_passed": content_passed,
            "qualified": qualified,
            "qualification": (
                f"合格：平均耗时{average_seconds:.9f}s≤"
                f"{case['threshold_seconds']:.0f}s，且输出完整"
                if qualified
                else "不合格：耗时或输出完整性未达到标准"
            ),
        }
        if case["task"] == "multi_device_coordination":
            replanned, replan_timing = _timed_plan(
                case["instruction"],
                benchmark_records=records,
                environment={
                    "indoor_temperature_c": 31.0,
                    "change_reason": "测试中室温临时升高至31℃",
                },
            )
            replan_evidence = _write_evidence(
                evidence_directory=evidence_directory,
                case_id=case["case_id"],
                run_index="replan",
                timing=replan_timing,
                result=replanned,
            )
            replan_elapsed = replan_timing["elapsed_seconds"]
            replan_passed = (
                replanned["completed"]
                and replanned["replanning"]["triggered"]
                and replan_elapsed <= case["threshold_seconds"]
            )
            case_report["replanning"] = {
                "triggered": replanned["replanning"]["triggered"],
                "reason": replanned["replanning"]["reason"],
                "elapsed_seconds": replan_elapsed,
                "actions": replanned["final_actions"],
                "real_output": replanned["real_output"],
                "qualified": replan_passed,
                **replan_evidence,
            }
            case_report["qualified"] = qualified and replan_passed
            if case_report["qualified"]:
                case_report["qualification"] += (
                    f"；环境变化重规划{replan_elapsed:.9f}s≤"
                    f"{case['threshold_seconds']:.0f}s"
                )
        report_cases.append(case_report)

    report = {
        "benchmark": "decision_planning_reference_3_docx",
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "timer": "time.perf_counter_ns",
        "timing_formula": (
            "elapsed_seconds=(ended_perf_counter_ns-started_perf_counter_ns)/1,000,000,000"
        ),
        "timing_scope": "从接收指令到完整结构化方案及可序列化真实输出生成完成",
        "random_seed": random_seed,
        "repetitions": repetitions,
        "evidence_directory": str(evidence_directory.resolve()),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor() or "未报告",
            "execution_boundary": "软件决策规划，不包含实体家电通信与物理执行",
        },
        "cases": report_cases,
        "all_qualified": all(case["qualified"] for case in report_cases),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--evidence-directory", type=Path, default=DEFAULT_EVIDENCE_DIRECTORY
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_benchmark(
        output_path=args.output,
        evidence_directory=args.evidence_directory,
        repetitions=args.repetitions,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
