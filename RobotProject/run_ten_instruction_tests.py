"""Run ten realistic household instructions through manual_instruction_test.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manual_instruction_test import DEFAULT_CHECKPOINT, run_instruction


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
REPORT = ROOT / "datasets" / "manual_instruction_ten_case_actual_report.json"

# Seven project task categories, with two deliberately realistic multi-object cases.
CASES = [
    ("ACT-001", "书房桌面收纳", "把书桌上的文件和文具整理整齐。", "organizing", "desk", "organize", None),
    ("ACT-002", "卧室阅读灯关闭", "睡前请关闭阅读灯并确认已经熄灭。", "appliance_management", "reading_lamp", "turn_off", ["locate(reading_lamp)", "turn_off(reading_lamp)", "inspect(reading_lamp)"]),
    ("ACT-003", "出门门锁安防检查", "出门前检查智能门锁是否已经锁好。", "security_monitoring", "smart_lock", "lock", None),
    ("ACT-004", "客厅遥控器取送", "请把茶几上的遥控器递给我。", "object_fetching", "remote_control", "fetch", None),
    ("ACT-005", "卫生间马桶清洁", "请清洁马桶内外表面。", "cleaning", "toilet", "clean", None),
    ("ACT-006", "废旧电池分类回收", "把门口的废电池放入专用回收盒。", "waste_disposal", "battery", "dispose", None),
    ("ACT-007", "早餐电饭煲烹饪", "烹饪前请启动电饭煲煮饭。", "smart_cooking", "rice_cooker", "cook", None),
    ("ACT-008", "玄关可视门铃巡检", "检查玄关的可视门铃是否正常。", "security_monitoring", "video_doorbell", "inspect", None),
    ("ACT-009", "客厅玩具收纳", "把茶几上的玩具放进收纳箱。", "organizing", "toy", "organize", None),
    ("ACT-010", "玄关拐杖取送", "把门口的拐杖拿给我。", "object_fetching", "walking_cane", "fetch", None),
]


def source_actions(expected_task: str, expected_object: str) -> list[str]:
    records = json.loads(CATALOG.read_text(encoding="utf-8"))
    for record in records:
        if record["acceptance_task"] == expected_task and record["object"] == expected_object:
            return list(record["actions"])
    raise ValueError(f"No canonical actions for {expected_task}::{expected_object}")


def main() -> None:
    output: list[dict] = []
    for case_id, scenario, instruction, expected_task, expected_object, expected_operation, action_override in CASES:
        per_case_report = REPORT.with_name(f"manual_instruction_case_{case_id[-3:]}.json")
        args = argparse.Namespace(
            checkpoint=DEFAULT_CHECKPOINT,
            beam_width=3,
            max_actions=12,
            device=None,
            report=per_case_report,
            no_model=False,
        )
        result = run_instruction(args, instruction)
        resolved = result["resolved"]
        expected_actions = action_override or source_actions(expected_task, expected_object)
        semantic_match = (
            resolved["task"] == expected_task
            and resolved["object"] == expected_object
            and resolved["operation"] == expected_operation
        )
        raw_sequence_match = result["predicted_actions"] == expected_actions
        passed = bool(result["prediction_available"] and semantic_match and raw_sequence_match)
        output.append(
            {
                "case_id": case_id,
                "scenario": scenario,
                "input_instruction": instruction,
                "expected": {
                    "task": expected_task,
                    "object": expected_object,
                    "operation": expected_operation,
                    "actions": expected_actions,
                },
                "actual": result,
                "semantic_match": semantic_match,
                "raw_sequence_match": raw_sequence_match,
                "passed": passed,
            }
        )
        print(f"{case_id}: passed={passed}", flush=True)

    passed = sum(item["passed"] for item in output)
    report = {
        "entrypoint": str(ROOT / "manual_instruction_test.py"),
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "selection_method": "stratified_random_project_category_selection_with_realistic_household_phrasing",
        "requested_categories": [
            "organizing", "appliance_management", "security_monitoring", "object_fetching",
            "bathroom_cleaning", "waste_disposal", "smart_cooking",
        ],
        "accuracy_basis": "expected_task_object_operation_and_raw_model_sequence_exact_match",
        "passed": passed,
        "total": len(output),
        "accuracy_percent": passed / len(output) * 100,
        "cases": output,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "total", "accuracy_percent")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
