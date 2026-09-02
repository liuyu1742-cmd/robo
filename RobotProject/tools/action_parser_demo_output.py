"""Incremental audit artifacts for the 500-case action parser demo."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tools.action_parser_demo_cases import DemoCase
from tools.action_parser_demo_runner import DemoResult, FAILURE_CATEGORIES


CATEGORY_ZH = {
    "generation_error": "测试命令生成错误",
    "runtime_error": "运行环境错误",
    "model_generation_error": "模型生成错误",
    "object_not_found": "未找到物体",
    "match_failure": "匹配失败",
    "task_mismatch": "任务不一致",
    "object_mismatch": "物体不一致",
    "operation_mismatch": "操作不一致",
    "target_mismatch": "目标不一致",
    "action_sequence_mismatch": "动作序列不一致",
    "other_failure": "其他失败",
}


def _display(value: Any) -> str:
    return "无" if value in (None, "", [], ()) else str(value)


def _numbered_actions(actions: Any) -> list[str]:
    values = list(actions or [])
    if not values:
        return ["  无"]
    return [f"  {index}. {action}" for index, action in enumerate(values, start=1)]


def _failure_reason(result: DemoResult) -> str:
    expected, actual = result.expected, result.actual
    category = result.failure_category
    if category == "task_mismatch":
        return (
            f"期望任务为 {_display(expected.get('task'))}，"
            f"实际匹配任务为 {_display(actual.get('task'))}。"
        )
    if category == "object_mismatch":
        return (
            f"任务匹配后，期望物体为 {_display(expected.get('object'))}，"
            f"实际匹配物体为 {_display(actual.get('object'))}。"
        )
    if category == "operation_mismatch":
        return (
            f"任务和物体已进入比较，期望操作为 {_display(expected.get('operation'))}，"
            f"实际匹配操作为 {_display(actual.get('operation'))}。"
        )
    if category == "target_mismatch":
        return (
            f"任务、物体和操作已匹配，期望目标为 {_display(expected.get('target'))}，"
            f"实际目标为 {_display(actual.get('target'))}。"
        )
    if category == "action_sequence_mismatch":
        return "任务语义字段已通过前置比较，但模型预测/生成动作序列与标准动作序列不完全一致。"
    if category == "object_not_found":
        return "解析链路未识别到可执行的目标物体，无法继续完成动作序列匹配。"
    if category == "match_failure":
        return "输入指令未能形成完整、可执行的任务—物体—操作匹配。"
    if category == "model_generation_error":
        return (
            "任务语义已匹配，但模型未正常完成动作生成："
            f"{_display(actual.get('prediction_error') or result.error)}。"
        )
    if category == "runtime_error":
        return f"完整链路执行期间发生异常：{_display(result.error)}。"
    if category == "generation_error":
        return f"测试指令或期望数据生成失败：{_display(result.error)}。"
    return f"未通过既定验收条件：{_display(result.error)}。"


def format_failure_details(
    results: Iterable[DemoResult], summary: dict[str, Any]
) -> str:
    """Create a grouped, human-readable diagnostic report for failed cases."""
    rows = list(results)
    failed = [result for result in rows if not result.passed]
    lines = [
        "动作解析失败用例详细记录",
        f"随机种子：{summary.get('seed', '无')}",
        f"实际完成：{summary.get('completed', len(rows))} 条",
        f"失败总数：{len(failed)} 条",
        "说明：每条用例同时保留期望语义、实际匹配结果、标准序列和模型生成序列。",
        "",
    ]
    if not failed:
        lines.append("本次测试没有失败用例。")
        return "\n".join(lines) + "\n"

    for category in FAILURE_CATEGORIES:
        category_rows = [row for row in failed if row.failure_category == category]
        lines.append(f"## {CATEGORY_ZH[category]}（{len(category_rows)}条）")
        if not category_rows:
            lines.extend(["本类型没有失败用例。", ""])
            continue
        for result in category_rows:
            expected, actual = result.expected, result.actual
            lines.extend([
                f"用例ID：{result.case_id}",
                f"输入指令：{result.instruction}",
                f"期望任务：{_display(expected.get('task'))}",
                f"实际任务：{_display(actual.get('task'))}",
                f"期望物体：{_display(expected.get('object'))}",
                f"实际物体：{_display(actual.get('object'))}",
                f"期望操作：{_display(expected.get('operation'))}",
                f"实际操作：{_display(actual.get('operation'))}",
                f"期望目标：{_display(expected.get('target'))}",
                f"实际目标：{_display(actual.get('target'))}",
                "标准/期望动作序列：",
                *_numbered_actions(expected.get("actions")),
                "模型预测/生成动作序列：",
                *_numbered_actions(actual.get("actions")),
                f"具体原因：{_failure_reason(result)}",
                f"异常信息：{_display(result.error)}",
                "-" * 72,
            ])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class DemoRunOutput:
    def __init__(self, output_root: Path, *, seed: int, requested: int) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(output_root) / f"{stamp}_seed{seed}_n{requested}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.generated_path = self.run_dir / "generated_cases.json"
        self.results_path = self.run_dir / "case_results.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.csv_path = self.run_dir / "results.csv"
        self.log_path = self.run_dir / "test.log"
        self.summary_text_path = self.run_dir / "summary.txt"
        self.failure_details_path = self.run_dir / "failure_details.txt"
        self._results: list[DemoResult] = []
        self.log(f"测试批次已创建：seed={seed}, requested={requested}")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
            handle.flush()

    def write_generated_cases(self, cases: Iterable[DemoCase]) -> None:
        rows = [case.to_dict() for case in cases]
        self.generated_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.results_path.touch()
        self.log(f"测试集已冻结：{len(rows)} 条")

    def append_result(self, result: DemoResult) -> None:
        self._results.append(result)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
        status = "通过" if result.passed else f"未通过/{result.failure_category}"
        self.log(f"{result.case_id} {status} {result.elapsed_seconds:.3f}s {result.instruction}")

    def finalize(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._write_csv()
        self.summary_text_path.write_text(self._summary_text(summary), encoding="utf-8")
        self.failure_details_path.write_text(
            format_failure_details(self._results, summary), encoding="utf-8"
        )
        self.log(
            f"测试结束：completed={summary['completed']}, passed={summary['passed']}, "
            f"failed={summary['failed']}, accuracy={summary['accuracy_percent']:.2f}%"
        )

    def _write_csv(self) -> None:
        fields = [
            "case_id", "source", "instruction", "expected_task", "expected_object",
            "expected_operation", "expected_target", "actual_route", "actual_task",
            "actual_object", "actual_operation", "actual_target", "expected_actions",
            "actual_actions", "passed", "failure_category", "error", "elapsed_seconds",
        ]
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in self._results:
                expected, actual = result.expected, result.actual
                writer.writerow({
                    "case_id": result.case_id,
                    "source": result.source,
                    "instruction": result.instruction,
                    "expected_task": expected.get("task"),
                    "expected_object": expected.get("object"),
                    "expected_operation": expected.get("operation"),
                    "expected_target": expected.get("target"),
                    "actual_route": actual.get("route"),
                    "actual_task": actual.get("task"),
                    "actual_object": actual.get("object"),
                    "actual_operation": actual.get("operation"),
                    "actual_target": actual.get("target"),
                    "expected_actions": " -> ".join(expected.get("actions") or []),
                    "actual_actions": " -> ".join(actual.get("actions") or []),
                    "passed": result.passed,
                    "failure_category": result.failure_category,
                    "error": result.error,
                    "elapsed_seconds": result.elapsed_seconds,
                })

    @staticmethod
    def _summary_text(summary: dict[str, Any]) -> str:
        status = "未完成（提前停止）" if summary.get("stopped_early") else "已完成"
        lines = [
            "动作解析完整链路测试汇总",
            f"运行状态：{status}",
            f"随机种子：{summary['seed']}",
            f"计划测试：{summary['requested']} 条",
            f"实际完成：{summary['completed']} 条",
            f"通过：{summary['passed']} 条",
            f"未通过：{summary['failed']} 条",
            f"正确率：{summary['accuracy_percent']:.2f}%",
            "",
            "数据源统计：",
        ]
        for source in ("legacy", "vla"):
            item = summary.get("sources", {}).get(source, {"total": 0, "passed": 0, "failed": 0})
            lines.append(
                f"- {source}: 总数 {item['total']}，通过 {item['passed']}，未通过 {item['failed']}"
            )
        lines.extend(["", "失败原因统计："])
        counts = summary.get("failure_categories", {})
        for category in FAILURE_CATEGORIES:
            lines.append(f"- {CATEGORY_ZH[category]}：{counts.get(category, 0)}")
        return "\n".join(lines) + "\n"


__all__ = ["CATEGORY_ZH", "DemoRunOutput", "format_failure_details"]
