"""Evidence bundle writer for random detailed-action visual tests."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from tools.detailed_action_random_cases import RandomDetailedActionCase
from tools.detailed_action_random_runner import RandomDetailedActionResult


class RandomRunOutput:
    def __init__(self, base_dir: Path, *, seed: int, requested: int) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(base_dir) / f"run_{stamp}_seed{seed}_n{requested}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._results: list[RandomDetailedActionResult] = []

    def write_generated_cases(self, cases: Sequence[RandomDetailedActionCase]) -> None:
        payload = {
            "format": "detailed_action_runtime_generated_cases_v1",
            "generation_policy": "runtime_compositional_grammar_no_split_text",
            "cases": [case.to_dict() for case in cases],
        }
        self._write_json("generated_cases.json", payload)

    def append_result(self, result: RandomDetailedActionResult) -> None:
        self._results.append(result)
        with (self.run_dir / "results.jsonl").open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def finalize(self, summary: dict[str, Any]) -> None:
        self._write_json("summary.json", summary)
        self._write_csv()
        (self.run_dir / "report.md").write_text(self._report(summary), encoding="utf-8")

    def _write_json(self, name: str, payload: Any) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_csv(self) -> None:
        fields = (
            "case_id", "case_type", "instruction", "expected_label", "expected_name",
            "actual_result_type", "actual_label", "actual_name", "detailed_actions",
            "passed", "failure_reason", "clarification_reason", "resolution_backend", "error", "started_perf_counter_ns",
            "ended_perf_counter_ns", "elapsed_ns", "elapsed_ms", "elapsed_seconds",
            "scene_parse_elapsed_ns", "scene_parse_elapsed_ms", "child_operation_count",
            "child_operation_total_elapsed_ns", "child_operation_total_elapsed_ms", "child_operations",
        )
        with (self.run_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in self._results:
                row = result.to_dict()
                row["detailed_actions"] = " -> ".join(row["detailed_actions"])
                row["child_operations"] = json.dumps(
                    row["child_operations"], ensure_ascii=False, separators=(",", ":")
                )
                writer.writerow({field: row.get(field, "") for field in fields})

    @staticmethod
    def _ms(value: Any) -> str:
        return "无样本" if value is None else f"{float(value):.6f} ms"

    def _report(self, summary: dict[str, Any]) -> str:
        overall = summary["overall"]
        return "\n".join((
            "# 细分动作随机口语测试报告",
            "",
            f"- 完成条数：{summary['completed']}",
            f"- 总体准确率：{overall['accuracy_percent']:.2f}%",
            f"- 运行设备：{summary.get('effective_device', 'unknown')} / {summary.get('device_name', 'unknown')}",
            f"- 独立物体平均耗时：{self._ms(summary['object']['average_elapsed_ms'])}",
            f"- 多设备联动总平均耗时：{self._ms(summary['scene']['average_elapsed_ms'])}",
            f"- 场景父解析平均耗时：{self._ms(summary['scene'].get('average_scene_parse_elapsed_ms'))}",
            f"- 单个子操作平均耗时：{self._ms(summary['scene'].get('average_child_operation_elapsed_ms'))}",
            f"- 每条联动子操作合计平均耗时：{self._ms(summary['scene'].get('average_child_total_elapsed_ms'))}",
            f"- 总体平均耗时：{self._ms(overall['average_elapsed_ms'])}",
            "",
            "## 计时说明",
            "",
            "独立物体执行 1 次 `DetailedActionParser.resolve()`。多设备联动执行 N+1 次：先解析 1 次父场景，再对场景涉及的 N 个物体逐一生成口语子命令并各解析 1 次；联动总耗时为这些调用的纯推理耗时之和。命令生成、模型初始化、界面刷新和文件写入不计入正式耗时。逐条及逐子调用纳秒证据见 `results.jsonl` 和 `results.csv`。",
            "",
            "## 生成说明",
            "",
            "本批命令由运行时组合语法现场生成，不读取 train、dev 或冻结独立测试集文本。标准标签随结构化生成过程确定。",
            "",
        ))
