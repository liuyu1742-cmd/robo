import json
import tempfile
import unittest
from pathlib import Path

from tools.detailed_action_random_cases import RandomDetailedActionCase
from tools.detailed_action_random_output import RandomRunOutput
from tools.detailed_action_random_runner import ChildOperationResult, RandomDetailedActionResult


class DetailedActionRandomOutputTest(unittest.TestCase):
    def test_writes_complete_evidence_bundle(self):
        case = RandomDetailedActionCase("RND-00001", "object", "请整理牙刷", "object:vla:vla_080", "牙刷")
        result = RandomDetailedActionResult(
            case_id=case.case_id, case_type=case.case_type, instruction=case.instruction,
            expected_label=case.expected_label, expected_name=case.expected_name,
            actual_result_type="object_detailed_action", actual_label=case.expected_label,
            actual_name="牙刷", detailed_actions=("竖直放入杯中",), passed=True,
            failure_reason="", error="", started_perf_counter_ns=100,
            ended_perf_counter_ns=2100, elapsed_ns=2000,
            scene_parse_elapsed_ns=500,
            child_operation_count=1,
            child_operation_total_elapsed_ns=1500,
            child_operations=(ChildOperationResult(
                index=1, instruction="请把牙刷整理好", expected_label=case.expected_label,
                actual_result_type="object_detailed_action", actual_label=case.expected_label,
                actual_name="牙刷", detailed_actions=("竖直放入杯中",), passed=True,
                failure_reason="", error="", started_perf_counter_ns=600,
                ended_perf_counter_ns=2100, elapsed_ns=1500,
            ),),
        )
        summary = {
            "completed": 1,
            "overall": {"accuracy_percent": 100.0, "average_elapsed_ms": 0.002},
            "object": {"average_elapsed_ms": 0.002},
            "scene": {
                "average_elapsed_ms": None,
                "average_scene_parse_elapsed_ms": None,
                "average_child_operation_elapsed_ms": None,
                "average_child_total_elapsed_ms": None,
            },
            "acceptance": {"status": "达标", "passed": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = RandomRunOutput(Path(directory), seed=3, requested=1)
            output.write_generated_cases([case])
            output.append_result(result)
            output.finalize(summary)
            expected = {"generated_cases.json", "results.jsonl", "results.csv", "summary.json", "report.md"}
            self.assertTrue(expected.issubset({path.name for path in output.run_dir.iterdir()}))
            saved = json.loads((output.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["overall"]["accuracy_percent"], 100.0)
            report = (output.run_dir / "report.md").read_text(encoding="utf-8")
            csv_text = (output.run_dir / "results.csv").read_text(encoding="utf-8-sig")
            jsonl = json.loads((output.run_dir / "results.jsonl").read_text(encoding="utf-8"))
            self.assertIn("独立物体平均耗时", report)
            self.assertIn("N+1", report)
            self.assertNotIn("验收判定", report)
            self.assertNotIn("达标", report)
            self.assertIn("scene_parse_elapsed_ms", csv_text.splitlines()[0])
            self.assertIn("child_operations", csv_text.splitlines()[0])
            self.assertEqual(jsonl["child_operations"][0]["elapsed_ns"], 1500)


if __name__ == "__main__":
    unittest.main()
