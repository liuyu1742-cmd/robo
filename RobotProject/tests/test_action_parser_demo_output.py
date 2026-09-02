import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.action_parser_demo_cases import DemoCase
from tools.action_parser_demo_output import DemoRunOutput
from tools.action_parser_demo_runner import DemoResult, summarize_results


class ActionParserDemoOutputTest(unittest.TestCase):
    def test_writer_persists_all_artifacts_with_closed_totals(self):
        case = DemoCase(
            case_id="CASE-0001", source="vla", instruction="把汽水罐收一下",
            expected_task="waste_disposal", expected_object="soda_can",
            expected_operation="投放", expected_target=None,
            expected_actions=("template_step(拿起汽水罐)",),
            template_id="表1:2", seed=7,
        )
        result = DemoResult.from_outcome(case, actual={"actions": []}, passed=False,
                                         failure_category="action_sequence_mismatch",
                                         elapsed_seconds=0.1)
        summary = summarize_results([result], requested=1, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            writer = DemoRunOutput(Path(directory), seed=7, requested=1)
            writer.write_generated_cases([case])
            writer.append_result(result)
            writer.finalize(summary)
            expected = {"generated_cases.json", "case_results.jsonl", "summary.json",
                        "results.csv", "test.log", "summary.txt", "failure_details.txt"}
            self.assertEqual({p.name for p in writer.run_dir.iterdir()}, expected)
            self.assertEqual(len(writer.results_path.read_text(encoding="utf-8").splitlines()), 1)
            loaded = json.loads(writer.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["failed"], 1)
            with writer.csv_path.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            text = writer.summary_text_path.read_text(encoding="utf-8")
            self.assertIn("正确率：0.00%", text)
            self.assertIn("动作序列不一致：1", text)
            details = writer.failure_details_path.read_text(encoding="utf-8")
            self.assertIn("动作序列不一致（1条）", details)
            self.assertIn("用例ID：CASE-0001", details)
            self.assertIn("输入指令：把汽水罐收一下", details)
            self.assertIn("期望任务：waste_disposal", details)
            self.assertIn("实际任务：无", details)
            self.assertIn("标准/期望动作序列：", details)
            self.assertIn("模型预测/生成动作序列：", details)
            self.assertIn("具体原因：", details)

    def test_failure_details_explains_task_mismatch_with_both_values(self):
        case = DemoCase(
            case_id="CASE-0002", source="legacy", instruction="把书整理好",
            expected_task="organizing", expected_object="book",
            expected_operation="organize", expected_target=None,
            expected_actions=("locate(book)", "place(book)"),
            template_id="legacy:book", seed=7,
        )
        result = DemoResult.from_outcome(
            case,
            actual={"task": "object_fetching", "object": "book", "operation": "fetch",
                    "target": None, "actions": ["locate(book)", "grasp(book)"]},
            passed=False, failure_category="task_mismatch", elapsed_seconds=0.1,
        )
        summary = summarize_results([result], requested=1, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            writer = DemoRunOutput(Path(directory), seed=7, requested=1)
            writer.write_generated_cases([case])
            writer.append_result(result)
            writer.finalize(summary)
            details = writer.failure_details_path.read_text(encoding="utf-8")
            self.assertIn("任务不一致（1条）", details)
            self.assertIn("期望任务：organizing", details)
            self.assertIn("实际任务：object_fetching", details)
            self.assertIn(
                "期望任务为 organizing，实际匹配任务为 object_fetching", details
            )
            self.assertIn("1. locate(book)", details)
            self.assertIn("2. grasp(book)", details)


if __name__ == "__main__":
    unittest.main()
