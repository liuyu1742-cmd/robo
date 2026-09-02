import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import manual_instruction_test as manual_app
from tools.manual_instruction_entry import (
    build_manual_result,
    find_expected_actions,
    format_result_text,
    load_benchmark_records,
    resolve_manual_instruction,
)
from tools.vla_instruction_planner import (
    format_vla_instruction_result,
    plan_vla_instruction,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOURCE_FILES = (
    "manual_instruction_test.py",
    "final_demo.py",
    "tools/vla_instruction_planner.py",
    "tools/manual_instruction_entry.py",
    "tools/action_parser_demo_cases.py",
    "tools/action_parser_demo_runner.py",
    "tools/run_decision_planning_benchmark.py",
    "tools/update_decision_test_docx.py",
    "tools/build_decision_evidence_dashboard.py",
)
FORBIDDEN_SOURCE_MARKERS = (
    "detailed_action_sequence",
    "get_vla_detailed_action_sequence",
    "vla_detailed_action_sequences",
    "额外细分动作",
    "expected_details",
)


def _collect_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _collect_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _collect_keys(nested)


class LegacyDetailedActionIsolationTest(unittest.TestCase):
    def assert_no_detailed_payload(self, value):
        self.assertNotIn("detailed_action_sequence", set(_collect_keys(value)))

    def test_vla_planner_returns_only_original_template_actions(self):
        result = plan_vla_instruction("把客厅里的多个汽水罐扔进厨房垃圾桶")

        self.assertIsNotNone(result)
        self.assertTrue(result["template_steps"])
        self.assertTrue(result["final_actions"])
        self.assert_no_detailed_payload(result)
        self.assertNotIn("额外细分动作", result["real_output"])
        self.assertNotIn("额外细分动作", format_vla_instruction_result(result))

    def test_manual_builder_and_failure_results_have_no_detailed_field(self):
        self.assertNotIn(
            "detailed_action_sequence",
            inspect.signature(build_manual_result).parameters,
        )
        resolved = resolve_manual_instruction("把杯子洗了")
        actions = find_expected_actions(resolved, load_benchmark_records())
        result = build_manual_result("把杯子洗了", resolved, actions, actions)
        failure = manual_app.build_failure_result("未知命令", ValueError("无法解析"))

        self.assert_no_detailed_payload(result)
        self.assert_no_detailed_payload(failure)
        self.assertNotIn("额外细分动作", format_result_text(result))

    def test_manual_entry_json_does_not_transmit_detailed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "manual.json"
            args = argparse.Namespace(
                no_model=True,
                report=report,
                checkpoint=Path(directory) / "unused.pt",
                device=None,
                beam_width=1,
                max_actions=8,
            )
            result = manual_app.run_instruction(
                args, "把客厅里的多个汽水罐扔进厨房垃圾桶"
            )
            persisted = json.loads(report.read_text(encoding="utf-8"))

        self.assert_no_detailed_payload(result)
        self.assert_no_detailed_payload(persisted)
        self.assertNotIn("额外细分动作", manual_app.format_result_text(result))

    def test_multi_device_result_has_no_nested_detailed_payload(self):
        instruction = (
            "准备就寝环境：空调设为24℃、关闭电动窗帘、"
            "将卧室灯调至低亮度，并检查各设备状态。"
        )
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                no_model=True,
                report=Path(directory) / "multi-device.json",
                checkpoint=Path(directory) / "unused.pt",
                device=None,
                beam_width=1,
                max_actions=8,
            )
            result = manual_app.run_instruction(args, instruction)

        self.assertEqual(result["result_type"], "multi_device_coordination")
        self.assert_no_detailed_payload(result)
        self.assertNotIn("额外细分动作", manual_app.format_result_text(result))

    def test_legacy_source_chain_has_no_detailed_action_dependencies(self):
        for relative_path in LEGACY_SOURCE_FILES:
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            for marker in FORBIDDEN_SOURCE_MARKERS:
                with self.subTest(file=relative_path, marker=marker):
                    self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
