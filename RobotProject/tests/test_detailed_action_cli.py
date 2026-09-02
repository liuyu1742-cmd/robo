import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import detailed_action_test


def fake_args(report: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=detailed_action_test.DEFAULT_CHECKPOINT,
        catalog=detailed_action_test.DEFAULT_CATALOG,
        device="cpu",
        model_weight=0.35,
        minimum_confidence=0.52,
        minimum_margin=0.08,
        report=report or detailed_action_test.DEFAULT_REPORT,
    )


class _SuccessfulParser:
    def resolve(self, instruction: str) -> dict:
        return {
            "result_type": "object_detailed_action",
            "input": instruction,
            "object_name_zh": "地面",
            "detailed_actions": ["吸尘", "拖地"],
            "control_parameters": "低速",
        }


class DetailedActionCliTest(unittest.TestCase):
    def test_success_contains_only_detailed_actions_and_valid_timing(self):
        with patch("detailed_action_test.parser_from_args", return_value=_SuccessfulParser()), patch(
            "detailed_action_test.time.perf_counter_ns", side_effect=(100, 450)
        ):
            result, text = detailed_action_test.timed_detailed_instruction(
                fake_args(), "帮我把地面清理干净"
            )

        serialized = json.dumps(result, ensure_ascii=False)
        self.assertIn("detailed_actions", serialized)
        self.assertNotIn("original_actions", serialized)
        self.assertNotIn("final_actions", serialized)
        self.assertNotIn("expected_actions", serialized)
        self.assertEqual(result["elapsed_ns"], 350)
        self.assertEqual(
            result["elapsed_ns"],
            result["ended_perf_counter_ns"] - result["started_perf_counter_ns"],
        )
        self.assertEqual(result["elapsed_seconds"], 350 / 1_000_000_000)
        self.assertIn("运行耗时", text)

    def test_report_write_happens_after_the_timed_result_is_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "result.json"
            with patch("detailed_action_test.parser_from_args", return_value=_SuccessfulParser()), patch(
                "detailed_action_test.time.perf_counter_ns", side_effect=(10, 20)
            ), patch("detailed_action_test._write_report") as write_report:
                result = detailed_action_test.run_detailed_instruction(
                    fake_args(report), "帮我把地面清理干净"
                )

        self.assertEqual(result["elapsed_ns"], 10)
        write_report.assert_called_once_with(report, result)

    def test_popup_failure_falls_back_to_terminal(self):
        with patch(
            "detailed_action_test._get_popup_instruction", side_effect=RuntimeError("no display")
        ), patch("builtins.input", return_value="帮我准备早餐") as terminal_input, patch(
            "detailed_action_test.run_detailed_instruction",
            return_value={"result_type": "clarification_required", "input": "帮我准备早餐", "reason": "x", "candidates": []},
        ):
            self.assertEqual(detailed_action_test.main_for_test([]), 0)

        terminal_input.assert_called_once()

    def test_cli_instruction_avoids_gui_and_writes_utf8_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "nested" / "result.json"
            with patch("detailed_action_test.parser_from_args", return_value=_SuccessfulParser()), patch(
                "detailed_action_test._show_popup_result"
            ) as popup:
                code = detailed_action_test.main_for_test(
                    ["--instruction", "帮我把地面清理干净", "--no-gui", "--report", str(report)]
                )
            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["input"], "帮我把地面清理干净")
        self.assertIn("detailed_actions", payload)
        popup.assert_not_called()

    def test_parser_failure_is_readable_and_returns_nonzero(self):
        with patch(
            "detailed_action_test.run_detailed_instruction", side_effect=RuntimeError("checkpoint missing")
        ), patch("builtins.print") as printed:
            code = detailed_action_test.main_for_test(
                ["--instruction", "帮我把地面清理干净", "--no-gui"]
            )

        self.assertEqual(code, 1)
        self.assertIn("解析失败", str(printed.call_args_list))
        self.assertIn("checkpoint missing", str(printed.call_args_list))

    def test_main_propagates_nonzero_process_exit_code(self):
        with patch("detailed_action_test.main_for_test", return_value=1):
            with self.assertRaises(SystemExit) as raised:
                detailed_action_test.main()

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
