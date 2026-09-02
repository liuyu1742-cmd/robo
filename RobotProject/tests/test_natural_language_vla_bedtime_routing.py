import argparse
import tempfile
import unittest
from pathlib import Path

import manual_instruction_test as app
from tools.multi_device_coordination import is_multi_device_instruction


def _args(directory: str) -> argparse.Namespace:
    return argparse.Namespace(
        no_model=True,
        report=Path(directory) / "manual_instruction_result.json",
        checkpoint=Path(directory) / "unused.pt",
        device=None,
        beam_width=1,
        max_actions=8,
    )


class NaturalLanguageVlaBedtimeRoutingTest(unittest.TestCase):
    def test_screenshot_vla_phrases_pass_without_generic_parser_failure(self):
        examples = (
            ("把牙膏放到洗手台的杯中", "vla_081"),
            ("把数码相机安装到三脚架上", "vla_046"),
        )
        for instruction, object_id in examples:
            with self.subTest(instruction=instruction), tempfile.TemporaryDirectory() as directory:
                try:
                    result = app.run_instruction(_args(directory), instruction)
                except Exception as exc:  # noqa: BLE001 - convert the current bug to an assertion failure.
                    self.fail(f"入口不应拒绝VLA指令：{exc}")
                self.assertEqual(result["result_type"], "vla_task_decision")
                self.assertEqual(result["resolved"]["object"], object_id)
                self.assertNotIn("detailed_action_sequence", result)
                self.assertTrue(result["completed"])
                self.assertIn("完成情况：通过", app.format_result_text(result))

    def test_screenshot_and_approved_bedtime_phrases_trigger_three_devices(self):
        instructions = ("我要睡觉了", "我想睡觉了", "准备睡觉", "该睡觉了")
        for instruction in instructions:
            with self.subTest(instruction=instruction), tempfile.TemporaryDirectory() as directory:
                try:
                    result = app.run_instruction(_args(directory), instruction)
                except Exception as exc:  # noqa: BLE001 - convert the current bug to an assertion failure.
                    self.fail(f"入口不应拒绝就寝指令：{exc}")
                self.assertEqual(result["result_type"], "multi_device_coordination")
                self.assertEqual(
                    set(result["devices"]),
                    {"air_conditioner", "electric_curtain", "bedroom_lamp"},
                )
                self.assertTrue(result["completed"])
                self.assertIn("完成情况：通过", app.format_result_text(result))

    def test_negative_bedtime_phrases_do_not_trigger_coordination(self):
        instructions = (
            "我不想睡觉",
            "今天不睡觉",
            "不用准备睡觉环境",
            "不要进入睡眠模式",
        )
        for instruction in instructions:
            with self.subTest(instruction=instruction):
                self.assertFalse(is_multi_device_instruction(instruction))


if __name__ == "__main__":
    unittest.main()
