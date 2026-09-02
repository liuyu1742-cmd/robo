import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import manual_instruction_test as app
from tools.project_runtime import RuntimeGeneration


class ManualInstructionRuntimeTest(unittest.TestCase):
    def test_manual_runtime_routes_bedtime_environment_to_multi_device_plan(self):
        from argparse import Namespace
        from manual_instruction_test import run_instruction

        instruction = (
            "准备就寝环境：空调设为24℃、关闭电动窗帘、"
            "将卧室灯调至低亮度，并检查各设备状态。"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_instruction(
                Namespace(
                    no_model=True,
                    report=Path(directory) / "multi_device.json",
                    checkpoint=Path(directory) / "unused.pt",
                    device=None,
                    beam_width=1,
                    max_actions=8,
                ),
                instruction,
            )

        self.assertEqual(result["result_type"], "multi_device_coordination")
        self.assertTrue(result["completed"])
        self.assertEqual(len(result["device_plans"]), 3)
        text = app.format_result_text(result)
        self.assertIn("多设备协同方案", text)
        self.assertIn("冲突检查：通过", text)

    def test_manual_runtime_outputs_vla_template(self):
        from argparse import Namespace
        from manual_instruction_test import run_instruction

        with tempfile.TemporaryDirectory() as directory:
            result = run_instruction(
                Namespace(
                    no_model=True,
                    report=Path(directory) / "template.json",
                    checkpoint=Path(directory) / "unused.pt",
                    device=None,
                    beam_width=1,
                    max_actions=8,
                ),
                "把客厅里的多个汽水罐扔进厨房垃圾桶",
            )
        self.assertEqual(
            result["final_actions"],
            [
                "template_step(拿起汽水罐)",
                "template_step(移动至厨房垃圾桶)",
                "template_step(投放汽水罐)",
                "template_step(确认已放入)",
            ],
        )
        self.assertEqual(result["template_selection"]["object_name_zh"], "垃圾桶")
        self.assertIn("预设动作模板", app.format_result_text(result))

    def test_manual_runtime_keeps_only_original_template_actions(self):
        from argparse import Namespace
        from manual_instruction_test import run_instruction

        with tempfile.TemporaryDirectory() as directory:
            result = run_instruction(
                Namespace(
                    no_model=True,
                    report=Path(directory) / "detailed.json",
                    checkpoint=Path(directory) / "unused.pt",
                    device=None,
                    beam_width=1,
                    max_actions=8,
                ),
                "把客厅里的多个汽水罐扔进厨房垃圾桶",
            )

        self.assertEqual(
            result["final_actions"],
            [
                "template_step(拿起汽水罐)",
                "template_step(移动至厨房垃圾桶)",
                "template_step(投放汽水罐)",
                "template_step(确认已放入)",
            ],
        )
        self.assertEqual(result["template_selection"]["steps"][0], "拿起汽水罐")
        self.assertNotIn("detailed_action_sequence", result)
        text = app.format_result_text(result)
        self.assertNotIn("额外细分动作", text)

    def test_run_instruction_passes_resolved_input_to_v2_runtime(self):
        captured = {}

        def fake_generate(model, resolved, **kwargs):
            del model, kwargs
            captured["resolved"] = resolved
            return RuntimeGeneration(
                actions=["locate(flower_pot)"],
                score=1.0,
                source="epic_kitchens_100",
                verb="inspect",
                terminated=True,
                truncated=False,
            )

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                checkpoint=Path("models/action_transformer_project_final.pt"),
                device="cpu",
                beam_width=3,
                max_actions=12,
                no_model=False,
                report=Path(directory) / "manual.json",
            )
            with patch.object(
                app, "load_project_v2_model", return_value=(object(), {})
            ), patch.object(app, "generate_for_instruction", fake_generate):
                result = app.run_instruction(args, "帮我照顾花")

        self.assertEqual(captured["resolved"].object, "flower_pot")
        self.assertTrue(result["model_used"])
        self.assertEqual(result["predicted_actions"], ["locate(flower_pot)"])
        self.assertEqual(result["generation"]["source"], "epic_kitchens_100")

    def test_runtime_reports_corrected_actions_when_model_is_wrong(self):
        wrong = RuntimeGeneration(
            actions=[
                "locate(reading_lamp)",
                "turn_on(reading_lamp)",
                "inspect(reading_lamp)",
            ],
            score=1.0,
            source="manual_instruction",
            verb="turn-off",
            terminated=True,
            truncated=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                checkpoint=Path("models/action_transformer_project_final.pt"),
                device="cpu",
                beam_width=3,
                max_actions=12,
                no_model=False,
                report=Path(directory) / "manual.json",
            )
            with patch.object(
                app, "load_project_v2_model", return_value=(object(), {})
            ), patch.object(app, "generate_for_instruction", return_value=wrong):
                result = app.run_instruction(args, "把阅读灯关掉")

        self.assertTrue(result["constraint_applied"])
        self.assertIn("turn_off(reading_lamp)", result["final_actions"])
        self.assertNotIn("turn_on(reading_lamp)", result["final_actions"])
        self.assertTrue(result["completed"])
        text = app.format_result_text(result)
        self.assertIn("动作约束纠正", text)
        self.assertIn("最终采用动作序列", text)


if __name__ == "__main__":
    unittest.main()
