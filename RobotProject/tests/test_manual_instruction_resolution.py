import unittest

from tools.manual_instruction_entry import (
    ClarificationRequiredError,
    resolve_manual_instruction,
)


class ManualInstructionResolutionTest(unittest.TestCase):
    def test_current_cleaning_relation_resolves_without_legacy_field_error(self):
        resolved = resolve_manual_instruction("清洁一下大理石地面")

        self.assertEqual(resolved.task, "cleaning")
        self.assertEqual(resolved.object, "marble_floor")

    def test_ambiguous_object_only_request_requires_clarification(self):
        with self.assertRaisesRegex(ClarificationRequiredError, "请说明"):
            resolve_manual_instruction("帮我处理杯子")

    def test_close_reading_lamp_never_resolves_to_turn_on(self):
        resolved = resolve_manual_instruction("睡前把阅读灯关了")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.operation),
            ("appliance_management", "reading_lamp", "turn_off"),
        )

    def test_watering_is_not_plant_maintenance(self):
        resolved = resolve_manual_instruction("花有点干，帮我浇点水")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.operation),
            ("maintenance_management", "flower_pot", "water_plants"),
        )

    def test_tv_switch_off_uses_entertainment_task(self):
        resolved = resolve_manual_instruction("电视不用看了，帮我关掉")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.operation),
            ("entertainment_service", "television", "turn_off"),
        )

    def test_specific_bedroom_object_beats_bed_substring(self):
        self.assertEqual(resolve_manual_instruction("把床垫整理好").object, "mattress")

    def test_specific_window_object_beats_window_substring(self):
        self.assertEqual(resolve_manual_instruction("清洁纱窗").object, "screen_window")

if __name__ == "__main__":
    unittest.main()
