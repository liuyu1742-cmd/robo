import unittest

from tools.action_constraints import constrain_actions
from tools.manual_instruction_entry import resolve_manual_instruction


class ActionConstraintTest(unittest.TestCase):
    def test_merged_cleaning_uses_current_acceptance_plan(self):
        resolved = resolve_manual_instruction("清洁一下地毯")
        expected = ["locate(carpet)", "clean(carpet)", "inspect(carpet)"]

        result = constrain_actions(resolved, expected)

        self.assertTrue(result.accepted)
        self.assertEqual(result.final_actions, expected)

    def test_turn_off_light_replaces_opposite_model_action(self):
        resolved = resolve_manual_instruction("把阅读灯关掉")
        result = constrain_actions(
            resolved,
            ["locate(reading_lamp)", "turn_on(reading_lamp)", "inspect(reading_lamp)"],
        )

        self.assertFalse(result.accepted)
        self.assertIn("turn_off(reading_lamp)", result.final_actions)
        self.assertNotIn("turn_on(reading_lamp)", result.final_actions)

    def test_television_uses_confirm_state_instead_of_generic_inspect(self):
        resolved = resolve_manual_instruction("打开电视")
        result = constrain_actions(
            resolved,
            ["locate(television)", "turn_on(television)", "inspect(television)"],
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.final_actions[-1], "confirm_state(television)")

    def test_watering_does_not_expand_into_pest_removal(self):
        resolved = resolve_manual_instruction("帮我浇花")
        result = constrain_actions(
            resolved,
            [
                "locate(flower_pot)",
                "inspect_plant_and_pests(flower_pot)",
                "remove_pests(flower_pot)",
            ],
        )

        self.assertFalse(result.accepted)
        self.assertIn("water(flower_pot,watering_can)", result.final_actions)
        self.assertNotIn("remove_pests(flower_pot)", result.final_actions)


if __name__ == "__main__":
    unittest.main()
