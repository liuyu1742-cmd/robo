import unittest

from tools.build_compliance_catalog import action_plan_for
from tools.manual_instruction_entry import find_expected_actions, resolve_manual_instruction


EXPECTED_PLANS = {
    ("maintenance_management", "flower_pot"): [
        "locate(flower_pot)",
        "inspect_plant_and_pests(flower_pot)",
        "remove_pests(flower_pot)",
        "organize(flower_pot)",
        "inspect(flower_pot)",
    ],
    ("maintenance_management", "watering_can"): [
        "locate(watering_can)",
        "grasp(watering_can)",
        "fill(watering_can)",
        "move(flower_pot)",
        "water(flower_pot,watering_can)",
        "place(watering_can)",
    ],
    ("entertainment_service", "game_controller"): [
        "locate(game_controller)",
        "grasp(game_controller)",
        "connect(game_controller,entertainment_device)",
        "start_game(entertainment_device)",
        "control(game_controller,entertainment_device)",
        "place(game_controller)",
    ],
    ("entertainment_service", "speaker"): [
        "locate(speaker)",
        "turn_on(speaker)",
        "connect(speaker,audio_source)",
        "play(speaker)",
        "adjust_volume(speaker)",
    ],
}


class FourObjectActionTests(unittest.TestCase):
    def test_action_plan_for_returns_exact_custom_sequences(self):
        for key, expected in EXPECTED_PLANS.items():
            with self.subTest(task=key[0], object_id=key[1]):
                self.assertEqual(action_plan_for(*key)[1], expected)

    def test_chinese_and_english_instructions_resolve_to_custom_sequences(self):
        examples = {
            "检查花盆并清除害虫": ("maintenance_management", "flower_pot"),
            "Please fill the watering can and water the flowers": (
                "maintenance_management", "watering_can"
            ),
            "连接游戏手柄并启动游戏": ("entertainment_service", "game_controller"),
            "Turn on the speaker and play some music": (
                "entertainment_service", "speaker"
            ),
        }
        for instruction, expected_key in examples.items():
            with self.subTest(instruction=instruction):
                resolved = resolve_manual_instruction(instruction)
                self.assertEqual((resolved.task, resolved.object), expected_key)
                self.assertEqual(find_expected_actions(resolved, []), EXPECTED_PLANS[expected_key])


if __name__ == "__main__":
    unittest.main()
