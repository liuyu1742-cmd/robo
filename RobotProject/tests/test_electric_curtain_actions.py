import unittest

from tools.manual_instruction_entry import (
    find_expected_actions,
    load_benchmark_records,
    resolve_manual_instruction,
)


class ElectricCurtainActionsTest(unittest.TestCase):
    def test_electric_curtain_open_and_close_are_position_actions(self):
        records = load_benchmark_records()
        cases = (
            (
                "打开电动窗帘",
                "open",
                [
                    "locate(electric_curtain)",
                    "open(electric_curtain)",
                    "inspect(electric_curtain)",
                ],
            ),
            (
                "关闭电动窗帘",
                "close",
                [
                    "locate(electric_curtain)",
                    "close(electric_curtain)",
                    "inspect(electric_curtain)",
                ],
            ),
        )
        for instruction, operation, actions in cases:
            with self.subTest(instruction=instruction):
                resolved = resolve_manual_instruction(instruction)
                self.assertEqual(resolved.task, "appliance_management")
                self.assertEqual(resolved.object, "electric_curtain")
                self.assertEqual(resolved.operation, operation)
                self.assertEqual(find_expected_actions(resolved, records), actions)


if __name__ == "__main__":
    unittest.main()
