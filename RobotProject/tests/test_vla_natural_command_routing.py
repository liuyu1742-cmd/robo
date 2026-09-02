import unittest

from tools.vla_action_templates import select_vla_action_template


class VlaNaturalCommandRoutingTest(unittest.TestCase):
    def assertRoutesTo(self, instruction: str, object_id: str) -> None:
        selected = select_vla_action_template(instruction)
        self.assertIsNotNone(selected, instruction)
        self.assertEqual(selected.object_id, object_id, instruction)

    def test_distinguishes_three_onion_preparation_objects(self):
        self.assertRoutesTo(
            "用削皮刀把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
            "vla_006",
        )
        self.assertRoutesTo(
            "在砧板上把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
            "vla_016",
        )
        self.assertRoutesTo(
            "用餐刀把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
            "vla_022",
        )

    def test_distinguishes_oven_from_oven_door(self):
        self.assertRoutesTo("做完饭把烤箱门关好", "vla_012")
        self.assertRoutesTo("烤箱不用了，把烤箱门关好", "vla_043")


if __name__ == "__main__":
    unittest.main()
