import unittest

from tools.action_parser_command_quality import command_quality_violations


class ActionParserCommandQualityTest(unittest.TestCase):
    def test_rejects_mechanical_operation_wording(self):
        violations = command_quality_violations(
            "按关闭烤箱门的方式操作厨房里的烤箱",
            task="cooking_heating",
            object_id="vla_012",
            operation="关闭烤箱门",
        )
        self.assertIn("mechanical_wording", violations)

    def test_rejects_bedroom_as_laundry_execution_location(self):
        violations = command_quality_violations(
            "在卧室里清洗羊毛衫",
            task="laundry",
            object_id="wool_garment",
            operation="wash",
        )
        self.assertIn("location_action_conflict", violations)

    def test_accepts_bedroom_as_source_when_item_moves_to_washer(self):
        violations = command_quality_violations(
            "把卧室衣柜里的羊毛衫拿到洗衣机里洗一下",
            task="laundry",
            object_id="wool_garment",
            operation="wash",
        )
        self.assertEqual(violations, [])

    def test_rejects_bedding_on_desktop(self):
        violations = command_quality_violations(
            "把桌面上的被子整理好",
            task="bedroom_service",
            object_id="quilt",
            operation="organize",
        )
        self.assertIn("object_location_conflict", violations)

    def test_rejects_bedroom_as_cooking_execution_location(self):
        violations = command_quality_violations(
            "在卧室里用平底锅做饭",
            task="smart_cooking",
            object_id="frying_pan",
            operation="cook",
        )
        self.assertIn("location_action_conflict", violations)


if __name__ == "__main__":
    unittest.main()
