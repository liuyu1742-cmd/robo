import unittest

from tools.vla_instruction_planner import plan_vla_instruction


class VlaActionTemplateRoutingTest(unittest.TestCase):
    def test_plain_legacy_object_request_is_not_claimed_by_vla(self):
        self.assertIsNone(plan_vla_instruction("麻烦把笔记本电脑递给我"))

    def test_full_operation_label_disambiguates_duplicate_vla_object(self):
        result = plan_vla_instruction("请对收纳盒执行盒装饮料：台面→橱柜")
        self.assertIsNotNone(result)
        self.assertEqual(result["resolved"]["object"], "vla_031")
        self.assertEqual(result["resolved"]["task"], "organizing_storage")


if __name__ == "__main__":
    unittest.main()
