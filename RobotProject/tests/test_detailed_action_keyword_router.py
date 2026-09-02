import unittest

from tools.detailed_action_catalog import load_detailed_action_catalog
from tools.detailed_action_keyword_router import KeywordEnhancedDetailedActionParser


class FakeBaseParser:
    def __init__(self, reason="BGE 执行门控未达到校准阈值"):
        self.reason = reason

    def resolve(self, instruction):
        return {"result_type": "clarification_required", "input": instruction, "reason": self.reason}


class DetailedActionKeywordRouterTest(unittest.TestCase):
    def setUp(self):
        self.catalog = load_detailed_action_catalog()
        self.entry = next(entry for entry in self.catalog["entries"] if entry["label"] == "object:vla:vla_080")

    def test_rescues_unique_object_with_compatible_daily_action(self):
        parser = KeywordEnhancedDetailedActionParser(FakeBaseParser(), catalogue=self.catalog)
        result = parser.resolve(f"麻烦把{self.entry['object_name_zh']}放置好吧")
        self.assertEqual(result["label"], self.entry["label"])
        self.assertEqual(result["result_type"], "object_detailed_action")
        self.assertEqual(result["resolution_backend"], "keyword_catalog_rescue")

    def test_never_rescues_safety_rejection(self):
        parser = KeywordEnhancedDetailedActionParser(
            FakeBaseParser("检测到否定或取消冲突，不执行动作"), catalogue=self.catalog
        )
        result = parser.resolve(f"不要把{self.entry['object_name_zh']}放置好")
        self.assertEqual(result["result_type"], "clarification_required")

    def test_does_not_rescue_incompatible_action(self):
        parser = KeywordEnhancedDetailedActionParser(FakeBaseParser(), catalogue=self.catalog)
        result = parser.resolve(f"把{self.entry['object_name_zh']}加热熟")
        self.assertEqual(result["result_type"], "clarification_required")


if __name__ == "__main__":
    unittest.main()

