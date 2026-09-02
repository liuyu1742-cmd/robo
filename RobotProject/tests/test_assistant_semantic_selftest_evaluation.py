import unittest

from evaluate_assistant_semantic_selftest import markdown_report


class AssistantSemanticSelftestEvaluationTest(unittest.TestCase):
    def test_markdown_never_labels_assistant_selftest_as_human_blind_test(self):
        text = markdown_report(
            {
                "cases": 1,
                "relation_accuracy": 1.0,
                "raw_exact_sequence_accuracy": 1.0,
                "final_exact_sequence_accuracy": 1.0,
            }
        )
        self.assertIn("助手生成自测", text)
        self.assertIn("不能替代独立人工终测", text)


if __name__ == "__main__":
    unittest.main()
