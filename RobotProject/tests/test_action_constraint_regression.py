import unittest

from tools.evaluate_action_constraint_regression import evaluate_cases, load_cases


class ActionConstraintRegressionTest(unittest.TestCase):
    def test_regression_cases_pass_for_semantic_equivalents_and_directional_pairs(self):
        report = evaluate_cases(load_cases())

        self.assertEqual(report["parse_accuracy"], 1.0)
        self.assertEqual(report["final_action_accuracy"], 1.0)
        self.assertGreaterEqual(report["cases"], 10)


if __name__ == "__main__":
    unittest.main()
