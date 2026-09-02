import unittest

from evaluate_official_fixed_command_acceptance import evaluate_predictions


class OfficialFixedCommandAcceptanceTest(unittest.TestCase):
    def test_exact_sequence_counts_only_identical_ordered_actions(self):
        report = evaluate_predictions(
            [{"id": "a", "expected_actions": ["locate(cup)", "wash(cup)"]}],
            {"a": ["wash(cup)", "locate(cup)"]},
        )
        self.assertEqual(report["raw_passes"], 0)
        self.assertEqual(report["raw_exact_sequence_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
