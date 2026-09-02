import unittest

from tools.blind_error_diagnostics import summarize_failures


class BlindErrorDiagnosticsTest(unittest.TestCase):
    def test_groups_catalog_and_sequence_failures_without_mutating_cases(self):
        cases = [
            {
                "acceptance_task": "cleaning",
                "object": "marble_floor",
                "error": "KeyError: 'objects'",
                "raw_exact": False,
            },
            {
                "acceptance_task": "laundry",
                "object": "towel",
                "error": None,
                "raw_exact": False,
            },
            {
                "acceptance_task": "laundry",
                "object": "towel",
                "error": None,
                "raw_exact": True,
            },
        ]

        report = summarize_failures(cases)

        self.assertEqual(report["total_cases"], 3)
        self.assertEqual(report["failures"], 2)
        self.assertEqual(
            report["by_reason"],
            {"catalog_compatibility": 1, "sequence_mismatch": 1},
        )
        self.assertEqual(report["by_task"], {"cleaning": 1, "laundry": 1})
        self.assertEqual(cases[0]["error"], "KeyError: 'objects'")


if __name__ == "__main__":
    unittest.main()
