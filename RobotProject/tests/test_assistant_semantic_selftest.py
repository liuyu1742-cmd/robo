import json
import tempfile
import unittest
from pathlib import Path

from build_assistant_semantic_selftest import build_assistant_selftest


class AssistantSemanticSelftestTest(unittest.TestCase):
    def test_selftest_has_one_assistant_authored_case_per_relation(self):
        catalog = [
            {
                "relation_key": "appliance_management::reading_lamp",
                "acceptance_task": "appliance_management",
                "object": "reading_lamp",
                "target": "none",
                "actions": ["locate(reading_lamp)", "turn_off(reading_lamp)"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = root / "final_hashes.json"
            hashes.write_text(json.dumps({"hashes": []}), encoding="utf-8")
            bundle = build_assistant_selftest(catalog, hashes, [], [])
        self.assertEqual(len(bundle["cases"]), 1)
        self.assertEqual(
            bundle["evidence_level"], "assistant_generated_development_selftest"
        )
        self.assertTrue(bundle["not_independent_human_evidence"])


if __name__ == "__main__":
    unittest.main()
