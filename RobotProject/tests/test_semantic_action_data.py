import json
import tempfile
import unittest
from pathlib import Path

from tools.semantic_action_data import (
    assert_no_blind_overlap,
    build_semantic_examples,
    normalized_instruction_hash,
)


class SemanticActionDataTest(unittest.TestCase):
    def test_generator_creates_unique_examples_for_one_relation(self):
        records = [
            {
                "relation_key": "appliance_management::reading_lamp",
                "acceptance_task": "appliance_management",
                "object": "reading_lamp",
                "target": "none",
                "actions": ["locate(reading_lamp)", "turn_off(reading_lamp)"],
            }
        ]

        rows = build_semantic_examples(records, variants_per_relation=4)

        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["instruction"] for row in rows}), 4)
        self.assertEqual(
            {row["relation_key"] for row in rows},
            {"appliance_management::reading_lamp"},
        )

    def test_overlap_check_rejects_a_blind_instruction_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind_hashes.json"
            path.write_text(
                json.dumps({"hashes": [normalized_instruction_hash("睡前把阅读灯关了")]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                assert_no_blind_overlap(
                    [{"instruction": "睡前把阅读灯关了"}], [path]
                )


if __name__ == "__main__":
    unittest.main()
