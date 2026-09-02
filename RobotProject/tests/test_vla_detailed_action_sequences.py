import unittest

from tools.vla_action_templates import load_vla_action_templates
from tools.vla_detailed_action_sequences import (
    get_vla_detailed_action_sequence,
    load_vla_detailed_action_sequences,
    validate_vla_detailed_action_sequences,
)


class VlaDetailedActionSequencesTest(unittest.TestCase):
    def test_catalogue_matches_existing_82_template_ids(self):
        detailed = load_vla_detailed_action_sequences()
        original = load_vla_action_templates()

        self.assertEqual(validate_vla_detailed_action_sequences(detailed), [])
        self.assertEqual(
            {row["object_id"] for row in detailed["sequences"]},
            {row["object_id"] for row in original["candidates"]},
        )

    def test_each_sequence_is_independent_and_fine_grained(self):
        for row in load_vla_detailed_action_sequences()["sequences"]:
            with self.subTest(object_id=row["object_id"]):
                self.assertGreaterEqual(len(row["steps"]), 4)
                self.assertEqual(len(row["steps"]), len(set(row["steps"])))
                self.assertTrue(all(step.strip() for step in row["steps"]))

    def test_lookup_returns_approved_garbage_can_steps(self):
        sequence = get_vla_detailed_action_sequence("vla_001")

        self.assertIsNotNone(sequence)
        self.assertEqual(sequence.object_name_zh, "垃圾桶")
        self.assertEqual(
            sequence.steps[0],
            "识别垃圾桶位置、开口/桶盖类型、内袋是否到位及剩余容量",
        )
        self.assertEqual(len(sequence.steps), 5)

    def test_lookup_returns_none_for_non_vla_object(self):
        self.assertIsNone(get_vla_detailed_action_sequence("cup"))


if __name__ == "__main__":
    unittest.main()
