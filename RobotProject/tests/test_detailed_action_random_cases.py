import unittest

from tools.detailed_action_catalog import load_detailed_action_catalog
from tools.detailed_action_random_cases import generate_object_subcommand, generate_random_cases


class DetailedActionRandomCasesTest(unittest.TestCase):
    def test_generation_is_reproducible_unique_and_labelled(self):
        first = generate_random_cases(40, 20260828)
        second = generate_random_cases(40, 20260828)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual(len({row.instruction for row in first}), 40)
        labels = {entry["label"] for entry in load_detailed_action_catalog()["entries"]}
        self.assertTrue(all(row.expected_label in labels for row in first))

    def test_batch_covers_both_result_types(self):
        rows = generate_random_cases(20, 7)
        self.assertEqual({row.case_type for row in rows}, {"object", "scene"})

    def test_supports_reference_scale_without_reusing_commands(self):
        rows = generate_random_cases(500, 500)
        self.assertEqual(len(rows), 500)
        self.assertEqual(len({row.instruction for row in rows}), 500)

    def test_scene_instruction_is_not_catalogue_command_copy(self):
        catalog = load_detailed_action_catalog()
        commands = {
            entry["command"]
            for entry in catalog["entries"]
            if entry["entry_type"] == "scene"
        }
        rows = generate_random_cases(80, 19)
        self.assertTrue(all(row.instruction not in commands for row in rows))

    def test_invalid_count_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_random_cases(0, 1)

    def test_object_subcommand_targets_requested_label_deterministically(self):
        catalog = load_detailed_action_catalog()
        entry = next(item for item in catalog["entries"] if item["entry_type"] == "object")

        first = generate_object_subcommand(entry["label"], 1234, catalogue=catalog)
        second = generate_object_subcommand(entry["label"], 1234, catalogue=catalog)

        self.assertEqual(first, second)
        self.assertEqual(first.case_type, "object")
        self.assertEqual(first.expected_label, entry["label"])
        self.assertEqual(first.expected_name, entry["object_name_zh"])
        self.assertTrue(first.instruction)

    def test_object_subcommand_rejects_unknown_and_scene_labels(self):
        catalog = load_detailed_action_catalog()
        scene = next(item for item in catalog["entries"] if item["entry_type"] == "scene")

        with self.assertRaises(KeyError):
            generate_object_subcommand("object:missing", 1, catalogue=catalog)
        with self.assertRaises(ValueError):
            generate_object_subcommand(scene["label"], 1, catalogue=catalog)


if __name__ == "__main__":
    unittest.main()
