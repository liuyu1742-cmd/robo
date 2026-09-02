import unittest

from tools.semantic_action_augmentation import (
    assert_disjoint_splits,
    build_augmented_split,
)


def single_reading_lamp_record():
    return {
        "relation_key": "appliance_management::reading_lamp",
        "acceptance_task": "appliance_management",
        "object": "reading_lamp",
        "target": "none",
        "actions": ["locate(reading_lamp)", "turn_off(reading_lamp)"],
    }


class SemanticActionAugmentationTest(unittest.TestCase):
    def test_augmented_splits_are_mutually_disjoint(self):
        records = [single_reading_lamp_record()]
        train = build_augmented_split(records, "train", 4)
        dev = build_augmented_split(records, "dev", 2)
        assistant = build_augmented_split(records, "assistant_selftest", 1)
        assert_disjoint_splits(train, dev, assistant)
        self.assertEqual({row["split"] for row in assistant}, {"assistant_selftest"})

    def test_assistant_selftest_contains_operation_language(self):
        row = build_augmented_split(
            [single_reading_lamp_record()], "assistant_selftest", 1
        )[0]
        self.assertNotIn("\u5904\u7406", row["instruction"])
        self.assertNotIn("\u5f04\u4e00\u4e0b", row["instruction"])
        self.assertIn("\u5173", row["instruction"])


if __name__ == "__main__":
    unittest.main()
