import unittest

import torch

from tools.instruction_benchmark import score_predictions, validate_benchmark
from train_instruction_action_transformer import collate_batch


def record(case_id, split, instruction):
    return {
        "id": case_id,
        "instruction": instruction,
        "task": "floor_cleaning",
        "object": "carpet",
        "target": "floor",
        "actions": ["locate(carpet)", "clean(carpet)", "inspect(carpet)"],
        "split": split,
    }


class InstructionBenchmarkTest(unittest.TestCase):
    def test_validate_accepts_disjoint_templates_and_full_split_coverage(self):
        records = [
            record("a", "train", "执行地面清洁任务：处理 carpet"),
            record("b", "validation", "完成地面清洁，目标物体是 carpet"),
            record("c", "test", "请帮我处理 carpet，执行地面清洁任务"),
        ]

        summary = validate_benchmark(records, expected_tasks=1, expected_objects=1)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["template_overlap_count"], 0)

    def test_validate_rejects_template_leakage_after_object_normalization(self):
        records = [
            record("a", "train", "清洁 carpet"),
            record("b", "validation", "完成地面清洁，目标物体是 carpet"),
            record("c", "test", "清洁  carpet"),
        ]

        with self.assertRaisesRegex(ValueError, "template leakage"):
            validate_benchmark(records, expected_tasks=1, expected_objects=1)

    def test_score_predictions_counts_exact_and_omitted_steps(self):
        expected = [
            record("a", "test", "清洁 carpet"),
            record("b", "test", "再次清洁 carpet"),
        ]
        predictions = {
            "a": ["locate(carpet)", "clean(carpet)", "inspect(carpet)"],
            "b": ["locate(carpet)", "clean(carpet)"],
        }

        metrics = score_predictions(expected, predictions)

        self.assertEqual(metrics["exact_sequence_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["step_accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["omission_rate"], 1 / 6)

    def test_collate_batch_pads_action_and_argument_sequences(self):
        batch = [
            {
                "task": torch.tensor(0),
                "object": torch.tensor(0),
                "target": torch.tensor(0),
                "actions": torch.tensor([1, 3, 2]),
                "arguments": torch.tensor([1, 2, 1]),
            },
            {
                "task": torch.tensor(0),
                "object": torch.tensor(0),
                "target": torch.tensor(0),
                "actions": torch.tensor([1, 3, 4, 2]),
                "arguments": torch.tensor([1, 2, 2, 1]),
            },
        ]

        result = collate_batch(batch)

        self.assertEqual(tuple(result["actions"].shape), (2, 4))
        self.assertEqual(int(result["actions"][0, -1]), 0)


if __name__ == "__main__":
    unittest.main()


class BlindSplitValidationTest(unittest.TestCase):
    def test_validate_accepts_single_split_blind_test_when_requested(self):
        records = [
            record("blind-a", "test", "clean carpet in a new human wording"),
        ]

        summary = validate_benchmark(
            records,
            expected_tasks=1,
            expected_objects=1,
            expected_splits=("test",),
        )

        self.assertEqual(summary["splits"]["test"]["rows"], 1)
