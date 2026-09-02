import hashlib
import json
import unittest
from pathlib import Path

from evaluate_acceptance_instruction_benchmark import evaluate_records


ROOT = Path(__file__).resolve().parents[1]


class AcceptanceInstructionBenchmarkTest(unittest.TestCase):
    def test_formal_benchmark_has_exact_canonical_task_set_and_real_predictions(self):
        catalog = json.loads(
            (ROOT / "meta/household_task_catalog.json").read_text(encoding="utf-8")
        )
        records = json.loads(
            (ROOT / "datasets/acceptance_instruction_action_benchmark_15tasks.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {task["id"] for task in catalog["acceptance_tasks"]}
        self.assertEqual({row["acceptance_task"] for row in records}, expected)
        self.assertEqual(len(records), 139)
        self.assertEqual(len({row["object"] for row in records}), 124)
        self.assertTrue(all(row["instruction"] and row["actions"] for row in records))
        cleaning = next(row for row in records if row["acceptance_task"] == "cleaning")
        self.assertIn(cleaning["legacy_task_id"], {"floor_cleaning", "dish_washing", "bathroom_cleaning"})

        report = evaluate_records(records)
        self.assertEqual(report["metrics"]["exact_sequences"], 139)
        self.assertEqual(report["metrics"]["exact_sequence_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["instruction_parse_accuracy"], 1.0)
        self.assertEqual(set(report["metrics"]["per_task"]), expected)
        expected_keys = sorted(
            f"{row['task_id']}::{row['object_id']}"
            for row in catalog["task_object_relations"]
        )
        self.assertEqual(report["audit"]["relation_keys"], expected_keys)
        self.assertEqual(
            report["audit"]["relation_digest"],
            hashlib.sha256("\n".join(expected_keys).encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
