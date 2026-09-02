import json
import unittest
from collections import defaultdict
from pathlib import Path

from tools.manual_instruction_entry import resolve_manual_instruction


ROOT = Path(__file__).resolve().parents[1]


class DiverseInstructionCatalogTest(unittest.TestCase):
    def test_all_objects_have_natural_non_generic_direct_presets(self):
        records = json.loads(
            (ROOT / "datasets" / "standard_instruction_action_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        direct_by_object = {
            record["object"]: record["preset_instructions"]["direct"]
            for record in records
        }

        self.assertEqual(len(direct_by_object), 124)
        for obj, instruction in direct_by_object.items():
            with self.subTest(obj=obj):
                self.assertTrue(instruction.strip())
                self.assertNotIn("：处理", instruction)

    def test_benchmark_contains_diverse_instruction_styles_and_action_parse(self):
        catalog = json.loads(
            (ROOT / "meta" / "compliance_catalog_15x120.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_pairs = {
            (task["id"], obj)
            for task in catalog["tasks"]
            for obj in task["objects"]
        }
        records = json.loads(
            (ROOT / "datasets" / "standard_instruction_action_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        by_pair = defaultdict(set)
        for record in records:
            self.assertIn("operation", record)
            self.assertTrue(record["operation"])
            by_pair[(record["task"], record["object"])].add(record.get("instruction_style"))
            self.assertIn("action_parse", record)
            action_parse = record["action_parse"]
            self.assertIn("action_primitives", action_parse)
            self.assertIn("visual_observation", action_parse)
            self.assertIn("execution_parameters", action_parse)
            self.assertIn("success_criteria", action_parse)
            self.assertIn("primary_action", action_parse)
            self.assertIn("primitive_details", action_parse)
            self.assertIn("temporal_segments", action_parse)
            self.assertIn("contact_relations", action_parse)
            for primitive in action_parse["primitive_details"]:
                self.assertIn("category", primitive)
                self.assertIn("definition", primitive)
                self.assertIn("kinematic_features", primitive)
                self.assertIn("robot_difficulty", primitive)
                self.assertIn("intent", primitive)

        self.assertEqual(len({obj for _, obj in by_pair}), 124)
        expected_pairs = baseline_pairs | {
            ("maintenance_management", "flower_pot"),
            ("maintenance_management", "watering_can"),
            ("entertainment_service", "game_controller"),
            ("entertainment_service", "speaker"),
        }
        self.assertEqual(len(by_pair), len(expected_pairs))
        self.assertEqual(set(by_pair), expected_pairs)
        for pair, styles in by_pair.items():
            with self.subTest(pair=pair):
                self.assertTrue({"direct", "scene", "constraint"}.issubset(styles))

    def test_pour_water_record_has_chinese_fill_and_pour_primitives(self):
        records = json.loads(
            (ROOT / "datasets" / "standard_instruction_action_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        record = next(
            row for row in records
            if row["task"] == "food_serving"
            and row["object"] == "water_cup"
            and row.get("operation") == "pour_water"
        )
        self.assertIn("fill(water_cup)", record["actions"])
        self.assertIn("pour(water_cup)", record["actions"])
        self.assertIn("接水/注水", record["action_parse"]["action_primitives"])
        self.assertIn("倒水", record["action_parse"]["action_primitives"])

    def test_all_direct_presets_round_trip_to_task_object_and_operation(self):
        records = json.loads(
            (ROOT / "datasets" / "standard_instruction_action_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        direct = [row for row in records if row["instruction_style"] == "direct"]
        for record in direct:
            with self.subTest(record=record["id"]):
                resolved = resolve_manual_instruction(
                    record["instruction"], benchmark_records=records
                )
                self.assertEqual(resolved.task, record["task"])
                self.assertEqual(resolved.object, record["object"])
                self.assertEqual(resolved.operation, record["operation"])

    def test_every_benchmark_instruction_round_trips_to_its_labels(self):
        records = json.loads(
            (ROOT / "datasets" / "standard_instruction_action_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        for record in records:
            with self.subTest(record=record["id"]):
                resolved = resolve_manual_instruction(
                    record["instruction"], benchmark_records=records
                )
                self.assertEqual(resolved.task, record["task"])
                self.assertEqual(resolved.object, record["object"])
                self.assertEqual(resolved.operation, record["operation"])

    def test_direct_preset_actions_respect_operation_final_state(self):
        records = json.loads(
            (ROOT / "datasets" / "standard_instruction_action_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        for record in records:
            if record["instruction_style"] != "direct":
                continue
            actions = record["actions"]
            with self.subTest(record=record["id"]):
                if record["operation"] in {"open", "turn_on"}:
                    self.assertFalse(any(action.startswith(("close(", "turn_off(")) for action in actions))
                if record["operation"] in {"close", "turn_off"}:
                    self.assertFalse(any(action.startswith(("open(", "turn_on(")) for action in actions))
                if record["operation"] == "pour_water":
                    self.assertTrue(any(action.startswith("fill(") for action in actions))
                    self.assertTrue(any(action.startswith("pour(") for action in actions))


if __name__ == "__main__":
    unittest.main()
