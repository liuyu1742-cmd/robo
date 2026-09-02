import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.build_all_object_coordination_catalog import build_catalog
from tools.vla_detailed_action_sequences import load_vla_detailed_action_sequences


class AllObjectCoordinationCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog()

    def test_exact_source_and_total_counts(self):
        summary = self.catalog["summary"]
        self.assertEqual(summary["original_unique_objects"], 123)
        self.assertEqual(summary["vla_objects"], 82)
        self.assertEqual(summary["total_objects"], 205)

    def test_scenarios_are_real_multi_object_coordination_sized(self):
        scenarios = self.catalog["scenarios"]
        self.assertGreaterEqual(len(scenarios), 45)
        self.assertLessEqual(len(scenarios), 60)
        self.assertTrue(all(2 <= row["object_count"] <= 5 for row in scenarios))
        self.assertTrue(all(row["task_count"] >= 1 for row in scenarios))
        self.assertTrue(all(row["command"].strip() for row in scenarios))
        self.assertTrue(all(row["review_status"] == "待审核" for row in scenarios))

    def test_coordination_commands_are_short_colloquial_and_unique(self):
        commands = [row["command"] for row in self.catalog["scenarios"]]
        self.assertEqual(len(commands), len(set(commands)))
        self.assertTrue(all(len(command) <= 18 for command in commands))
        self.assertTrue(
            all(
                any(marker in command for marker in ("帮我", "我要", "该", "准备", "开始"))
                for command in commands
            )
        )
        self.assertTrue(all("协调处理" not in command for command in commands))
        self.assertTrue(all("先检查环境" not in command for command in commands))

    def test_test_steps_only_request_extra_detailed_actions(self):
        steps = [row["test_steps"] for row in self.catalog["object_details"]]
        self.assertTrue(all("额外细分动作序列" in text for text in steps))
        self.assertTrue(all("原动作序列" not in text for text in steps))

    def test_command_keywords_match_involved_objects(self):
        objects_by_scenario = {}
        for row in self.catalog["object_details"]:
            objects_by_scenario.setdefault(row["scenario_id"], []).append(row)
        floor_ids = {
            "bathroom_floor", "carpet", "laminate_floor", "marble_floor",
            "tile_floor", "vinyl_floor", "wood_floor",
        }
        book_ids = {"book", "vla_023"}
        for scenario in self.catalog["scenarios"]:
            rows = objects_by_scenario[scenario["scenario_id"]]
            object_ids = {row["object_id"] for row in rows}
            command = scenario["command"]
            if "地面" in command:
                self.assertTrue(object_ids & floor_ids, (command, object_ids))
            if "早餐" in command:
                self.assertFalse(object_ids & book_ids, (command, object_ids))
                self.assertTrue(
                    any(
                        row["task"] in {
                            "smart_cooking", "cooking_heating", "food_serving",
                            "tableware_placement",
                        }
                        for row in rows
                    ),
                    (command, object_ids),
                )

    def test_each_scenario_uses_one_explicit_semantic_intent(self):
        self.assertTrue(all(row.get("intent_key") for row in self.catalog["scenarios"]))
        intents_by_scenario = {}
        for row in self.catalog["object_details"]:
            intents_by_scenario.setdefault(row["scenario_id"], set()).add(
                row.get("intent_key")
            )
        self.assertTrue(all(values and len(values) == 1 for values in intents_by_scenario.values()))

    def test_bedtime_command_links_the_three_required_devices(self):
        scenario = next(
            row for row in self.catalog["scenarios"] if row["command"] == "我要睡觉了"
        )
        object_ids = {
            row["object_id"]
            for row in self.catalog["object_details"]
            if row["scenario_id"] == scenario["scenario_id"]
        }
        self.assertEqual(
            object_ids,
            {"air_conditioner", "electric_curtain", "bedroom_lamp"},
        )

    def test_every_object_has_traceable_original_and_detailed_actions(self):
        details = self.catalog["object_details"]
        unique_keys = {(row["source"], row["object_id"]) for row in details}
        self.assertEqual(len(details), 205)
        self.assertEqual(len(unique_keys), 205)
        self.assertTrue(all(row["scenario_id"].startswith("MDC-") for row in details))
        self.assertTrue(all(row["subtask_instruction"].strip() for row in details))
        self.assertTrue(all(row["original_actions"] for row in details))
        self.assertTrue(all(4 <= len(row["detailed_actions"]) <= 8 for row in details))
        self.assertTrue(all(row["target_state"].strip() for row in details))
        self.assertTrue(all(row["conflict_rule"].strip() for row in details))
        self.assertTrue(all(row["review_status"] == "待审核" for row in details))
        self.assertTrue(
            all(row["original_actions"] != row["detailed_actions"] for row in details)
        )

    def test_vla_detailed_actions_are_copied_from_approved_catalog(self):
        approved = {
            row["object_id"]: row["steps"]
            for row in load_vla_detailed_action_sequences()["sequences"]
        }
        actual = {
            row["object_id"]: row["detailed_actions"]
            for row in self.catalog["object_details"]
            if row["source"] == "VLA"
        }
        self.assertEqual(set(actual), set(approved))
        for object_id, steps in approved.items():
            self.assertEqual(actual[object_id], steps)

    def test_coverage_rows_are_complete_and_unique(self):
        coverage = self.catalog["coverage"]
        self.assertEqual(len(coverage), 205)
        self.assertEqual(
            len({(row["source"], row["object_id"]) for row in coverage}), 205
        )
        self.assertTrue(all(row["covered"] for row in coverage))
        self.assertTrue(all(row["detail_count"] >= 1 for row in coverage))

    def test_builder_can_run_directly_from_project_root(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "tools" / "build_all_object_coordination_catalog.py"),
                    "--output",
                    str(output),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertIn("total=205/205", completed.stdout)


if __name__ == "__main__":
    unittest.main()
