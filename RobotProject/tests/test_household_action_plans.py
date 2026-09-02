import unittest

from tools.build_compliance_catalog import action_parse, action_plan_for, build_records


class HouseholdActionPlansTest(unittest.TestCase):
    def test_semantic_operations_override_generic_task_templates(self):
        self.assertEqual(
            action_plan_for("clothing_care", "bed_sheet", "iron")[1],
            ["locate(bed_sheet)", "grasp(iron)", "iron(bed_sheet)", "inspect(bed_sheet)"],
        )
        self.assertEqual(
            action_plan_for("elderly_assistance", "blood_pressure_monitor", "inspect")[1],
            ["locate(blood_pressure_monitor)", "inspect(blood_pressure_monitor)"],
        )
        self.assertEqual(
            action_plan_for("clothing_care", "iron", "inspect")[1],
            ["locate(iron)", "inspect(iron)"],
        )

    def test_training_records_include_explicit_table_destination(self):
        records = build_records()
        table_records = [
            record for record in records
            if record["task"] == "food_serving"
            and record["object"] == "food_tray"
            and record["target"] == "table"
        ]

        self.assertEqual(len(table_records), 4)
        self.assertTrue(
            all("move(table)" in record["actions"] for record in table_records)
        )

    def test_window_operation_plans_do_not_add_opposite_actions(self):
        self.assertEqual(
            action_plan_for("window_care", "window", "open")[1],
            ["locate(window)", "open(window)", "inspect(window)"],
        )
        self.assertEqual(
            action_plan_for("window_care", "window", "close")[1],
            ["locate(window)", "close(window)", "inspect(window)"],
        )
        self.assertEqual(
            action_plan_for("window_care", "window", "clean")[1],
            ["locate(window)", "wipe(window)", "inspect(window)"],
        )

    def test_water_pouring_plan_contains_fill_and_pour(self):
        self.assertEqual(
            action_plan_for("food_serving", "water_cup", "pour_water")[1],
            [
                "locate(water_cup)",
                "grasp(water_cup)",
                "move(water_source)",
                "fill(water_cup)",
                "pour(water_cup)",
                "move(person)",
                "place(water_cup)",
            ],
        )

    def test_object_specific_direct_operations_have_matching_actions(self):
        cases = (
            ("floor_cleaning", "robot_vacuum", "robot_clean", "turn_on(robot_vacuum)"),
            ("security_monitoring", "smart_lock", "lock", "lock(smart_lock)"),
            ("security_monitoring", "padlock", "lock", "lock(padlock)"),
            ("security_monitoring", "doorbell_button", "press", "press(doorbell_button)"),
            ("bedroom_service", "bedroom_lamp", "turn_on", "turn_on(bedroom_lamp)"),
        )
        for task, obj, operation, required_action in cases:
            with self.subTest(task=task, obj=obj):
                actions = action_plan_for(task, obj, operation)[1]
                self.assertIn(required_action, actions)
                if operation == "turn_on":
                    self.assertNotIn(f"turn_off({obj})", actions)

    def test_all_bedroom_objects_have_specific_plans(self):
        objects = ("bed", "pillow", "quilt", "mattress", "nightstand", "bedroom_lamp", "clothes_hanger", "laundry_basket")
        plans = {obj: action_plan_for("bedroom_service", obj)[1] for obj in objects}
        self.assertEqual(len({tuple(plan) for plan in plans.values()}), len(objects))

    def test_pillow_plan_is_complete(self):
        self.assertEqual(action_plan_for("bedroom_service", "pillow")[1], [
            "locate(pillow)", "grasp(pillow)", "fluff(pillow)",
            "orient(pillow)", "place(pillow)", "inspect(pillow)",
        ])

    def test_mattress_plan_is_local_and_never_grasps(self):
        actions = action_plan_for("bedroom_service", "mattress")[1]
        self.assertEqual(actions, ["locate(mattress)", "inspect(mattress)", "straighten(mattress)", "inspect(mattress)"])
        self.assertNotIn("grasp(mattress)", actions)

    def test_new_bedroom_verbs_have_safe_taxonomy_mappings(self):
        for obj in ("pillow", "quilt", "mattress"):
            target, actions = action_plan_for("bedroom_service", obj)
            details = action_parse("bedroom_service", obj, target, actions)["primitive_details"]
            self.assertTrue(all(not detail["category"].startswith("未分类") for detail in details))
            self.assertTrue(all(detail["primitive"] not in {"fluff", "orient", "spread", "smooth", "straighten"} for detail in details))


if __name__ == "__main__":
    unittest.main()
