import hashlib
import unittest

from run_acceptance_check import judge, metric_status


class AcceptanceScopeTest(unittest.TestCase):
    def setUp(self):
        relation_keys = {f"task_{index:03d}::object_{index:03d}" for index in range(139)}
        relation_digest = hashlib.sha256(
            "\n".join(sorted(relation_keys)).encode("utf-8")
        ).hexdigest()
        self.files = [{"exists": True}]
        self.metrics = {
            "instruction_exact_sequence_accuracy": 0.95,
            "instruction_benchmark_tasks": 15,
            "instruction_benchmark_objects": 124,
            "instruction_benchmark_cases": 139,
            "instruction_benchmark_task_ids": {
                "cleaning",
                "organizing",
                "smart_cooking",
                "appliance_management",
                "security_monitoring",
                "laundry",
                "waste_disposal",
                "clothing_care",
                "window_care",
                "bedroom_service",
                "food_serving",
                "object_fetching",
                "elderly_assistance",
                "maintenance_management",
                "entertainment_service",
            },
            "canonical_acceptance_task_ids": {
                "cleaning",
                "organizing",
                "smart_cooking",
                "appliance_management",
                "security_monitoring",
                "laundry",
                "waste_disposal",
                "clothing_care",
                "window_care",
                "bedroom_service",
                "food_serving",
                "object_fetching",
                "elderly_assistance",
                "maintenance_management",
                "entertainment_service",
            },
            "canonical_relation_keys": set(relation_keys),
            "instruction_benchmark_relation_keys": set(relation_keys),
            "instruction_report_relation_keys": set(relation_keys),
            "canonical_relation_digest": relation_digest,
            "instruction_benchmark_relation_digest": relation_digest,
            "instruction_report_relation_digest": relation_digest,
            "instruction_template_overlap_count": 0,
            "acceptance_task_count": 15,
            "cleaning_object_count": 24,
            "unique_object_count": 124,
            "task_object_relation_count": 139,
            "legacy_cleaning_map_ok": True,
            "maintenance_object_count": 10,
            "entertainment_object_count": 8,
            "acceptance_reference_directory_count": 3,
            "acceptance_reference_rows": 38,
            "reference_image_count": 0,
            "reference_video_count": 0,
            "reference_annotation_count": 0,
            "reference_annotation_template_rows": 0,
            "reference_totals_match_source_totals": True,
            "local_data_source": "datasets/midterm_15task_delivery",
            "reference_media_copied": 0,
            "reference_paths_outside_midterm": 0,
            "split_leak_check_applicable": False,
            "split_leak_check_reason": "reference-only views have no split metadata",
            "demo_case_count": 15,
            "visual_error_images": 1,
            "video_joint_accuracy": 0.02,
        }

    def test_instruction_benchmark_can_pass_without_video_joint_ninety_percent(self):
        result = judge(self.files, self.metrics)
        self.assertTrue(result["passed"])
        self.assertTrue(result["instruction_benchmark_check_passed"])

    def test_instruction_benchmark_below_ninety_percent_fails(self):
        self.metrics["instruction_exact_sequence_accuracy"] = 0.89
        result = judge(self.files, self.metrics)
        self.assertFalse(result["passed"])

    def test_repeated_relations_do_not_inflate_unique_objects(self):
        self.metrics["task_object_relation_count"] = 132
        result = judge(self.files, self.metrics)
        self.assertTrue(result["unique_object_scope_passed"])
        self.assertEqual(result["task_object_relation_count"], 132)

    def test_task_hierarchy_requires_all_new_association_counts_and_legacy_map(self):
        for key, bad_value in (
            ("acceptance_task_count", 14),
            ("cleaning_object_count", 23),
            ("legacy_cleaning_map_ok", False),
            ("maintenance_object_count", 9),
            ("entertainment_object_count", 7),
        ):
            with self.subTest(metric=key):
                original = self.metrics[key]
                self.metrics[key] = bad_value
                self.assertFalse(judge(self.files, self.metrics)["task_hierarchy_passed"])
                self.metrics[key] = original

    def test_any_reference_media_count_fails_local_data_integrity(self):
        for key in (
            "reference_image_count",
            "reference_video_count",
            "reference_annotation_count",
            "reference_annotation_template_rows",
        ):
            with self.subTest(metric=key):
                self.metrics[key] = 1
                result = judge(self.files, self.metrics)
                self.assertFalse(result["local_data_integrity_passed"])
                self.assertFalse(result["passed"])
                self.metrics[key] = 0

    def test_outside_midterm_reference_fails_acceptance(self):
        self.metrics["reference_paths_outside_midterm"] = 1
        result = judge(self.files, self.metrics)
        self.assertFalse(result["local_data_integrity_passed"])
        self.assertFalse(result["passed"])

    def test_exact_acceptance_task_id_set_is_required(self):
        self.metrics["instruction_benchmark_task_ids"].remove("cleaning")
        self.metrics["instruction_benchmark_task_ids"].add("floor_cleaning")
        result = judge(self.files, self.metrics)
        self.assertFalse(result["instruction_benchmark_check_passed"])
        self.assertEqual(result["instruction_benchmark_missing_task_ids"], ["cleaning"])
        self.assertEqual(result["instruction_benchmark_unexpected_task_ids"], ["floor_cleaning"])

    def test_120_case_benchmark_with_all_tasks_and_objects_still_fails(self):
        self.metrics["instruction_benchmark_cases"] = 120
        omitted = sorted(self.metrics["instruction_benchmark_relation_keys"])[120:]
        self.metrics["instruction_benchmark_relation_keys"] -= set(omitted)
        result = judge(self.files, self.metrics)
        self.assertFalse(result["instruction_benchmark_check_passed"])
        self.assertEqual(len(result["instruction_benchmark_missing_relation_keys"]), 19)

    def test_reference_directories_and_source_totals_are_required(self):
        for key, bad_value in (
            ("acceptance_reference_directory_count", 2),
            ("acceptance_reference_rows", 37),
            ("reference_totals_match_source_totals", False),
            ("local_data_source", "datasets"),
        ):
            with self.subTest(metric=key):
                original = self.metrics[key]
                self.metrics[key] = bad_value
                self.assertFalse(judge(self.files, self.metrics)["local_data_integrity_passed"])
                self.metrics[key] = original

    def test_canonical_coverage_populates_legacy_report_labels(self):
        metrics = metric_status()
        self.assertEqual(metrics["coverage_total"], 124)
        self.assertEqual(metrics["coverage_tasks"], 15)


if __name__ == "__main__":
    unittest.main()
