import unittest

from tools.object_operation_augmentation import build_ontology_augmentation_rows
from tools.calibrate_object_operation_specialist import select_safe_candidate


class ObjectOperationAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {
                "label_id": "object:test:carpet",
                "object_name_zh": "地毯",
                "operation_phrases": ["吸尘地毯", "点洗地毯"],
                "forbidden_actions": [
                    "不得把地毯当作通电设备启动、调档或读取运行程序",
                    "不得把地毯当作容器执行清空、装载或封口",
                ],
            }
        ]

    def test_builds_affirmative_negative_requests_without_negation_leakage(self):
        rows = build_ontology_augmentation_rows(self.entries, templates_per_phrase=2)
        negatives = [row for row in rows if not row["expected_executable"]]

        self.assertEqual(len(negatives), 4)
        self.assertTrue(all(row["safety_class"] == "object_operation_incompatibility" for row in negatives))
        self.assertTrue(all(row["object_mentions"] == ["地毯"] for row in negatives))
        self.assertTrue(all("地毯" in row["text"] for row in negatives))
        self.assertTrue(all(not any(word in row["text"] for word in ("不得", "禁止", "不能", "不要")) for row in negatives))

    def test_builds_positive_protection_rows_with_original_label(self):
        rows = build_ontology_augmentation_rows(self.entries, templates_per_phrase=1)
        positives = [row for row in rows if row["expected_executable"]]

        self.assertEqual(len(positives), 2)
        self.assertTrue(all(row["label_id"] == "object:test:carpet" for row in positives))
        self.assertTrue(all(row["safety_class"] is None for row in positives))

    def test_generation_is_deterministic_and_ids_are_unique(self):
        first = build_ontology_augmentation_rows(self.entries, templates_per_phrase=3)
        second = build_ontology_augmentation_rows(self.entries, templates_per_phrase=3)

        self.assertEqual(first, second)
        self.assertEqual(len({row["sample_id"] for row in first}), len(first))

    def test_calibration_prefers_lowest_class_without_regressing_hard_metrics(self):
        reports = [
            {"threshold": -4.0, "gate": 0.951, "joint": 0.870, "core": 0.961, "incompat": 0.70},
            {"threshold": -3.0, "gate": 0.9502, "joint": 0.869, "core": 0.961, "incompat": 0.80},
            {"threshold": -2.0, "gate": 0.949, "joint": 0.871, "core": 0.958, "incompat": 0.90},
        ]
        selected = select_safe_candidate(reports, minimum_gate=0.9501, minimum_core=0.9606)
        self.assertEqual(selected["threshold"], -3.0)


if __name__ == "__main__":
    unittest.main()
