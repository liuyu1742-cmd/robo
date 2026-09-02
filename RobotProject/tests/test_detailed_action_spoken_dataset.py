import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tools.detailed_action_catalog import build_detailed_action_catalog


class DetailedActionSpokenDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_detailed_action_catalog()

    def make_dataset(self):
        entries = self.catalog["entries"]
        records = {"train": [], "dev": [], "independent_test": []}
        quotas = {"train": 10, "dev": 3, "independent_test": 3}
        split_offsets = {"train": 0, "dev": 10, "independent_test": 13}
        spoken_identifiers = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳"
        for label_number, entry in enumerate(entries):
            for split, quota in quotas.items():
                for index in range(quota):
                    spoken_identifier = spoken_identifiers[split_offsets[split] + index]
                    if entry["entry_type"] == "scene":
                        text = f"请协调物品编号{spoken_identifier}{label_number:04d}处理好"
                    else:
                        text = f"请把物品编号{spoken_identifier}{label_number:04d}处理好"
                    records[split].append(
                        {
                            "sample_id": f"{split}-positive-{label_number:03d}-{index:02d}",
                            "text": text,
                            "label_id": entry["label"],
                            "result_type": (
                                "scene_coordination"
                                if entry["entry_type"] == "scene"
                                else "object_detailed_action"
                            ),
                            "annotation_source": f"independent_annotator_{split}",
                            "object_mentions": ["物品"],
                            "operation_evidence": ["处理"],
                            "safety_class": None,
                            "expected_executable": True,
                        }
                    )

        safety_classes = [
            "cancel_negation",
            "restrictive_negation_or_in_word_boundary",
            "information_question",
            "completed_statement",
            "same_name_object_ambiguity",
            "object_operation_incompatibility",
            "out_of_domain",
            "low_confidence_near_candidate",
            "multi_clause_turn_correction",
        ]
        safety_quotas = {"train": 60, "dev": 20, "independent_test": 20}
        executable_safety_quotas = {
            "cancel_negation": {"train": 0, "dev": 0, "independent_test": 0},
            "restrictive_negation_or_in_word_boundary": {"train": 30, "dev": 10, "independent_test": 10},
            "information_question": {"train": 0, "dev": 0, "independent_test": 0},
            "completed_statement": {"train": 0, "dev": 0, "independent_test": 0},
            "same_name_object_ambiguity": {"train": 0, "dev": 0, "independent_test": 0},
            "object_operation_incompatibility": {"train": 0, "dev": 0, "independent_test": 0},
            "out_of_domain": {"train": 0, "dev": 0, "independent_test": 0},
            "low_confidence_near_candidate": {"train": 0, "dev": 0, "independent_test": 0},
            "multi_clause_turn_correction": {"train": 30, "dev": 10, "independent_test": 10},
        }
        split_markers = {"train": "天", "dev": "地", "independent_test": "玄"}
        for split, quota in safety_quotas.items():
            for class_number, safety_class in enumerate(safety_classes):
                for index in range(quota):
                    executable = index < executable_safety_quotas[safety_class][split]
                    entry = self.catalog["entries"][(class_number * 100 + index) % len(self.catalog["entries"])]
                    identifier = f"{split_markers[split]}{spoken_identifiers[class_number]}{index:02d}"
                    text = (
                        f"请协调物品编号{identifier}处理好"
                        if entry["entry_type"] == "scene"
                        else f"请把物品编号{identifier}处理好"
                    )
                    records[split].append(
                        {
                            "sample_id": f"{split}-safety-{safety_class}-{index:02d}",
                            "text": text if executable else f"安全{split}{safety_class}{index:02d}",
                            "label_id": entry["label"] if executable else None,
                            "result_type": (
                                "scene_coordination"
                                if executable and entry["entry_type"] == "scene"
                                else "object_detailed_action"
                                if executable
                                else "clarification_required"
                            ),
                            "annotation_source": f"independent_annotator_{split}",
                            "object_mentions": ["物品"] if executable else [],
                            "operation_evidence": ["处理"] if executable else [],
                            "safety_class": safety_class,
                            "expected_executable": executable,
                        }
                    )
        return records

    def test_complete_schema_and_required_counts_are_valid(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()

        self.assertEqual(validate_dataset(**records, catalog=self.catalog), [])

    def test_executable_safety_is_counted_separately_from_core_label_quota(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()
        core_positive_count = sum(
            1
            for split_records in records.values()
            for record in split_records
            if record["label_id"] is not None and record["safety_class"] is None
        )
        executable_safety_count = sum(
            1
            for split_records in records.values()
            for record in split_records
            if record["expected_executable"] and record["safety_class"] is not None
        )

        self.assertEqual(core_positive_count, 4064)
        self.assertEqual(executable_safety_count, 100)
        self.assertEqual(validate_dataset(**records, catalog=self.catalog), [])

    def test_schema_catalog_source_and_positive_evidence_errors_name_id_and_split(self):
        from tools.detailed_action_spoken_dataset import validate_split

        records = self.make_dataset()["train"]
        broken = deepcopy(records[0])
        broken.pop("expected_executable")
        broken["label_id"] = "object:original:not-in-catalog"
        broken["annotation_source"] = "independent_annotator_dev"
        broken["operation_evidence"] = []
        records[0] = broken

        errors = validate_split(records, "train", self.catalog)

        self.assertTrue(errors)
        self.assertTrue(all("train" in error and broken["sample_id"] in error for error in errors))
        self.assertTrue(any("schema" in error for error in errors))
        self.assertTrue(any("catalog" in error for error in errors))
        self.assertTrue(any("annotation_source" in error for error in errors))
        self.assertTrue(any("operation_evidence" in error for error in errors))

    def test_cross_split_text_skeleton_and_long_ngram_leaks_are_rejected(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()
        label = records["train"][0]["label_id"]
        train = records["train"][0]
        dev = next(item for item in records["dev"] if item["label_id"] == label)
        train.update(text="请把杯子放到桌上", object_mentions=["杯子"])
        dev.update(text="请把盘子放到桌上", object_mentions=["盘子"])
        train_ngram = records["train"][10]
        dev_ngram = next(
            item for item in records["dev"] if item["label_id"] == train_ngram["label_id"]
        )
        train_ngram.update(text="甲特别独特的操作短语乙", object_mentions=["甲"])
        dev_ngram.update(text="丙特别独特的操作短语丁", object_mentions=["丙"])

        errors = validate_dataset(**records, catalog=self.catalog)

        self.assertTrue(any("expression skeleton" in error for error in errors))
        self.assertTrue(any("long n-gram" in error for error in errors))

    def test_rejected_record_cannot_leak_an_action(self):
        from tools.detailed_action_spoken_dataset import validate_split

        records = self.make_dataset()["train"]
        rejected = next(item for item in records if item["label_id"] is None)
        rejected["operation_evidence"] = ["打开"]

        errors = validate_split(records, "train", self.catalog)

        self.assertTrue(any("rejected record" in error and "operation_evidence" in error for error in errors))

    def test_safety_class_requires_approved_executable_rejection_mix(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()
        for record in records["train"]:
            if record["safety_class"] is not None and record["expected_executable"]:
                record.update(
                    label_id=None,
                    result_type="clarification_required",
                    object_mentions=[],
                    operation_evidence=[],
                    expected_executable=False,
                )

        errors = validate_dataset(**records, catalog=self.catalog)

        self.assertTrue(any("split=train" in error and "restrictive_negation_or_in_word_boundary" in error and "executable" in error for error in errors))
        self.assertTrue(any("split=train" in error and "multi_clause_turn_correction" in error and "executable" in error for error in errors))

    def test_positive_evidence_must_be_grounded_in_normalized_text(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()
        mention_failure = records["train"][0]
        evidence_failure = records["train"][1]
        mention_failure["object_mentions"] = ["冰箱"]
        evidence_failure["operation_evidence"] = ["开关"]

        errors = validate_dataset(**records, catalog=self.catalog)

        self.assertTrue(any(mention_failure["sample_id"] in error and "object_mentions" in error for error in errors))
        self.assertTrue(any(evidence_failure["sample_id"] in error and "operation_evidence" in error for error in errors))

    def test_non_list_top_level_inputs_return_errors_instead_of_raising(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()
        for invalid_train in (None, {"records": records["train"]}):
            with self.subTest(invalid_train=type(invalid_train).__name__):
                errors = validate_dataset(
                    invalid_train,
                    records["dev"],
                    records["independent_test"],
                    self.catalog,
                )

                self.assertIsInstance(errors, list)
                self.assertTrue(any("split=train" in error and "records must be a list" in error for error in errors))

    def test_malformed_schema_is_reported_instead_of_crashing_dataset_validation(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()
        broken = records["train"][0]
        broken["object_mentions"] = None

        errors = validate_dataset(**records, catalog=self.catalog)

        self.assertTrue(any("object_mentions" in error and broken["sample_id"] in error for error in errors))

    def test_executable_safety_boundary_sample_keeps_a_real_label(self):
        from tools.detailed_action_spoken_dataset import validate_dataset

        records = self.make_dataset()
        boundary = next(
            record
            for record in records["train"]
            if record["safety_class"] is not None and record["expected_executable"]
        )

        self.assertIsNotNone(boundary["label_id"])
        self.assertNotEqual(boundary["result_type"], "clarification_required")
        self.assertIn(boundary["object_mentions"][0], boundary["text"])
        self.assertIn(boundary["operation_evidence"][0], boundary["text"])
        self.assertEqual(validate_dataset(**records, catalog=self.catalog), [])

    def test_writer_writes_verified_json_and_manifest_sha256(self):
        from tools.detailed_action_spoken_dataset import write_dataset

        records = self.make_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "dataset"
            manifest = write_dataset(**records, output_dir=output_dir)
            manifest_path = output_dir / "manifest.json"

            self.assertEqual(manifest, json.loads(manifest_path.read_text(encoding="utf-8")))
            for split in ("train", "dev", "independent_test"):
                path = output_dir / f"{split}.json"
                self.assertTrue(path.is_file())
                self.assertEqual(
                    manifest["splits"][split]["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )

    def test_cli_validates_a_split_file(self):
        records = self.make_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "train.json"
            path.write_text(json.dumps(records["train"], ensure_ascii=False), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.detailed_action_spoken_dataset",
                    "--validate-split",
                    "train",
                    str(path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("valid", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
