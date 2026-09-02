import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

from tools.independent_blind_evaluation import evaluate_blind_cases, wilson_interval
from evaluate_independent_blind_action_test import (
    DEFAULT_BENCHMARK,
    DEFAULT_BUNDLE,
    checkpoint_report_metadata,
    evaluate_cases,
    load_frozen_bundle,
    parse_args,
)


ROOT = Path(__file__).resolve().parents[1]


def frozen_bundle_with_one_case(actions):
    return {
        "format": "independent_blind_action_bundle_v1",
        "source_csv_sha256": "0" * 64,
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "variants_per_relation": 1,
        "audit": {"relations": 1, "cases": 1, "relation_keys": ["food_serving::water_cup"]},
        "cases": [
            {
                "id": "blind_case__v1",
                "source_id": "case",
                "relation_key": "food_serving::water_cup",
                "variant": 1,
                "instruction": "帮我倒水",
                "acceptance_task": "food_serving",
                "legacy_task_id": "food_serving",
                "object": "water_cup",
                "target": "table",
                "actions": actions,
            }
        ],
    }


class IndependentBlindEvaluationTest(unittest.TestCase):
    def test_checkpoint_report_metadata_is_json_serializable_without_weights(self):
        summary = checkpoint_report_metadata(
            {
                "epoch": 10,
                "model_config": {"d_model": 64},
                "model_state_dict": {"weight": torch.tensor([1.0])},
                "optimizer_state_dict": {"state": {0: {"momentum": torch.tensor([1.0])}}},
            }
        )

        self.assertEqual(summary["epoch"], 10)
        self.assertNotIn("model_state_dict", summary)
        self.assertNotIn("optimizer_state_dict", summary)
        json.dumps(summary)

    def test_default_bundle_load_does_not_require_external_csv(self):
        bundle = load_frozen_bundle(DEFAULT_BUNDLE, DEFAULT_BENCHMARK)

        self.assertEqual(len(bundle["cases"]), 414)

    def test_explicit_wrong_csv_still_fails_with_hash_evidence(self):
        bundle = frozen_bundle_with_one_case(["locate(water_cup)"])
        source_bytes = b"original,csv\n"
        bundle["source_csv_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.json"
            csv_path = root / "filled.csv"
            benchmark_path = root / "benchmark.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            csv_path.write_bytes(b"changed,csv\n")
            benchmark_path.write_text(
                json.dumps([{"relation_key": "food_serving::water_cup"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                r"expected=.*; actual=.*; path=",
            ):
                load_frozen_bundle(bundle_path, benchmark_path, csv_path)

    def test_explicit_matching_csv_is_accepted(self):
        bundle = frozen_bundle_with_one_case(["locate(water_cup)"])
        source_bytes = b"original,csv\n"
        bundle["source_csv_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.json"
            csv_path = root / "frozen.csv"
            benchmark_path = root / "benchmark.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            csv_path.write_bytes(source_bytes)
            benchmark_path.write_text(
                json.dumps([{"relation_key": "food_serving::water_cup"}]),
                encoding="utf-8",
            )

            loaded = load_frozen_bundle(bundle_path, benchmark_path, csv_path)

        self.assertEqual(loaded["cases"], bundle["cases"])

    def test_default_loader_rejects_tampered_bundle_relations(self):
        bundle = frozen_bundle_with_one_case(["locate(water_cup)"])
        bundle["audit"]["relation_keys"] = ["tampered::relation"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.json"
            benchmark_path = root / "benchmark.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            benchmark_path.write_text(
                json.dumps([{"relation_key": "food_serving::water_cup"}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "relation coverage"):
                load_frozen_bundle(bundle_path, benchmark_path)

    def test_cli_source_csv_defaults_to_none(self):
        with patch("sys.argv", ["evaluate_independent_blind_action_test.py"]):
            args = parse_args()

        self.assertIsNone(args.source_csv)

    def test_status_distinguishes_mixed_diagnostic_from_human_only_retest(self):
        status = (ROOT / "docs/CURRENT_PROJECT_STATUS_2026-07-04.md").read_text(encoding="utf-8")

        self.assertIn("123种去重物体", status)
        self.assertIn("138条任务—物体操作关系", status)
        self.assertIn("120条测试样例", status)
        self.assertIn("65.22%", status)
        self.assertIn("原始模型", status)
        self.assertIn("最终系统", status)
        self.assertIn("human_only_instruction_retest_template_414.csv", status)

    def test_status_says_natural_presentation_does_not_change_scoring(self):
        status = (ROOT / "docs/CURRENT_PROJECT_STATUS_2026-07-04.md").read_text(encoding="utf-8")

        self.assertIn("自然语言展示不改变解析、原始模型分数或最终系统分数", status)

    def test_runtime_runner_keeps_raw_generation_when_constraint_repairs_it(self):
        bundle = frozen_bundle_with_one_case(
            ["locate(reading_lamp)", "turn_off(reading_lamp)"]
        )

        report = evaluate_cases(
            bundle,
            resolver=lambda instruction: SimpleNamespace(task="appliance_management", object="reading_lamp"),
            generator=lambda resolved: ["locate(reading_lamp)", "turn_on(reading_lamp)"],
            constraint=lambda resolved, raw: ["locate(reading_lamp)", "turn_off(reading_lamp)"],
        )

        case = report["cases"][0]
        self.assertEqual(case["raw_actions"][-1], "turn_on(reading_lamp)")
        self.assertEqual(case["final_actions"][-1], "turn_off(reading_lamp)")
        self.assertTrue(case["constraint_applied"])

    def test_evaluator_reports_raw_failure_and_final_success_separately(self):
        bundle = frozen_bundle_with_one_case(["locate(water_cup)", "pour_water(water_cup)"])
        report = evaluate_blind_cases(
            bundle,
            raw_predictions={"blind_case__v1": ["locate(water_cup)", "turn_on(water_cup)"]},
            final_predictions={"blind_case__v1": ["locate(water_cup)", "pour_water(water_cup)"]},
        )

        self.assertEqual(report["metrics"]["raw_exact_sequence_accuracy"], 0.0)
        self.assertEqual(report["metrics"]["final_exact_sequence_accuracy"], 1.0)

    def test_wilson_interval_is_bounded_and_contains_observed_rate(self):
        lower, upper = wilson_interval(9, 10)

        self.assertLess(lower, 0.9)
        self.assertGreater(upper, 0.9)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)


if __name__ == "__main__":
    unittest.main()
