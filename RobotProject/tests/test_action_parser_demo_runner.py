import unittest

from tools.action_parser_demo_cases import DemoCase
from tools.action_parser_demo_runner import (
    DemoResult,
    classify_failure,
    summarize_results,
)
from tools.manual_instruction_entry import resolve_manual_instruction


def make_case(**overrides):
    values = {
        "case_id": "CASE-0001",
        "source": "legacy",
        "instruction": "帮我把客厅里的杯子洗一下",
        "expected_task": "dish_washing",
        "expected_object": "cup",
        "expected_operation": "wash",
        "expected_target": "sink",
        "expected_actions": ("locate(cup)", "wash(cup)", "inspect(cup)"),
        "template_id": "example",
        "seed": 20260813,
    }
    values.update(overrides)
    return DemoCase(**values)


class ActionParserDemoRunnerTest(unittest.TestCase):
    def test_cleaning_keyword_resolves_to_clean_operation(self):
        resolved = resolve_manual_instruction("做一下清洁，清洁地毯")
        self.assertEqual(resolved.task, "cleaning")
        self.assertEqual(resolved.object, "carpet")
        self.assertEqual(resolved.operation, "clean")

    def test_classifies_each_failure_stage(self):
        case = make_case()
        base = {
            "task": case.expected_task,
            "object": case.expected_object,
            "operation": case.expected_operation,
            "target": case.expected_target,
            "actions": list(case.expected_actions),
        }
        self.assertIsNone(classify_failure(case, base))
        self.assertEqual(classify_failure(case, None, error_kind="match_failure"), "match_failure")
        self.assertEqual(classify_failure(case, None, error_kind="object_not_found"), "object_not_found")
        self.assertEqual(classify_failure(case, {**base, "task": "cleaning"}), "task_mismatch")
        self.assertEqual(classify_failure(case, {**base, "object": "plate"}), "object_mismatch")
        self.assertEqual(classify_failure(case, {**base, "operation": "clean"}), "operation_mismatch")
        self.assertEqual(classify_failure(case, {**base, "target": "table"}), "target_mismatch")
        self.assertEqual(classify_failure(case, {**base, "prediction_error": "invalid"}), "model_generation_error")
        self.assertEqual(classify_failure(case, {**base, "actions": ["locate(cup)"]}), "action_sequence_mismatch")

    def test_summary_counts_pass_fail_categories_and_sources(self):
        legacy = make_case()
        vla = make_case(case_id="CASE-0002", source="vla", expected_target=None)
        results = [
            DemoResult.from_outcome(legacy, actual={}, passed=True, elapsed_seconds=0.1),
            DemoResult.from_outcome(
                vla,
                actual={},
                passed=False,
                failure_category="object_not_found",
                error="no template",
                elapsed_seconds=0.2,
            ),
        ]
        summary = summarize_results(results, requested=2, seed=7)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["failure_categories"]["object_not_found"], 1)
        self.assertEqual(summary["sources"]["legacy"]["passed"], 1)
        self.assertEqual(summary["sources"]["vla"]["failed"], 1)
        self.assertEqual(summary["accuracy_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
