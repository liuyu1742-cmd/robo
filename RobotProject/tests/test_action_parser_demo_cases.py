import unittest

from tools.action_parser_demo_cases import _naturalize_vla_goal, generate_demo_cases
from tools.action_parser_command_quality import command_quality_violations
from tools.manual_instruction_entry import load_benchmark_records, resolve_manual_instruction
from tools.vla_instruction_planner import plan_vla_instruction


class ActionParserDemoCasesTest(unittest.TestCase):
    def test_same_seed_generates_same_unique_500_cases(self):
        first = generate_demo_cases(500, 20260813)
        second = generate_demo_cases(500, 20260813)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 500)
        self.assertEqual(len({case.instruction for case in first}), 500)
        self.assertEqual(first[0].case_id, "CASE-0001")
        self.assertEqual(first[-1].case_id, "CASE-0500")

    def test_generator_covers_legacy_and_vla_tasks(self):
        cases = generate_demo_cases(500, 20260813)

        self.assertEqual({case.source for case in cases}, {"legacy", "vla"})
        legacy_tasks = {
            case.expected_task for case in cases if case.source == "legacy"
        }
        vla_tasks = {
            case.expected_task for case in cases if case.source == "vla"
        }
        self.assertGreaterEqual(len(legacy_tasks), 15)
        self.assertGreaterEqual(len(vla_tasks), 10)
        self.assertTrue(all(case.expected_actions for case in cases))

    def test_different_seed_changes_generated_instructions(self):
        first = generate_demo_cases(100, 20260813)
        second = generate_demo_cases(100, 20260814)

        self.assertNotEqual(
            [case.instruction for case in first],
            [case.instruction for case in second],
        )

    def test_invalid_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count"):
            generate_demo_cases(0, 20260813)

    def test_generated_commands_reach_at_least_90_percent_semantic_match(self):
        records = load_benchmark_records()
        passed = 0
        for case in generate_demo_cases(500, 20260813):
            vla = plan_vla_instruction(case.instruction)
            if vla is not None:
                actual = vla["resolved"]
                matches = (
                    actual["task"] == case.expected_task
                    and actual["object"] == case.expected_object
                    and actual["operation"] == case.expected_operation
                )
            else:
                try:
                    actual = resolve_manual_instruction(
                        case.instruction, benchmark_records=records
                    )
                except ValueError:
                    matches = False
                else:
                    expected_target = (
                        None if case.expected_target in (None, "none")
                        else case.expected_target
                    )
                    actual_target = None if actual.target in (None, "none") else actual.target
                    matches = (
                        actual.task == case.expected_task
                        and actual.object == case.expected_object
                        and actual.operation == case.expected_operation
                        and actual_target == expected_target
                    )
            passed += int(matches)
        self.assertGreaterEqual(passed, 450)

    def test_commands_do_not_expose_task_or_scene_category_prompts(self):
        cases = generate_demo_cases(500, 20260813)
        forbidden = (
            "场景中", "做一下清洁，", "整理收纳一下，", "做饭的时候，",
            "帮我管一下家电，", "做一下安全检查，", "清洗衣物时，",
            "处理垃圾时，", "护理衣物时，", "处理门窗时，", "整理卧室时，",
            "送餐服务时，", "帮我取物，", "照顾老人时，", "做维护保养时，",
            "提供娱乐服务时，",
        )
        for case in cases:
            self.assertFalse(
                any(text in case.instruction for text in forbidden), case.instruction
            )

    def test_all_500_commands_pass_natural_language_quality_gate(self):
        cases = generate_demo_cases(500, 20260813)
        failures = [
            (case.case_id, case.instruction, violations)
            for case in cases
            if (
                violations := command_quality_violations(
                    case.instruction,
                    task=case.expected_task,
                    object_id=case.expected_object,
                    operation=case.expected_operation,
                )
            )
        ]
        self.assertEqual(failures, [])

    def test_laundry_commands_move_items_to_washing_machine(self):
        cases = [
            case for case in generate_demo_cases(500, 20260813)
            if case.source == "legacy" and case.expected_task == "laundry"
        ]
        self.assertTrue(cases)
        self.assertTrue(all("洗衣机" in case.instruction for case in cases))

    def test_bedding_commands_use_bedroom_locations_only(self):
        bedding = {
            "bed", "mattress", "quilt", "blanket", "pillow", "pillowcase",
            "bed_sheet", "duvet_cover",
        }
        cases = [
            case for case in generate_demo_cases(500, 20260813)
            if case.expected_object in bedding
        ]
        invalid = ("桌面", "书桌", "餐桌", "厨房", "门口", "走廊")
        self.assertTrue(cases)
        self.assertFalse(
            any(term in case.instruction for case in cases for term in invalid)
        )

    def test_vla_arrow_path_becomes_direct_request(self):
        value = _naturalize_vla_goal({
            "object_id": "vla_050",
            "object_name": "咖啡杯",
            "source_operation_label": "橱柜→台面",
        })
        self.assertEqual(value, "把橱柜里的咖啡杯拿到台面上")

    def test_complex_onion_instruction_is_natural(self):
        value = _naturalize_vla_goal({
            "object_id": "vla_006",
            "object_name": "削皮刀",
            "source_operation_label": "洋葱切丁：水槽→砧板→碗，刀具与砧板回放水槽",
        })
        self.assertIn("用削皮刀把洋葱切成丁", value)
        self.assertIn("刀和砧板用完放回水槽", value)
        self.assertNotIn("按", value)
        self.assertNotIn("具体是", value)


if __name__ == "__main__":
    unittest.main()
