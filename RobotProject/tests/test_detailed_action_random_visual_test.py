import tempfile
import threading
import unittest
import inspect
from pathlib import Path

from detailed_action_random_visual_test import (
    RunConfig,
    failure_reason_zh,
    format_failure_counts,
    resolve_device,
    result_detail,
    run_batch,
    run_cli,
)
from tools.detailed_action_random_cases import RandomDetailedActionCase


class FakeParser:
    def __init__(self):
        self.calls = []

    def resolve(self, instruction):
        self.calls.append(instruction)
        label = "scene:MDC-001" if instruction == "scene" else "object:vla:vla_080"
        result_type = "coordination_detailed_action" if instruction == "scene" else "object_detailed_action"
        if "牙膏" in instruction:
            label = "object:vla:vla_081"
        result = {"result_type": result_type, "label": label, "detailed_actions": ["动作"]}
        if instruction == "scene":
            result["objects"] = [
                {"object_label": "object:vla:vla_080", "object_name_zh": "牙刷", "detailed_actions": ["动作1"]},
                {"object_label": "object:vla:vla_081", "object_name_zh": "牙膏", "detailed_actions": ["动作2"]},
            ]
        return result


class FakeVar:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class FakeTree:
    def __init__(self, selected):
        self.selected = selected

    def selection(self):
        return (self.selected,)


def fake_cases(count, seed):
    rows = [
        RandomDetailedActionCase("RND-00001", "object", "object", "object:vla:vla_080", "牙刷"),
        RandomDetailedActionCase("RND-00002", "scene", "scene", "scene:MDC-001", "地面清洁"),
    ]
    return rows[:count]


class DetailedActionRandomVisualTest(unittest.TestCase):
    def test_tree_selection_uses_stable_item_id_to_show_full_row(self):
        from detailed_action_random_visual_test import RandomVisualApp

        selected = {
            "case_id": "RND-00002",
            "case_type": "object",
            "instruction": "第二条命令",
            "expected_label": "object:vla:vla_080",
            "actual_result_type": "object_detailed_action",
            "actual_label": "object:vla:vla_080",
            "resolution_backend": "bge",
            "elapsed_ms": 9.25,
            "passed": True,
            "detailed_actions": ["动作甲", "动作乙"],
            "failure_reason": "",
            "clarification_reason": "",
            "error": "",
        }
        app = object.__new__(RandomVisualApp)
        app.tree = FakeTree("RND-00002")
        app.row_by_iid = {"RND-00002": selected}
        app.current_var = FakeVar()
        app.expected_var = FakeVar()
        app.actual_var = FakeVar()
        app.actions_var = FakeVar()
        app.failure_detail_var = FakeVar()
        app.case_time_var = FakeVar()
        app._on_tree_select()
        self.assertEqual(app.current_var.value, "第二条命令")
        self.assertEqual(app.actions_var.value, "动作甲 → 动作乙")
        self.assertEqual(app.failure_detail_var.value, "-")

    def test_failed_tree_selection_shows_exact_reason_and_marks_candidate_actions(self):
        from detailed_action_random_visual_test import RandomVisualApp

        selected = {
            "case_id": "RND-00009",
            "case_type": "object",
            "instruction": "失败命令",
            "expected_label": "object:vla:vla_080",
            "actual_result_type": "clarification_required",
            "actual_label": "",
            "resolution_backend": "bge",
            "elapsed_ms": 1.5,
            "passed": False,
            "detailed_actions": ["候选动作"],
            "failure_reason": "result_type_mismatch",
            "clarification_reason": "BGE 执行门控未达到校准阈值",
            "error": "",
        }
        app = object.__new__(RandomVisualApp)
        app.tree = FakeTree("RND-00009")
        app.row_by_iid = {"RND-00009": selected}
        app.current_var = FakeVar()
        app.expected_var = FakeVar()
        app.actual_var = FakeVar()
        app.actions_var = FakeVar()
        app.failure_detail_var = FakeVar()
        app.case_time_var = FakeVar()
        app._on_tree_select()
        self.assertEqual(app.failure_detail_var.value, "BGE 执行门控未达到校准阈值")
        self.assertEqual(app.actions_var.value, "非最终采用：候选动作")

    def test_gui_build_removes_acceptance_card_and_adds_failure_sections(self):
        from detailed_action_random_visual_test import RandomVisualApp

        source = inspect.getsource(RandomVisualApp._build)
        self.assertNotIn('(\"验收\", \"acceptance\")', source)
        self.assertIn("失败原因统计", source)
        self.assertIn("失败原因", source)
        self.assertIn("联动总平均耗时", source)
        self.assertIn("场景解析平均耗时", source)
        self.assertIn("子操作平均耗时", source)
        self.assertNotIn('("联动平均耗时", "scene_avg")', source)
        self.assertNotIn("acceptance']['status", inspect.getsource(run_cli))

    def test_failure_reason_prefers_specific_clarification_and_counts_categories(self):
        rows = [
            {
                "passed": False,
                "clarification_reason": "BGE 执行门控未达到校准阈值",
                "failure_reason": "result_type_mismatch",
                "error": "",
            },
            {
                "passed": False,
                "clarification_reason": "",
                "failure_reason": "label_mismatch",
                "error": "",
            },
            {
                "passed": True,
                "clarification_reason": "",
                "failure_reason": "",
                "error": "",
            },
        ]
        self.assertEqual(failure_reason_zh(rows[0]), "执行门控未通过")
        self.assertEqual(
            format_failure_counts(rows),
            "执行门控未通过 1 条｜标签不匹配 1 条",
        )
        self.assertEqual(format_failure_counts([rows[2]]), "无")

    def test_result_detail_shows_actions_for_pass_and_reason_for_failure(self):
        common = {
            "case_id": "RND-00001",
            "case_type": "object",
            "instruction": "请把牙刷放好",
            "expected_label": "object:vla:vla_080",
            "actual_result_type": "object_detailed_action",
            "actual_label": "object:vla:vla_080",
            "resolution_backend": "bge",
            "elapsed_ms": 12.5,
            "error": "",
        }
        passed = {
            **common,
            "passed": True,
            "detailed_actions": ["步骤一", "步骤二"],
            "failure_reason": "",
            "clarification_reason": "",
        }
        failed = {
            **common,
            "passed": False,
            "detailed_actions": [],
            "failure_reason": "label_mismatch",
            "clarification_reason": "",
        }
        self.assertEqual(result_detail(passed)["actions"], "步骤一 → 步骤二")
        self.assertEqual(result_detail(passed)["failure"], "-")
        self.assertEqual(result_detail(failed)["failure"], "标签不匹配")
        self.assertEqual(result_detail(failed)["actions"], "未输出细分动作")

    def test_scene_result_detail_lists_every_child_command_time_and_actions(self):
        row = {
            "case_type": "scene", "instruction": "我要睡觉了", "expected_label": "scene:MDC-001",
            "actual_result_type": "coordination_detailed_action", "actual_label": "scene:MDC-001",
            "resolution_backend": "bge", "elapsed_ms": 30.0, "passed": True,
            "detailed_actions": ["父场景动作"], "failure_reason": "", "clarification_reason": "", "error": "",
            "scene_parse_elapsed_ms": 10.0, "child_operation_total_elapsed_ms": 20.0,
            "child_operations": [
                {"index": 1, "instruction": "把空调调好", "expected_label": "object:air_conditioner",
                 "actual_label": "object:air_conditioner", "passed": True, "elapsed_ms": 8.0,
                 "detailed_actions": ["设定温度", "切换睡眠模式"], "failure_reason": "", "error": ""},
                {"index": 2, "instruction": "把卧室灯关了", "expected_label": "object:bedroom_light",
                 "actual_label": "object:bedroom_light", "passed": True, "elapsed_ms": 12.0,
                 "detailed_actions": ["关闭灯光"], "failure_reason": "", "error": ""},
            ],
        }
        detail = result_detail(row)
        self.assertIn("把空调调好", detail["actions"])
        self.assertIn("8.000000 ms", detail["actions"])
        self.assertIn("设定温度 → 切换睡眠模式", detail["actions"])
        self.assertIn("把卧室灯关了", detail["actions"])
        self.assertIn("关闭灯光", detail["actions"])
        self.assertIn("父场景解析 10.000000 ms", detail["elapsed"])
        self.assertIn("联动总计 30.000000 ms", detail["elapsed"])

    def test_auto_device_prefers_cuda_and_explicit_choices_are_preserved(self):
        self.assertEqual(resolve_device("auto", cuda_available=True), "cuda")
        self.assertEqual(resolve_device("auto", cuda_available=False), "cpu")
        self.assertEqual(resolve_device("cpu", cuda_available=True), "cpu")
        self.assertEqual(resolve_device("cuda", cuda_available=True), "cuda")

    def test_explicit_cuda_fails_clearly_when_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            resolve_device("cuda", cuda_available=False)

    def test_batch_emits_results_summary_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(2, 9, "cpu", Path("checkpoint.pt"), Path("encoder"), Path(directory))
            events = []
            parser = FakeParser()
            summary, run_dir = run_batch(
                config, events.append, threading.Event(), parser_factory=lambda **_: parser,
                case_generator=fake_cases,
            )
            self.assertEqual(summary["overall"]["accuracy_percent"], 100.0)
            self.assertTrue(summary["acceptance"]["passed"])
            self.assertEqual(len([event for event in events if event["type"] == "result"]), 2)
            self.assertEqual(len([event for event in events if event["type"] == "warmup"]), 1)
            self.assertNotEqual(parser.calls[0], "object")
            self.assertEqual(parser.calls[1], "object")
            self.assertEqual(len(parser.calls), 5)
            self.assertTrue((run_dir / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
