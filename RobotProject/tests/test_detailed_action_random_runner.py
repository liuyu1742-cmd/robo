import unittest

from tools.detailed_action_random_cases import RandomDetailedActionCase
from tools.detailed_action_random_runner import run_random_cases, summarize_random_results


class FakeParser:
    def __init__(self):
        self.calls = []

    def resolve(self, instruction):
        self.calls.append(instruction)
        if instruction == "boom":
            raise RuntimeError("broken")
        if instruction.startswith("scene"):
            return {
                "result_type": "coordination_detailed_action",
                "label": "scene:MDC-001",
                "objects": [
                    {"object_label": "object:vla:vla_080", "object_name_zh": "牙刷", "detailed_actions": ["父动作1"]},
                    {"object_label": "object:vla:vla_081", "object_name_zh": "牙膏", "detailed_actions": ["父动作2"]},
                ],
            }
        label = "object:vla:vla_081" if "081" in instruction else "object:vla:vla_080"
        return {"result_type": "object_detailed_action", "label": label, "detailed_actions": ["动作"]}


class FakeClock:
    def __init__(self, values=(100, 2100, 3000, 8000, 9000, 12000)):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class DetailedActionRandomRunnerTest(unittest.TestCase):
    @staticmethod
    def child_factory(label, seed):
        suffix = "081" if label.endswith("081") else "080"
        return RandomDetailedActionCase(
            f"CHILD-{seed}", "object", f"child-{suffix}", label, "子物体"
        )

    def test_records_each_timing_and_category_averages(self):
        cases = [
            RandomDetailedActionCase("RND-0001", "object", "object one", "object:vla:vla_080", "牙刷"),
            RandomDetailedActionCase("RND-0002", "scene", "scene one", "scene:MDC-001", "地面清洁"),
        ]
        clock = FakeClock((100, 2100, 3000, 8000, 9000, 12000, 13000, 17000))
        results = run_random_cases(
            FakeParser(), cases, clock_ns=clock, child_case_factory=self.child_factory
        )
        self.assertEqual([row.elapsed_ns for row in results], [2000, 12000])
        self.assertTrue(all(row.passed for row in results))
        self.assertEqual(results[0].resolution_backend, "")
        self.assertEqual(results[1].scene_parse_elapsed_ns, 5000)
        self.assertEqual(results[1].child_operation_total_elapsed_ns, 7000)
        self.assertEqual(results[1].child_operation_count, 2)
        self.assertEqual([child.elapsed_ns for child in results[1].child_operations], [3000, 4000])
        self.assertEqual(
            [(child.started_perf_counter_ns, child.ended_perf_counter_ns) for child in results[1].child_operations],
            [(9000, 12000), (13000, 17000)],
        )
        self.assertEqual(results[1].ended_perf_counter_ns, 17000)
        summary = summarize_random_results(results, requested=2, seed=3)
        self.assertEqual(summary["object"]["average_elapsed_ns"], 2000)
        self.assertEqual(summary["scene"]["average_elapsed_ns"], 12000)
        self.assertEqual(summary["scene"]["average_scene_parse_elapsed_ms"], 0.005)
        self.assertEqual(summary["scene"]["average_child_operation_elapsed_ms"], 0.0035)
        self.assertEqual(summary["scene"]["average_child_total_elapsed_ms"], 0.007)
        self.assertEqual(summary["overall"]["average_elapsed_ns"], 7000)
        self.assertTrue(summary["acceptance"]["passed"])

    def test_child_failure_propagates_but_remaining_children_still_run(self):
        class OneBadChildParser(FakeParser):
            def resolve(self, instruction):
                result = super().resolve(instruction)
                if instruction == "child-080":
                    return {"result_type": "object_detailed_action", "label": "object:wrong"}
                return result

        parser = OneBadChildParser()
        case = RandomDetailedActionCase("RND-0001", "scene", "scene one", "scene:MDC-001", "场景")
        result = run_random_cases(
            parser,
            [case],
            clock_ns=FakeClock((0, 10, 20, 50, 60, 100)),
            child_case_factory=self.child_factory,
        )[0]

        self.assertFalse(result.passed)
        self.assertEqual(result.failure_reason, "child_operation_failed")
        self.assertEqual(len(result.child_operations), 2)
        self.assertEqual([child.passed for child in result.child_operations], [False, True])
        self.assertEqual(parser.calls, ["scene one", "child-080", "child-081"])

    def test_scene_requires_at_least_two_objects(self):
        class OneObjectParser(FakeParser):
            def resolve(self, instruction):
                if instruction.startswith("scene"):
                    return {
                        "result_type": "coordination_detailed_action",
                        "label": "scene:MDC-001",
                        "objects": [{"object_label": "object:vla:vla_080"}],
                    }
                return super().resolve(instruction)

        case = RandomDetailedActionCase("RND-0001", "scene", "scene one", "scene:MDC-001", "场景")
        result = run_random_cases(OneObjectParser(), [case], clock_ns=FakeClock((0, 10)))[0]
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_reason, "coordination_requires_multiple_objects")
        self.assertEqual(result.child_operation_count, 0)

    def test_wrong_type_and_exception_fail_without_stopping_batch(self):
        cases = [
            RandomDetailedActionCase("RND-0001", "scene", "object one", "scene:MDC-001", "场景"),
            RandomDetailedActionCase("RND-0002", "object", "boom", "object:vla:vla_080", "牙刷"),
        ]
        results = run_random_cases(FakeParser(), cases, clock_ns=FakeClock())
        self.assertFalse(results[0].passed)
        self.assertEqual(results[0].failure_reason, "result_type_mismatch")
        self.assertEqual(results[0].clarification_reason, "")
        self.assertFalse(results[1].passed)
        self.assertIn("RuntimeError", results[1].error)


if __name__ == "__main__":
    unittest.main()
