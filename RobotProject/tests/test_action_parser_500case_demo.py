import tempfile
import threading
import unittest
import inspect
from pathlib import Path

from action_parser_500case_demo import DemoApp, RunConfig, parse_args, run_batch
from tools.action_parser_demo_runner import DemoResult


class FakeRunner:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, cases, *, on_result=None, stop_requested=None):
        results = []
        for case in cases:
            actual = {
                "task": case.expected_task, "object": case.expected_object,
                "operation": case.expected_operation, "target": case.expected_target,
                "actions": list(case.expected_actions),
            }
            result = DemoResult.from_outcome(case, actual=actual, passed=True, elapsed_seconds=0.01)
            results.append(result)
            if on_result:
                on_result(result, len(results))
        return results


class ActionParser500CaseDemoTest(unittest.TestCase):
    def test_gui_has_no_button_row_status_text_area(self):
        source = inspect.getsource(DemoApp)
        self.assertNotIn("status_var", source)

    def test_default_arguments_are_reproducible_500(self):
        args = parse_args([])
        self.assertEqual(args.count, 500)
        self.assertEqual(args.seed, 20260813)
        self.assertFalse(args.no_gui)

    def test_worker_emits_generated_progress_result_and_summary(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(count=2, seed=7, checkpoint=Path("unused.pt"), device="cpu",
                               output_dir=Path(directory), beam_width=3, max_actions=12)
            summary, run_dir = run_batch(config, events.append, threading.Event(),
                                         runner_factory=FakeRunner)
            self.assertEqual(summary["completed"], 2)
            self.assertTrue(run_dir.is_dir())
        self.assertGreaterEqual(
            {event["type"] for event in events},
            {"generated", "progress", "result", "summary"},
        )


if __name__ == "__main__":
    unittest.main()
