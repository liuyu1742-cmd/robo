import tempfile
import unittest
from pathlib import Path

from tools.run_decision_planning_benchmark import (
    DEFAULT_RANDOM_SEED,
    run_benchmark,
    select_test_cases,
)


class DecisionPlanningBenchmarkTest(unittest.TestCase):
    def test_seeded_selection_uses_two_different_single_tasks_and_coordination(self):
        cases = select_test_cases(DEFAULT_RANDOM_SEED)

        self.assertEqual(len(cases), 3)
        self.assertEqual(len({case["task"] for case in cases}), 3)
        self.assertTrue(all(case["source"] == "VLA" for case in cases[:2]))
        self.assertTrue(
            all(case["object_id"].startswith("vla_") for case in cases[:2])
        )
        self.assertNotEqual(cases[0]["task"], cases[1]["task"])
        self.assertNotEqual(cases[0]["object_id"], cases[1]["object_id"])
        self.assertEqual(cases[-1]["task"], "multi_device_coordination")
        self.assertEqual(cases[-1]["case_id"], "DEC-003")

    def test_benchmark_records_each_run_real_output_and_replanning(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark.json"
            evidence_directory = Path(directory) / "evidence"
            report = run_benchmark(
                output_path=output,
                evidence_directory=evidence_directory,
                repetitions=2,
                random_seed=DEFAULT_RANDOM_SEED,
            )
            self.assertEqual(report["repetitions"], 2)
            self.assertEqual(len(report["cases"]), 3)
            self.assertTrue(
                all(
                    case["structured_output"]["result_type"]
                    == "vla_task_decision"
                    for case in report["cases"][:2]
                )
            )
            for case in report["cases"]:
                self.assertEqual(len(case["elapsed_seconds"]), 2)
                self.assertEqual(len(case["runs"]), 2)
                self.assertGreaterEqual(case["average_seconds"], 0)
                self.assertTrue(case["real_output"])
                self.assertTrue(case["qualified"])
                for index, run in enumerate(case["runs"], start=1):
                    self.assertEqual(run["run_index"], index)
                    self.assertEqual(
                        run["elapsed_ns"],
                        run["ended_perf_counter_ns"]
                        - run["started_perf_counter_ns"],
                    )
                    self.assertEqual(
                        run["elapsed_seconds"], run["elapsed_ns"] / 1_000_000_000
                    )
                    self.assertEqual(len(run["output_sha256"]), 64)
                    evidence_path = evidence_directory / run["evidence_file"]
                    self.assertTrue(evidence_path.is_file())
                    evidence = evidence_path.read_text(encoding="utf-8")
                    self.assertIn(run["output_sha256"], evidence)
            coordinated = report["cases"][-1]
            self.assertEqual(len(coordinated["structured_output"]["subtasks"]), 3)
            self.assertTrue(coordinated["replanning"]["triggered"])
            self.assertLessEqual(
                coordinated["replanning"]["elapsed_seconds"], 5.0
            )
            self.assertIn(
                "set_mode(air_conditioner,cool)",
                coordinated["replanning"]["actions"],
            )
            self.assertTrue(
                (
                    evidence_directory
                    / coordinated["replanning"]["evidence_file"]
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
