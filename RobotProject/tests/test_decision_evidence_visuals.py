import tempfile
import unittest
from pathlib import Path

from tools.build_decision_evidence_dashboard import build_dashboard


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "outputs" / "decision_planning_benchmark_20260731.json"
EVIDENCE = ROOT / "outputs" / "decision_planning_evidence_20260731"


class DecisionEvidenceVisualsTest(unittest.TestCase):
    def test_dashboard_contains_five_verified_screenshot_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            summary = build_dashboard(BENCHMARK, EVIDENCE, output)
            html = output.read_text(encoding="utf-8")
            section_files_exist = {
                section_id: (output.parent / f"{section_id}.html").is_file()
                for section_id in summary["section_ids"]
            }

        self.assertEqual(summary["verified_evidence_files"], 31)
        self.assertEqual(
            summary["section_ids"],
            [
                "shot-overview",
                "shot-vla",
                "shot-multi",
                "shot-subtasks",
                "shot-chart",
                "shot-code",
            ],
        )
        for section_id in summary["section_ids"]:
            self.assertIn(f'id="{section_id}"', html)
            self.assertTrue(section_files_exist[section_id], section_id)
        self.assertIn("牙膏", html)
        self.assertIn("数码相机", html)
        self.assertIn("DEC-003-R10", html)
        self.assertIn("DEC-003-REPLAN", html)
        self.assertIn("appliance_management / air_conditioner", html)
        self.assertIn("appliance_management / electric_curtain", html)
        self.assertIn("bedroom_service / bedroom_lamp", html)
        self.assertIn("ended_perf_counter_ns-started_perf_counter_ns", html)
        self.assertIn("def _timed_plan(", html)
        self.assertIn("started_ns = time.perf_counter_ns()", html)
        self.assertIn("ended_ns = time.perf_counter_ns()", html)
        self.assertIn("elapsed_ns = ended_ns - started_ns", html)
        self.assertIn("output_sha256", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)


if __name__ == "__main__":
    unittest.main()
