"""Integrity/pollution sentinel only; Blind v2 is not an accuracy target."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "detailed_action_runtime_blind_v2.json"
SIDECAR = ROOT / "tests" / "fixtures" / "detailed_action_runtime_blind_v2.sha256"
RUNTIME = ROOT / "tools" / "detailed_action_runtime.py"


class DetailedActionRuntimeBlindV2PollutionSentinelTest(unittest.TestCase):
    def test_frozen_fixture_is_intact_and_not_loaded_by_runtime(self):
        expected, filename = SIDECAR.read_text(encoding="utf-8").strip().split()
        self.assertEqual(filename, FIXTURE.name)
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), expected)
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertNotIn(FIXTURE.name, source)

    def test_no_complete_frozen_instruction_was_copied_into_runtime(self):
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertTrue(all(case["instruction"] not in source for case in cases))


if __name__ == "__main__":
    unittest.main()
