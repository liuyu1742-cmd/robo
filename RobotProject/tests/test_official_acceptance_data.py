import json
import tempfile
import unittest
from pathlib import Path

from tools.official_acceptance_data import build_official_acceptance


def reading_lamp_record():
    return {
        "id": "acceptance_reading_lamp",
        "relation_key": "appliance_management::reading_lamp",
        "instruction": "请关闭阅读灯",
        "acceptance_task": "appliance_management",
        "object": "reading_lamp",
        "target": "none",
        "actions": ["locate(reading_lamp)", "turn_off(reading_lamp)"],
    }


class OfficialAcceptanceDataTest(unittest.TestCase):
    def test_official_acceptance_has_four_cases_per_relation(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes = Path(directory) / "final_hashes.json"
            hashes.write_text(json.dumps({"hashes": []}), encoding="utf-8")
            bundle = build_official_acceptance([reading_lamp_record()], [], hashes)
        self.assertEqual(len(bundle["cases"]), 4)
        self.assertEqual(bundle["minimum_raw_passes_at_90_percent"], 4)
        self.assertEqual(bundle["evidence_scope"], "fixed_command_acceptance")


if __name__ == "__main__":
    unittest.main()
