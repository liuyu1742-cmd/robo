import json
import tempfile
import unittest
from pathlib import Path

import prepare_blind_paraphrase_test as blind_entry
from tools.fill_missing_blind_instructions import everyday_instruction
from tools.blind_paraphrase import (
    build_human_only_retest_template_rows,
    build_blind_records,
    build_frozen_blind_bundle,
    build_template_rows,
    build_variant_template_rows,
    read_csv,
)


ROOT = Path(__file__).resolve().parents[1]


def sample_record(case_id, instruction):
    return {
        "id": case_id,
        "instruction": instruction,
        "task": "laundry",
        "object": "child_clothing",
        "target": "washer",
        "actions": [
            "locate(child_clothing)",
            "grasp(child_clothing)",
            "move(washer)",
            "wash(child_clothing)",
            "dry(child_clothing)",
        ],
        "split": "test",
    }


class BlindParaphraseWorkflowTest(unittest.TestCase):
    def test_assisted_instruction_is_daily_chinese_and_varies_by_variant(self):
        record = {
            "instruction": "智能烹饪：启动烤箱并完成烹饪操作（oven）。",
        }

        first = everyday_instruction(record, variant=1)
        second = everyday_instruction(record, variant=2)

        self.assertIn("烤箱", first)
        self.assertNotIn("oven", first)
        self.assertNotEqual(first, second)

    def test_read_csv_accepts_excel_gb18030_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.csv"
            path.write_bytes(
                "case_id,human_instruction\ncase__v1,帮我倒水\n".encode("gb18030")
            )

            rows = read_csv(path)

        self.assertEqual(rows[0]["human_instruction"], "帮我倒水")

    def test_formal_template_has_three_rows_for_every_acceptance_relation(self):
        records = json.loads(
            (ROOT / "datasets/acceptance_instruction_action_benchmark_15tasks.json").read_text(
                encoding="utf-8"
            )
        )
        rows = build_variant_template_rows(records, variants_per_relation=3)

        self.assertEqual(len(records), 138)
        self.assertEqual(len(rows), 414)
        self.assertEqual(len({row["relation_key"] for row in rows}), 138)
        self.assertEqual(
            blind_entry.DEFAULT_BENCHMARK,
            ROOT / "datasets/acceptance_instruction_action_benchmark_15tasks.json",
        )
        self.assertEqual(
            blind_entry.DEFAULT_TEMPLATE.name,
            "instruction_blind_paraphrase_template_414.csv",
        )

    def test_human_only_retest_template_has_blank_variants_and_no_actions(self):
        records = json.loads(
            (ROOT / "datasets/acceptance_instruction_action_benchmark_15tasks.json").read_text(
                encoding="utf-8"
            )
        )

        rows = build_human_only_retest_template_rows(records, variants_per_relation=3)

        self.assertEqual(len(rows), 414)
        self.assertTrue(all(not row["human_instruction"].strip() for row in rows))
        self.assertTrue(all("actions" not in row for row in rows))

    def test_variant_template_creates_three_unique_rows_without_actions(self):
        rows = build_variant_template_rows([sample_record("laundry_child", "old")])

        self.assertEqual([row["variant"] for row in rows], ["1", "2", "3"])
        self.assertEqual(
            {row["case_id"] for row in rows},
            {
                "laundry_child__v1",
                "laundry_child__v2",
                "laundry_child__v3",
            },
        )
        self.assertTrue(all("actions" not in row for row in rows))

    def test_frozen_bundle_rejects_duplicate_normalized_human_instructions(self):
        rows = [
            {"case_id": "laundry_child__v1", "human_instruction": "请洗衣物"},
            {"case_id": "laundry_child__v2", "human_instruction": "请  洗衣物"},
            {"case_id": "laundry_child__v3", "human_instruction": "帮我洗衣物"},
        ]

        with self.assertRaisesRegex(ValueError, "duplicate normalized instruction"):
            build_frozen_blind_bundle([sample_record("laundry_child", "old")], rows)

    def test_template_hides_expected_actions_but_keeps_human_context(self):
        rows = build_template_rows(
            [sample_record("laundry_child_clothing_test", "请帮我处理 child_clothing，执行衣物清洗任务")]
        )

        self.assertEqual(rows[0]["case_id"], "laundry_child_clothing_test")
        self.assertEqual(rows[0]["task"], "laundry")
        self.assertEqual(rows[0]["object"], "child_clothing")
        self.assertIn("human_instruction", rows[0])
        self.assertNotIn("actions", rows[0])

    def test_build_blind_records_attaches_hidden_answers_after_human_fill(self):
        standard = [sample_record("laundry_child_clothing_test", "old template")]
        filled_rows = [
            {
                "case_id": "laundry_child_clothing_test",
                "human_instruction": "把衣服清洗干净",
            }
        ]

        blind = build_blind_records(standard, filled_rows)

        self.assertEqual(blind[0]["id"], "blind_laundry_child_clothing_test")
        self.assertEqual(blind[0]["instruction"], "把衣服清洗干净")
        self.assertEqual(blind[0]["actions"], standard[0]["actions"])
        self.assertEqual(blind[0]["split"], "test")


if __name__ == "__main__":
    unittest.main()
