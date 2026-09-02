"""Prepare or build an independent human paraphrase blind-test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.blind_paraphrase import (
    VARIANT_TEMPLATE_COLUMNS,
    build_frozen_blind_bundle,
    build_human_only_retest_template_rows,
    build_variant_template_rows,
    read_csv,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_CATALOG = ROOT / "meta" / "household_task_catalog.json"
DEFAULT_TEMPLATE = ROOT / "datasets" / "instruction_blind_paraphrase_template_414.csv"
DEFAULT_HUMAN_ONLY_RETEST_TEMPLATE = ROOT / "datasets" / "human_only_instruction_retest_template_414.csv"
DEFAULT_OUTPUT = ROOT / "datasets" / "independent_blind_action_test_414.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("export-template", "build-test", "build-human-retest-template"),
        default="export-template",
        help="export-template creates a CSV for humans; build-test converts filled CSV to JSON.",
    )
    parser.add_argument(
        "--variants-per-relation",
        type=int,
        default=3,
        help="Independent human wordings for each task-object relation; formal protocol requires 3.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Source split for blind-test cases. Keep test for formal blind testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if args.variants_per_relation != 3:
        raise ValueError("The formal independent blind-test protocol requires --variants-per-relation 3.")
    if args.mode in {"export-template", "build-human-retest-template"}:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        task_names = {task["id"]: task["name_zh"] for task in catalog["acceptance_tasks"]}
        if args.mode == "build-human-retest-template":
            rows = build_human_only_retest_template_rows(
                records,
                variants_per_relation=args.variants_per_relation,
                task_names_zh=task_names,
            )
            template_path = DEFAULT_HUMAN_ONLY_RETEST_TEMPLATE
        else:
            rows = build_variant_template_rows(
                records,
                variants_per_relation=args.variants_per_relation,
                task_names_zh=task_names,
            )
            template_path = args.template
        write_csv(template_path, rows, fieldnames=VARIANT_TEMPLATE_COLUMNS)
        print(f"template_rows: {len(rows)}", flush=True)
        print(f"relations: {len(records)}", flush=True)
        print(f"template: {template_path.resolve()}", flush=True)
        print("请让人工只填写 human_instruction 列；不要把标准动作序列给填写者看。", flush=True)
        return

    filled_rows = read_csv(args.template)
    blind_bundle = build_frozen_blind_bundle(
        records,
        filled_rows,
        variants_per_relation=args.variants_per_relation,
        source_csv_bytes=args.template.read_bytes(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(blind_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"blind_records: {len(blind_bundle['cases'])}", flush=True)
    print(f"output: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
