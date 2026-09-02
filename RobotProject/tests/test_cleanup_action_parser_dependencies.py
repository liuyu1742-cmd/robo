import json
from pathlib import Path

from tools.manual_instruction_entry import (
    describe_actions,
    find_expected_actions,
    load_benchmark_records,
    resolve_manual_instruction,
)
from tools.project_runtime import load_project_v2_model


ROOT = Path(__file__).resolve().parents[1]


def test_action_parser_dependency_catalogs_are_present():
    """The manual action parser must retain its catalog sources after cleanup."""
    required = (
        ROOT / "meta" / "compliance_catalog_15x120.json",
        ROOT / "meta" / "household_task_catalog.json",
        ROOT / "meta" / "project_action_vocab.json",
        ROOT / "meta" / "project_condition_vocab.json",
        ROOT / "datasets" / "standard_instruction_action_benchmark.json",
    )
    assert all(path.is_file() for path in required)


def test_rebuilt_benchmark_resolves_a_canonical_cleaning_instruction():
    resolved = resolve_manual_instruction("清洁地毯", benchmark_records=load_benchmark_records())
    assert resolved.task == "cleaning"
    assert resolved.object == "carpet"
    assert find_expected_actions(resolved, load_benchmark_records())


def test_clean_action_can_be_rendered_for_the_manual_result():
    rendered = describe_actions(["clean(carpet)"])
    assert len(rendered) == 1


def test_rebuilt_catalog_includes_vla_task_and_object_extensions():
    catalog = json.loads(
        (ROOT / "meta" / "compliance_catalog_15x120.json").read_text(encoding="utf-8")
    )
    assert len(catalog["vla_tasks"]) == 10
    assert len(catalog["vla_objects"]) == 82


def test_existing_project_checkpoint_matches_the_restored_vocabularies():
    checkpoint = ROOT / "models" / "action_transformer_project_generation_semantic.pt"
    model, _ = load_project_v2_model(checkpoint, device="cpu")
    assert model is not None
