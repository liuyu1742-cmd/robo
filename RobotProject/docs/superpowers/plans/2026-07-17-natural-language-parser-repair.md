# Natural-language Action Parser Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make natural-language instruction resolution deterministic, catalog-compatible and auditable without training on the 414-case diagnostic set.

**Architecture:** `tools.manual_instruction_entry` remains the single resolver used by the popup and evaluator. A small diagnostic module aggregates frozen-case errors by task, object and failure reason. The action-constraint layer continues to score raw and final actions separately. A new human-only template is generated only after parser repairs and is not populated automatically.

**Tech Stack:** Python 3.11, standard library `unittest`, JSON/CSV, existing PyTorch runtime.

## Global Constraints

- Canonical scope is exactly 15 tasks, 123 unique objects and 138 task-object relations from `meta/household_task_catalog.json`.
- Historical `120/120` means 120 standard test cases, never 120 objects.
- Do not use any sentence in `datasets/independent_blind_action_test_414.json` as a training sample or as the final independent test.
- Raw model actions and constrained final actions must remain separately stored and scored.
- A new independent formal test must contain only human-written text, hide standard actions from its author, and remain frozen after creation.

---

## File structure

- Modify `tools/manual_instruction_entry.py`: resolve current canonical tasks consistently and return a structured clarification failure rather than an uncaught field/key error.
- Create `tools/blind_error_diagnostics.py`: classify frozen diagnostic-case failures without modifying their instructions.
- Modify `evaluate_independent_blind_action_test.py`: include the new diagnostic counters in its report.
- Modify `prepare_blind_paraphrase_test.py`: create a separate, human-only template whose identity differs from the mixed 414 template.
- Modify `tests/test_action_constraints.py` and create `tests/test_manual_instruction_resolution.py`, `tests/test_blind_error_diagnostics.py`: test the resolver and audit contract before production changes.
- Modify `docs/CURRENT_PROJECT_STATUS_2026-07-04.md`: report the current count and distinguish diagnostic, raw, final and future human-only scores.

### Task 1: Lock down catalog-compatible resolution and clarification behavior

**Files:**
- Create: `tests/test_manual_instruction_resolution.py`
- Modify: `tools/manual_instruction_entry.py`

**Interfaces:**
- Consumes: `resolve_manual_instruction(instruction: str, benchmark_records: list[dict] | None = None) -> ManualInstruction`
- Produces: `ClarificationRequiredError(ValueError)` for an instruction that has multiple valid task-object relations, and a `ManualInstruction` using an acceptance task ID for executable requests.

- [ ] **Step 1: Write failing tests**

```python
import unittest

from tools.manual_instruction_entry import (
    ClarificationRequiredError,
    resolve_manual_instruction,
)


class ManualInstructionResolutionTest(unittest.TestCase):
    def test_current_cleaning_relation_resolves_without_legacy_field_error(self):
        resolved = resolve_manual_instruction("清洁一下大理石地面")
        self.assertEqual(resolved.task, "cleaning")
        self.assertEqual(resolved.object, "marble_floor")

    def test_ambiguous_object_only_request_requires_clarification(self):
        with self.assertRaisesRegex(ClarificationRequiredError, "请说明"):
            resolve_manual_instruction("帮我处理杯子")
```

- [ ] **Step 2: Verify the tests fail for the expected missing behavior**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_resolution -v `

Expected: the import or resolution assertion fails because structured clarification is not implemented.

- [ ] **Step 3: Implement the minimal resolver contract**

```python
class ClarificationRequiredError(ValueError):
    """The instruction maps to multiple canonical task-object relations."""


def _clarification_error(candidates: list[str]) -> ClarificationRequiredError:
    return ClarificationRequiredError(
        "请说明具体要做什么，例如“清洗杯子”或“把水杯拿给我”；候选任务："
        + ", ".join(candidates)
    )
```

Replace ambiguous `ValueError("Ambiguous instruction; ...")` branches in `resolve_manual_instruction` with `_clarification_error(candidates)`. Ensure all catalog count access uses `load_household_catalog()` and `.get("acceptance_tasks", [])` / `.get("objects", [])`, never legacy catalog fields.

- [ ] **Step 4: Verify resolver tests pass**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_resolution tests.test_action_constraints -v `

Expected: PASS; no `KeyError: 'objects'` occurs for canonical floor requests.

- [ ] **Step 5: Commit**

This workspace is not a Git worktree. Record the completed task in this plan and leave version-control commit responsibility to the project owner.

### Task 2: Add high-risk intent regression tests and minimal phrase fixes

**Files:**
- Modify: `tests/test_manual_instruction_resolution.py`
- Modify: `tools/manual_instruction_entry.py`

**Interfaces:**
- Consumes: natural Chinese request text.
- Produces: `ManualInstruction(task, object, operation, target)` with the requested action direction retained.

- [ ] **Step 1: Write failing tests for known error families**

```python
def test_close_reading_lamp_never_resolves_to_turn_on(self):
    resolved = resolve_manual_instruction("睡前把阅读灯关了")
    self.assertEqual((resolved.task, resolved.object, resolved.operation),
                     ("appliance_management", "reading_lamp", "turn_off"))

def test_watering_is_not_plant_maintenance(self):
    resolved = resolve_manual_instruction("花有点干，帮我浇点水")
    self.assertEqual((resolved.task, resolved.object, resolved.operation),
                     ("maintenance_management", "flower_pot", "water_plants"))

def test_tv_switch_off_uses_entertainment_task(self):
    resolved = resolve_manual_instruction("电视不用看了，帮我关掉")
    self.assertEqual((resolved.task, resolved.object, resolved.operation),
                     ("entertainment_service", "television", "turn_off"))
```

- [ ] **Step 2: Verify the selected tests fail**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_resolution.ManualInstructionResolutionTest -v `

Expected: at least one requested colloquial form is unresolved or chooses the wrong operation before its phrase rule is added.

- [ ] **Step 3: Add only phrase rules supported by the tests**

Add entries to `INTENT_PHRASE_RULES` with the existing fields `task`, `object`, `operation`, `target`, `phrases`, and `priority`. For the three tests above, include `睡前把阅读灯关了`, `花有点干，帮我浇点水`, and `电视不用看了，帮我关掉`; do not add broad one-character rules such as `开` or `关`.

- [ ] **Step 4: Verify all parser and action-constraint tests pass**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_resolution tests.test_action_constraints -v `

Expected: PASS and the returned task/object/operation triples match the assertions exactly.

- [ ] **Step 5: Commit**

This workspace is not a Git worktree. Record the completed task in this plan and leave version-control commit responsibility to the project owner.

### Task 3: Make frozen diagnostic failures traceable without altering the 414 cases

**Files:**
- Create: `tools/blind_error_diagnostics.py`
- Create: `tests/test_blind_error_diagnostics.py`
- Modify: `evaluate_independent_blind_action_test.py`

**Interfaces:**
- Consumes: evaluation case records with `expected_actions`, `raw_actions`, `final_actions`, `error`, `acceptance_task`, and `object`.
- Produces: `summarize_failures(cases: Sequence[Mapping]) -> dict[str, object]` with `total_cases`, `failures`, `by_reason`, `by_task`, and `by_object`.

- [ ] **Step 1: Write a failing diagnostic test**

```python
from tools.blind_error_diagnostics import summarize_failures


def test_groups_key_error_and_sequence_mismatch_without_mutating_cases():
    cases = [
        {"acceptance_task": "cleaning", "object": "marble_floor", "error": "KeyError: 'objects'", "raw_exact_match": False},
        {"acceptance_task": "laundry", "object": "towel", "error": None, "raw_exact_match": False},
        {"acceptance_task": "laundry", "object": "towel", "error": None, "raw_exact_match": True},
    ]
    report = summarize_failures(cases)
    assert report["failures"] == 2
    assert report["by_reason"] == {"catalog_compatibility": 1, "sequence_mismatch": 1}
    assert report["by_task"]["cleaning"] == 1
```

- [ ] **Step 2: Verify it fails because the module is absent**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_blind_error_diagnostics -v `

Expected: FAIL with `ModuleNotFoundError: tools.blind_error_diagnostics`.

- [ ] **Step 3: Implement deterministic classification**

```python
def failure_reason(case: Mapping) -> str:
    error = str(case.get("error") or "")
    if "KeyError" in error or "catalog" in error.lower():
        return "catalog_compatibility"
    if "Ambiguous" in error or "请说明" in error:
        return "ambiguous_instruction"
    if error:
        return "runtime_error"
    return "sequence_mismatch"
```

Implement `summarize_failures` using `collections.Counter`, counting only cases where `raw_exact_match` is false. Insert its output into the existing report under `diagnostic_failures`; do not rewrite case instructions, predictions, fingerprints, or score fields.

- [ ] **Step 4: Verify diagnostic and evaluator tests pass**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_blind_error_diagnostics tests.test_independent_blind_evaluation -v `

Expected: PASS; existing raw/final scoring assertions remain unchanged.

- [ ] **Step 5: Commit**

This workspace is not a Git worktree. Record the completed task in this plan and leave version-control commit responsibility to the project owner.

### Task 4: Create a separate human-only independent retest template

**Files:**
- Modify: `prepare_blind_paraphrase_test.py`
- Modify: `tests/test_blind_paraphrase_workflow.py`
- Create: `datasets/human_only_instruction_retest_template_414.csv` only when the command is run

**Interfaces:**
- Consumes: the 138 canonical acceptance relations.
- Produces: a 414-row CSV with columns `id`, `relation_key`, `variant`, `scenario_hint`, `object_hint`, `human_instruction`; `human_instruction` is empty and expected actions are not exported.

- [ ] **Step 1: Write a failing template-contract test**

```python
def test_human_only_template_has_three_blank_variants_and_no_actions(tmp_path):
    output = tmp_path / "human_only.csv"
    build_human_only_retest_template(output)
    rows = read_csv(output)
    assert len(rows) == 414
    assert all(not row["human_instruction"].strip() for row in rows)
    assert "actions" not in rows[0]
    assert rows[0]["id"].startswith("human_retest__")
```

- [ ] **Step 2: Verify the test fails**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_blind_paraphrase_workflow -v `

Expected: FAIL because `build_human_only_retest_template` is not yet exported.

- [ ] **Step 3: Implement the template builder and CLI mode**

Add `build_human_only_retest_template(output: Path) -> None` to `tools.blind_paraphrase.py`. Reuse canonical relation discovery but set IDs to `human_retest__{relation_key}__v{variant}`; write only the six stated columns and leave `human_instruction` empty. Add `--mode build-human-retest-template` to `prepare_blind_paraphrase_test.py` to call it with output `datasets/human_only_instruction_retest_template_414.csv`.

- [ ] **Step 4: Verify the template is safe for independent authors**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_blind_paraphrase_workflow -v `

Then run: ` .\.venv\Scripts\python.exe prepare_blind_paraphrase_test.py --mode build-human-retest-template `

Expected: tests PASS; the generated CSV has 414 blank human-instruction fields and no action sequence column.

- [ ] **Step 5: Commit**

This workspace is not a Git worktree. Record the completed task in this plan and leave version-control commit responsibility to the project owner.

### Task 5: Publish an honest progress update and verify the focused workflow

**Files:**
- Modify: `docs/CURRENT_PROJECT_STATUS_2026-07-04.md`

**Interfaces:**
- Consumes: canonical catalog counts and the frozen pretest report.
- Produces: a status section identifying 414 mixed cases as diagnostics and the new 414-row human-only template as the formal future retest input.

- [ ] **Step 1: Add documentation assertions before editing prose**

Add to `tests/test_independent_blind_evaluation.py`:

```python
def test_status_uses_current_catalog_counts_and_distinguishes_test_cases(self):
    status = (ROOT / "docs/CURRENT_PROJECT_STATUS_2026-07-04.md").read_text(encoding="utf-8")
    self.assertIn("123种去重物体", status)
    self.assertIn("138条任务—物体操作关系", status)
    self.assertIn("120条测试样例", status)
    self.assertIn("65.22%", status)
```

- [ ] **Step 2: Verify the assertion fails only if wording is absent**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation.IndependentBlindEvaluationTest.test_status_uses_current_catalog_counts_and_distinguishes_test_cases -v `

Expected: FAIL until the status wording is updated.

- [ ] **Step 3: Update the status section**

State that the canonical directory is 15/123/138; label every residual historical `120/120` reference as a historical 120-case standard test; preserve the 65.22% raw and 65.70% final mixed pretest scores; state that the next formal claim depends on a new all-human frozen retest.

- [ ] **Step 4: Run focused end-to-end verification**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_resolution tests.test_action_constraints tests.test_blind_error_diagnostics tests.test_blind_paraphrase_workflow tests.test_independent_blind_evaluation -v `

Run: ` .\.venv\Scripts\python.exe -m compileall -q tools prepare_blind_paraphrase_test.py evaluate_independent_blind_action_test.py manual_instruction_test.py `

Expected: all focused tests PASS and compilation exits with code 0. Do not claim a new ≥90% result unless a newly created all-human frozen retest is evaluated.

- [ ] **Step 5: Commit**

This workspace is not a Git worktree. Record the completed task in this plan and leave version-control commit responsibility to the project owner.
