# VLA 82 Object Action Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only 10-task/82-object action-template catalogue to the manual action parsing output.

**Architecture:** A JSON catalogue stores workbook-derived candidates and Chinese step chains. A focused loader/selector validates the catalogue and chooses one candidate from instruction text, parsed task and object; the manual runtime overlays selected template steps while retaining legacy fallback behavior.

**Tech Stack:** Python, JSON, dataclasses, unittest.

## Global Constraints

- Do not modify, move, or write any video, video annotation, dataset directory, or external XLSX workbook.
- Do not retrain models or modify existing 15-task/120-object benchmark data.
- Cover exactly 10 tasks and 82 unique workbook objects; each candidate has at least three distinct Chinese steps.
- Select exactly one candidate for duplicate object operations, with source-row order as the tie breaker.

---

## File Structure

- `meta/vla_82_action_templates.json`: versioned operation candidates.
- `tools/vla_action_templates.py`: validation, object alias resolution, deterministic candidate selection.
- `tests/test_vla_action_templates.py`: coverage and selection tests.
- `manual_instruction_test.py`: template overlay in the existing runtime.
- `tools/manual_instruction_entry.py`: result metadata and formatted output.
- `tests/test_manual_instruction_runtime.py`: end-to-end output and legacy fallback tests.

### Task 1: Create and validate catalogue

**Files:** Create `meta/vla_82_action_templates.json`, `tools/vla_action_templates.py`, `tests/test_vla_action_templates.py`.

- [ ] Write a failing `unittest` asserting `len(tasks) == 10`, `len(unique_object_ids) == 82`, and every candidate has at least three steps.
- [ ] Implement `load_vla_action_templates() -> dict` and `validate_vla_action_templates(catalogue: Mapping[str, object]) -> list[str]`.
- [ ] Populate every data row in sheets `表5-8-1` to `表5-8-11` with `task_id`, `object_id`, `object_name_zh`, source label, keywords, Chinese steps, and `{sheet,row}` provenance.
- [ ] Verify with `.\\.venv\\Scripts\\python.exe -m unittest tests.test_vla_action_templates -v`.

### Task 2: Select a unique template

**Files:** Modify `tools/vla_action_templates.py`, `tests/test_vla_action_templates.py`.

- [ ] Write failing tests for "把客厅里的多个汽水罐扔进厨房垃圾桶" selecting `["拿起汽水罐", "移动至厨房垃圾桶", "投放汽水罐", "确认已放入"]`, and separate commands selecting distinct same-object operations.
- [ ] Implement `VlaActionCandidate` and `select_vla_action_template(instruction: str, *, task: str | None, object_id: str | None) -> VlaActionCandidate | None`.
- [ ] Score task/object/name/source-label/keywords, sorting candidates by `(source_sheet, source_row)` before choosing the highest score so ties select the earliest source row.
- [ ] Implement `template_actions(candidate) -> list[str]`, producing `template_step(<Chinese text>)` actions.
- [ ] Re-run the isolated tests until they pass.

### Task 3: Integrate with manual parsing output

**Files:** Modify `manual_instruction_test.py`, `tools/manual_instruction_entry.py`, `tests/test_manual_instruction_runtime.py`.

- [ ] Write a failing runtime test that invokes `run_instruction(..., "把客厅里的多个汽水罐扔进厨房垃圾桶")` with `no_model=True` and expects template actions plus selection metadata.
- [ ] Add a VLA object-alias resolver for instructions outside the legacy 120-object catalogue.
- [ ] In `run_instruction`, select a VLA template after legacy resolution and use `template_actions` only when selection succeeds; otherwise preserve `find_expected_actions` and existing constraint behavior.
- [ ] Extend `build_manual_result` with optional `template_selection` and render `预设动作模板：步骤1 → 步骤2` only when present.
- [ ] Re-run runtime regressions and the focused full suite.

### Task 4: Document and end-to-end verify

**Files:** Modify `docs/PROJECT_STRUCTURE.md`.

- [ ] State the catalogue location and its read-only relationship to the external workbook and all video data.
- [ ] Run `.\\.venv\\Scripts\\python.exe -m unittest tests.test_vla_action_templates tests.test_manual_instruction_runtime -v`.
- [ ] Run `.\\.venv\\Scripts\\python.exe manual_instruction_test.py --no-model --no-gui --instruction "把客厅里的多个汽水罐扔进厨房垃圾桶" --report datasets/manual_instruction_test_report.json` and verify the complete garbage-can chain is printed.
- [ ] Commit each independently testable task when a Git worktree is available.

## Self-Review

Tasks 1–2 cover complete data and deterministic multi-operation selection. Task 3 makes the existing parser output selected sequences without changing non-template behavior. Task 4 confirms scope and end-to-end output.
