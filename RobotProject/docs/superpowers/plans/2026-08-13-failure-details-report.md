# Failure Details Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a human-readable `failure_details.txt` artifact containing concrete diagnostics for every failed action-parser case.

**Architecture:** Extend `DemoRunOutput` with one focused formatter that groups existing `DemoResult` records by their mutually exclusive failure category. Finalization writes the new UTF-8 artifact alongside the existing six outputs without changing their schemas.

**Tech Stack:** Python 3.11, unittest, pathlib, existing action-parser result dataclasses.

## Global Constraints

- Preserve all existing output artifacts and calculation rules.
- List only failed cases and preserve execution order within each category.
- Include expected and actual semantic fields, reference and generated sequences, concrete reason, and raw exception.
- Always create `failure_details.txt`, including for a zero-failure run.

---

### Task 1: Failure Details Artifact

**Files:**
- Modify: `tools/action_parser_demo_output.py`
- Modify: `tests/test_action_parser_demo_output.py`

**Interfaces:**
- Consumes: accumulated `DemoResult` values already held by `DemoRunOutput`.
- Produces: `DemoRunOutput.failure_details_path` and UTF-8 `failure_details.txt` during `finalize(summary)`.

- [ ] **Step 1: Write a failing output test**

Create failed task-mismatch and action-sequence-mismatch results, finalize the writer, and assert that `failure_details.txt` contains the category headings, instruction, expected/actual values, standard sequence, generated sequence, and concrete reason.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_output -v`

Expected: failure because `failure_details.txt` is missing.

- [ ] **Step 3: Implement the formatter and output path**

Add `failure_details_path`, write it in `finalize`, group failed records in `FAILURE_CATEGORIES` order, and format missing values as `无` and action lists as numbered steps.

- [ ] **Step 4: Run focused and complete tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_output tests.test_action_parser_demo_runner tests.test_action_parser_500case_demo -v`

Expected: all tests pass.

- [ ] **Step 5: Regenerate the latest real report and verify counts**

Load the latest 500-case JSONL, reconstruct `DemoResult` values or provide a supported report regeneration helper, produce `failure_details.txt`, and verify the number of `用例ID` entries equals the summary failed count.

Expected: 84 failed-case entries for the latest fixed-fold run.
