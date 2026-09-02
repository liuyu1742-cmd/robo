# 500-Case Action Parser Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a recordable GUI and CLI entrypoint that reproducibly generates 500 colloquial commands across the legacy household catalogue and VLA 82 operations, runs the full project pipeline, classifies failures, and exports auditable accuracy logs.

**Architecture:** A pure test-data module creates immutable expected cases from the two approved catalogues. A batch runner executes legacy cases through the real parser/model runtime and VLA cases through the approved VLA planner, assigns one primary failure category, and streams results to an output writer. A Tkinter GUI runs the batch runner on a worker thread and displays progress and summaries without changing the existing manual test entrypoint.

**Tech Stack:** Python 3.11, unittest, Tkinter/ttk, PyTorch project runtime, JSON/JSONL/CSV/text logs.

## Global Constraints

- Default case count is exactly 500 and default seed is exactly `20260813`.
- Identical catalogues, seed, and count must produce byte-for-byte identical generated case payloads.
- Generated instruction texts must be unique within one run and must not copy benchmark sentences.
- Existing video data, task labels, object labels, benchmarks, and VLA catalogues remain read-only.
- Legacy cases use the real model checkpoint and compare raw model actions; VLA cases use the approved VLA planner and detailed action catalogue.
- Every failed case has exactly one primary failure category; category counts must sum to the failed total.
- The GUI remains responsive while the worker executes cases and supports safe stop after the current case.
- A stopped partial run is marked incomplete and is never presented as a completed 500-case accuracy result.

---

### Task 1: Reproducible Unified Case Generator

**Files:**
- Create: `tools/action_parser_demo_cases.py`
- Test: `tests/test_action_parser_demo_cases.py`

**Interfaces:**
- Consumes: `datasets/acceptance_instruction_action_benchmark_15tasks.json`, `tools.vla_action_templates.load_vla_action_templates`, and `tools.vla_detailed_action_sequences.load_vla_detailed_action_sequences`.
- Produces: `DemoCase` and `generate_demo_cases(count: int, seed: int) -> list[DemoCase]`.

- [ ] **Step 1: Write deterministic and coverage tests**

```python
def test_same_seed_generates_same_unique_500_cases():
    first = generate_demo_cases(500, 20260813)
    second = generate_demo_cases(500, 20260813)
    assert first == second
    assert len(first) == 500
    assert len({case.instruction for case in first}) == 500

def test_generator_covers_legacy_and_vla_tasks():
    cases = generate_demo_cases(500, 20260813)
    assert {case.source for case in cases} == {"legacy", "vla"}
    assert len({case.expected_task for case in cases if case.source == "legacy"}) >= 15
    assert len({case.expected_task for case in cases if case.source == "vla"}) >= 10
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_cases -v`

Expected: import failure because `tools.action_parser_demo_cases` does not exist.

- [ ] **Step 3: Implement immutable case records and stratified generation**

```python
@dataclass(frozen=True)
class DemoCase:
    case_id: str
    source: Literal["legacy", "vla"]
    instruction: str
    expected_task: str
    expected_object: str
    expected_operation: str
    expected_target: str | None
    expected_actions: tuple[str, ...]
    template_id: str
    seed: int
```

Load legacy relations from the acceptance benchmark, load VLA relations and detailed steps from approved catalogues, round-robin by source/task/relation, and use a local `random.Random(seed)` instance. Construct commands from request/direct/scenario/ellipsis phrase families plus object aliases, locations, quantities, and operation synonyms. Reject duplicate texts and benchmark sentence matches before assigning sequential `CASE-0001` IDs.

- [ ] **Step 4: Run generator tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_cases -v`

Expected: all generator tests pass.

### Task 2: Full-Chain Runner and Failure Classification

**Files:**
- Create: `tools/action_parser_demo_runner.py`
- Test: `tests/test_action_parser_demo_runner.py`

**Interfaces:**
- Consumes: `DemoCase`, legacy parser/model runtime functions, `plan_vla_instruction`, a reusable loaded model, and stop/progress callbacks.
- Produces: `DemoResult`, `classify_failure(case, actual, error) -> str | None`, `run_demo_case(...)`, and `summarize_results(...)`.

- [ ] **Step 1: Write mutually exclusive classification and summary tests**

```python
def test_failure_categories_are_exclusive_and_close_totals():
    results = sample_results_for_every_failure_category()
    summary = summarize_results(results, requested=12, stopped=False)
    assert summary["passed"] + summary["failed"] == summary["completed"]
    assert sum(summary["failure_categories"].values()) == summary["failed"]
    assert summary["accuracy_percent"] == summary["passed"] / summary["completed"] * 100

def test_missing_actual_object_is_object_not_found():
    assert classify_failure(case, {"resolved": None}, None) == "object_not_found"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_runner -v`

Expected: import failure because the runner module does not exist.

- [ ] **Step 3: Implement result records, full-chain execution, and priority classification**

```python
FAILURE_CATEGORIES = (
    "generation_error", "runtime_error", "model_generation_error",
    "object_not_found", "match_failure", "task_mismatch",
    "object_mismatch", "operation_mismatch", "target_mismatch",
    "action_sequence_mismatch", "other_failure",
)

@dataclass(frozen=True)
class DemoResult:
    case: DemoCase
    actual_task: str | None
    actual_object: str | None
    actual_operation: str | None
    actual_target: str | None
    actual_actions: tuple[str, ...]
    passed: bool
    failure_category: str | None
    error: str | None
    elapsed_seconds: float
```

For legacy cases call the actual parser and project generation runtime using one preloaded model, compare task/object/operation/required target and raw model actions. For VLA cases call `plan_vla_instruction` and compare task/object/operation, template actions, and detailed steps. Catch per-case exceptions, classify with the specified priority, emit progress, and continue.

- [ ] **Step 4: Run runner tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_runner -v`

Expected: all runner tests pass and category totals close.

### Task 3: Incremental Audit Output

**Files:**
- Create: `tools/action_parser_demo_output.py`
- Test: `tests/test_action_parser_demo_output.py`

**Interfaces:**
- Consumes: generated `DemoCase` objects, streamed `DemoResult` objects, and final summary.
- Produces: `DemoRunOutput` with `generated_cases.json`, `case_results.jsonl`, `summary.json`, `results.csv`, `test.log`, and `summary.txt`.

- [ ] **Step 1: Write output completeness tests**

```python
def test_output_writer_persists_every_required_artifact(tmp_path):
    writer = DemoRunOutput(tmp_path, seed=20260813, requested=2)
    writer.write_generated_cases(two_cases)
    writer.append_result(first_result)
    writer.append_result(second_result)
    writer.finalize(summary)
    assert {p.name for p in writer.run_dir.iterdir()} == {
        "generated_cases.json", "case_results.jsonl", "summary.json",
        "results.csv", "test.log", "summary.txt",
    }
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_output -v`

Expected: import failure because the output module does not exist.

- [ ] **Step 3: Implement UTF-8 incremental writers and Chinese summary**

Write generated cases once, append and flush one JSONL record after each completed case, maintain an in-memory CSV row list, and atomically replace final summary files. The text summary must show requested/completed/pass/fail/accuracy, completion state, source subtotals, and every named failure category including zero values.

- [ ] **Step 4: Run output tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_output -v`

Expected: all six files exist and contain consistent totals.

### Task 4: GUI and CLI Demonstration Entrypoint

**Files:**
- Create: `action_parser_500case_demo.py`
- Test: `tests/test_action_parser_500case_demo.py`

**Interfaces:**
- Consumes: generator, runner, output writer, checkpoint path, device, count, seed, and output directory.
- Produces: recordable Tkinter GUI by default and `--no-gui` CLI mode for automation.

- [ ] **Step 1: Write CLI argument and worker-message tests**

```python
def test_default_arguments_are_500_and_fixed_seed():
    args = parse_args([])
    assert args.count == 500
    assert args.seed == 20260813

def test_worker_events_include_progress_result_and_summary():
    events = list(fake_worker_events())
    assert {event["type"] for event in events} >= {"progress", "result", "summary"}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_500case_demo -v`

Expected: import failure because the GUI entrypoint does not exist.

- [ ] **Step 3: Implement CLI orchestration**

Add `--count`, `--seed`, `--checkpoint`, `--device`, `--output-dir`, and `--no-gui`. CLI mode prints periodic progress, writes all artifacts, and exits nonzero only for a run-level failure; individual failed cases remain normal benchmark results.

- [ ] **Step 4: Implement responsive Tkinter page**

Create parameter controls, Generate/Start/Stop/Open buttons, progress bar, current instruction/expected/actual/action fields, summary cards, failure counts, and filterable Treeview. Run generation/model work in one daemon worker thread and deliver dict events through `queue.Queue`; poll the queue with `root.after`. Stop sets a `threading.Event` and finalizes after the active case.

- [ ] **Step 5: Run entrypoint tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_500case_demo -v`

Expected: argument and worker protocol tests pass without opening a GUI window.

### Task 5: Integration Verification and 500-Case Acceptance Run

**Files:**
- Verify: all files created in Tasks 1-4.
- Create at runtime: `outputs/action_parser_demo/<timestamp>_seed20260813_n500/`.

**Interfaces:**
- Consumes: complete implementation and real project checkpoint.
- Produces: verified GUI/CLI entrypoint plus one real 500-case output directory.

- [ ] **Step 1: Run all new unit tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_cases tests.test_action_parser_demo_runner tests.test_action_parser_demo_output tests.test_action_parser_500case_demo -v`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run a real 10-case smoke test**

Run: `.\.venv\Scripts\python.exe action_parser_500case_demo.py --no-gui --count 10 --seed 20260813`

Expected: ten case records, six output artifacts, no run-level crash, and closed summary totals.

- [ ] **Step 3: Run the full real 500-case test**

Run: `.\.venv\Scripts\python.exe action_parser_500case_demo.py --no-gui --count 500 --seed 20260813`

Expected: `requested=500`, `completed=500`, `status=completed`, six output artifacts, and reported pass/fail/category/accuracy totals.

- [ ] **Step 4: Verify generated evidence consistency**

Read `generated_cases.json`, `case_results.jsonl`, `summary.json`, and `results.csv`; assert 500 generated cases, 500 JSONL records, 500 CSV data rows, unique instruction texts, `passed + failed = 500`, and `sum(failure_categories.values()) = failed`.

- [ ] **Step 5: Launch GUI for recordability check**

Run: `.\.venv\Scripts\python.exe action_parser_500case_demo.py`

Expected: the page shows default count 500 and seed 20260813, buttons and summary fields are visible, and starting a small configured run updates progress without freezing.

## Execution Note

The workspace is not a valid Git repository, so task-level commits cannot be produced. Preserve unrelated user files and report this limitation without changing `.git` metadata.
