# Independent Blind Action Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a frozen 414-instruction independent blind-test workflow that reports raw model action accuracy separately from constrained final-system accuracy.

**Architecture:** The canonical acceptance benchmark supplies 138 task—object relations. `tools.blind_paraphrase` will expand each relation into three human-fillable variants and validate/freeze the completed CSV. A new evaluator will first obtain raw model actions, then score those actions, and only afterwards apply `action_constraints` for a separately named final-system score.

**Tech Stack:** Python 3.11, standard library CSV/JSON/hashlib/statistics, PyTorch project runtime, existing `tools.instruction_benchmark.score_predictions`, unittest.

## Global Constraints

- Source coverage is exactly the current 138 rows in `datasets/acceptance_instruction_action_benchmark_15tasks.json`.
- The generated template has exactly 414 rows: three variants for every relation.
- The hidden expected actions must not be supplied to model generation.
- Raw and final scores have distinct report keys and must never be substituted for one another.
- Frozen input requires a matching SHA-256 fingerprint, 414 rows, three variants per relation, nonblank instructions, and no duplicate normalized instructions.
- This task does not retrain or modify model weights, source datasets, or the existing acceptance benchmark.

---

### Task 1: Variant-aware blind-template and freeze primitives

**Files:**
- Modify: `tools/blind_paraphrase.py`
- Modify: `tests/test_blind_paraphrase_workflow.py`

**Interfaces:**
- Produces `build_variant_template_rows(records, variants_per_relation=3, task_names_zh=None) -> list[dict[str, str]]`.
- Produces `build_frozen_blind_bundle(standard_records, filled_rows, variants_per_relation=3) -> dict`.
- Produces `validate_frozen_blind_bundle(bundle, expected_relations) -> None`.
- Bundle fields: `format`, `source_csv_sha256`, `created_at_utc`, `variants_per_relation`, `cases`, `audit`.

- [ ] **Step 1: Write failing tests for 414-style variants and hidden labels**

```python
def test_variant_template_creates_three_unique_rows_without_actions(self):
    rows = build_variant_template_rows([sample_record("laundry_child", "old")])
    assert [row["variant"] for row in rows] == ["1", "2", "3"]
    assert {row["case_id"] for row in rows} == {"laundry_child__v1", "laundry_child__v2", "laundry_child__v3"}
    assert all("actions" not in row for row in rows)

def test_frozen_bundle_rejects_duplicate_normalized_human_instructions(self):
    rows = [
        {"case_id": "laundry_child__v1", "human_instruction": "请洗衣物"},
        {"case_id": "laundry_child__v2", "human_instruction": "请  洗衣物"},
        {"case_id": "laundry_child__v3", "human_instruction": "帮我洗衣物"},
    ]
    with self.assertRaisesRegex(ValueError, "duplicate normalized instruction"):
        build_frozen_blind_bundle([sample_record("laundry_child", "old")], rows)
```

- [ ] **Step 2: Run the new unit tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_blind_paraphrase_workflow -v
```

Expected: `ImportError` or `AttributeError` for the new functions.

- [ ] **Step 3: Implement variant and freeze helpers**

Implement the three interfaces above. Extend the template row fields with `relation_key`, `variant`, `case_id`, task context, object context, target context, and `human_instruction`; do not add an actions column. Parse each `case_id` as `<source_id>__v<positive integer>`, attach standard labels only during bundle construction, calculate `sha256` from the UTF-8 CSV bytes, and store a stable sorted audit containing `relations`, `cases`, `variants_per_relation`, and `relation_digest`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_blind_paraphrase_workflow -v
```

Expected: all existing and new blind-workflow tests pass.

### Task 2: Export and freeze the 414-case human test set

**Files:**
- Modify: `prepare_blind_paraphrase_test.py`
- Modify: `tests/test_blind_paraphrase_workflow.py`

**Interfaces:**
- `--mode export-template` defaults to `datasets/acceptance_instruction_action_benchmark_15tasks.json` and writes 414 blank rows.
- `--mode build-test` writes `datasets/independent_blind_action_test_414.json` as a frozen bundle.
- CLI accepts `--variants-per-relation` with default `3` and rejects any value other than `3` for the formal default output.

- [ ] **Step 1: Write failing CLI/helper test for formal 138×3 coverage**

```python
def test_formal_template_has_three_rows_for_every_acceptance_relation(self):
    records = json.loads((ROOT / "datasets/acceptance_instruction_action_benchmark_15tasks.json").read_text(encoding="utf-8"))
    rows = build_variant_template_rows(records, variants_per_relation=3)
    self.assertEqual(len(records), 138)
    self.assertEqual(len(rows), 414)
    self.assertEqual(len({row["relation_key"] for row in rows}), 138)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_blind_paraphrase_workflow.BlindParaphraseWorkflowTest.test_formal_template_has_three_rows_for_every_acceptance_relation -v
```

Expected: FAIL because the variant helper is absent or the old benchmark is still selected.

- [ ] **Step 3: Update the entry script**

Replace its legacy 120 benchmark/catalog defaults with the acceptance benchmark. Read its `acceptance_task` and `legacy_task_id` fields without changing labels. In export mode call `build_variant_template_rows`; in build mode call `build_frozen_blind_bundle`; write human-readable output naming 138 relations and 414 rows.

- [ ] **Step 4: Run test and export the blank official template**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_blind_paraphrase_workflow -v
.\.venv\Scripts\python.exe prepare_blind_paraphrase_test.py --mode export-template
```

Expected: tests PASS; terminal reports `template_rows: 414` and writes `datasets/instruction_blind_paraphrase_template_414.csv`.

### Task 3: Raw/final evaluation core with confidence intervals

**Files:**
- Create: `tools/independent_blind_evaluation.py`
- Create: `tests/test_independent_blind_evaluation.py`

**Interfaces:**
- `wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]`.
- `evaluate_blind_cases(bundle: Mapping, raw_predictions: Mapping[str, Sequence[str]], final_predictions: Mapping[str, Sequence[str]]) -> dict`.
- Report metrics include `raw_exact_sequence_accuracy`, `final_exact_sequence_accuracy`, raw/final `score_predictions` metrics, `raw_wilson_95`, `final_wilson_95`, and per-case raw/final comparisons.

- [ ] **Step 1: Write failing tests for strict separation and confidence intervals**

```python
def test_evaluator_reports_raw_failure_and_final_success_separately(self):
    bundle = frozen_bundle_with_one_case(actions=["locate(cup)", "pour_water(cup)"])
    report = evaluate_blind_cases(
        bundle,
        raw_predictions={"blind_case__v1": ["locate(cup)", "turn_on(cup)"]},
        final_predictions={"blind_case__v1": ["locate(cup)", "pour_water(cup)"]},
    )
    self.assertEqual(report["metrics"]["raw_exact_sequence_accuracy"], 0.0)
    self.assertEqual(report["metrics"]["final_exact_sequence_accuracy"], 1.0)

def test_wilson_interval_is_bounded_and_contains_observed_rate(self):
    lower, upper = wilson_interval(9, 10)
    self.assertLess(lower, 0.9)
    self.assertGreater(upper, 0.9)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation -v
```

Expected: FAIL because `tools.independent_blind_evaluation` does not exist.

- [ ] **Step 3: Implement the pure evaluator**

Use `score_predictions` twice, once for raw predictions and once for final predictions. Build scoring records with `task` set to `acceptance_task`, and never call a model, parser, lookup function, or constraint in this pure module. Include the frozen bundle audit and one per-case record containing instruction, expected actions, raw actions, final actions, exact flags, and whether final differs from raw.

- [ ] **Step 4: Run evaluator tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation -v
```

Expected: PASS with raw and final values independently asserted.

### Task 4: Checkpoint-based blind-test command and report

**Files:**
- Create: `evaluate_independent_blind_action_test.py`
- Modify: `tests/test_independent_blind_evaluation.py`

**Interfaces:**
- `run_evaluation(bundle_path: Path, checkpoint: Path, *, device: str, beam_width: int, max_actions: int) -> dict`.
- CLI defaults: frozen bundle `datasets/independent_blind_action_test_414.json`, report `datasets/independent_blind_action_test_414_report.json`, Markdown `docs/INDEPENDENT_BLIND_ACTION_EVALUATION_REPORT.md`.
- The runner loads the checkpoint once; for each case it resolves the instruction, runs `generate_for_instruction`, records raw actions, then calls `constrain_actions` only after raw generation and records final actions.

- [ ] **Step 1: Write failing test for post-generation constraint order**

```python
def test_runtime_runner_keeps_raw_generation_when_constraint_repairs_it():
    report = run_evaluation_with_fake_runtime(
        frozen_bundle_with_one_case(actions=["locate(reading_lamp)", "turn_off(reading_lamp)"]),
        generated=["locate(reading_lamp)", "turn_on(reading_lamp)"],
    )
    case = report["cases"][0]
    self.assertEqual(case["raw_actions"][-1], "turn_on(reading_lamp)")
    self.assertEqual(case["final_actions"][-1], "turn_off(reading_lamp)")
    self.assertTrue(case["constraint_applied"])
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation -v
```

Expected: FAIL because the runtime runner does not exist.

- [ ] **Step 3: Implement the runner and Markdown report**

Inject the resolver, generator, and constraint function into a small `evaluate_cases` helper so the test uses real evaluator logic with a deterministic fake generator. The production CLI supplies `resolve_manual_instruction`, `load_project_v2_model`, `generate_for_instruction`, and `constrain_actions`. If parsing or generation fails, retain an empty raw sequence and the error; do not look up expected actions before the raw generation call. The Markdown title must state whether it reports raw model or final system metrics and print both Wilson intervals.

- [ ] **Step 4: Run unit tests and command help**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation tests.test_blind_paraphrase_workflow -v
.\.venv\Scripts\python.exe evaluate_independent_blind_action_test.py --help
```

Expected: all tests PASS; help lists checkpoint, bundle, report, Markdown, device, beam width, max actions, and `--fail-below-raw-threshold`.

### Task 5: Update the project status with non-misleading wording

**Files:**
- Modify: `docs/CURRENT_PROJECT_STATUS_2026-07-04.md`
- Test: `tests/test_independent_blind_evaluation.py`

**Interfaces:**
- Status wording distinguishes current standard-instruction/final-system scores from the not-yet-filled independent 414-case raw-model blind test.

- [ ] **Step 1: Write a failing documentation guard**

```python
def test_status_does_not_claim_raw_open_language_accuracy_before_blind_test_exists(self):
    status = (ROOT / "docs/CURRENT_PROJECT_STATUS_2026-07-04.md").read_text(encoding="utf-8")
    self.assertIn("414", status)
    self.assertIn("原始模型", status)
    self.assertIn("最终系统", status)
    self.assertIn("尚未填写", status)
```

- [ ] **Step 2: Run the guard and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation -v
```

Expected: FAIL because the status has no 414-case workflow statement.

- [ ] **Step 3: Add the status section**

State that the current 100% result covers the formal standard benchmark/final constrained pipeline, while the 414-case independent raw-model blind test is pending human completion. Include the export and evaluation commands created above. Do not change historical figures or claim a raw open-language score.

- [ ] **Step 4: Run focused verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation tests.test_blind_paraphrase_workflow tests.test_action_constraints tests.test_action_constraint_regression -v
```

Expected: PASS.

## Plan self-review

- Spec coverage: Tasks 1–2 implement 414-case generation and freeze validation; Tasks 3–4 implement separated raw/final scoring and 95% intervals; Task 5 prevents misleading status claims.
- Placeholder scan: no deferred implementation steps; each task names files, interfaces, tests, commands, and expected outcomes.
- Type consistency: the bundle produced in Task 1 is consumed by Task 3; Task 4 produces `raw_predictions` and `final_predictions` accepted by Task 3.

## Execution note

This workspace currently has no usable Git repository, so tasks are verified by test output rather than committed during execution.
