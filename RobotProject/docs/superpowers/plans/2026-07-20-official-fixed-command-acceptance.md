# Official Fixed-command Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a frozen 552-case fixed-command acceptance suite and a third-party-reproducible raw action-sequence report.

**Architecture:** Generate four separate fixed instruction variants for each of the 138 catalog relations, then hash-check them against all training, development, assistant-selftest, and prior-human-final manifests. The official evaluator uses the selected augmented semantic model, records raw actions before applying the canonical consistency check, scores exact action sequences, and writes immutable provenance hashes.

**Tech Stack:** Python 3.11, PyTorch, JSON, SHA-256, unittest.

## Global Constraints

- The 552 official cases must not overlap by normalized instruction hash with augmented train/dev, assistant self-test, or the old human-final hash manifest.
- Official cases never enter training or development data.
- The report must distinguish raw exact-sequence accuracy from constrained accuracy.
- The old 414-case human final is not opened, changed, or rerun.
- A 90% pass requires at least 497 of 552 raw exact sequences.

---

### Task 1: Generate and freeze four official expressions per relation

**Files:**
- Create: tools/official_acceptance_data.py
- Create: build_official_fixed_command_acceptance.py
- Create: tests/test_official_acceptance_data.py
- Create at runtime: datasets/official_fixed_command_acceptance_552.json
- Create at runtime: datasets/official_fixed_command_acceptance_hashes.json

**Interfaces:**
- Consumes: the 138-record acceptance catalog and hash manifests/data files for exclusion only.
- Produces: build_official_acceptance(records, exclusion_rows, final_hash_manifest) -> dict.
- Produces: 552 cases with id, relation_key, instruction, expected_actions, and an evidence-level declaration.

- [ ] **Step 1: Write a failing coverage test**

~~~python
def test_official_acceptance_has_four_cases_per_relation():
    bundle = build_official_acceptance([reading_lamp_record()], [], empty_hash_manifest)
    self.assertEqual(len(bundle["cases"]), 4)
    self.assertEqual(bundle["minimum_raw_passes_at_90_percent"], 4)
~~~

- [ ] **Step 2: Run the test before implementation**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_official_acceptance_data -v

Expected: import failure for tools.official_acceptance_data.

- [ ] **Step 3: Implement four distinct official template families**

Implement a standard-command template and three fixed daily-language templates. Templates must retain explicit operations, include task context for genuinely duplicated object/operation pairs, and reject a duplicate normalized instruction. Build expected actions only from the canonical catalog relation.

The bundle header must include:

~~~json
{
  "format": "official_fixed_command_acceptance_v1",
  "cases": 552,
  "relations": 138,
  "minimum_raw_passes_at_90_percent": 497,
  "evidence_scope": "fixed_command_acceptance"
}
~~~

- [ ] **Step 4: Build and verify the frozen artifacts**

Run: .\.venv\Scripts\python.exe build_official_fixed_command_acceptance.py

Expected: 552 cases; no hash overlap exception; separate hash manifest written.

### Task 2: Implement reproducible raw and constrained scoring

**Files:**
- Create: evaluate_official_fixed_command_acceptance.py
- Create: tests/test_official_fixed_command_acceptance.py
- Create at runtime: models/semantic_action_augmented/official_fixed_command_acceptance_report.json
- Create at runtime: docs/OFFICIAL_FIXED_COMMAND_ACCEPTANCE_REPORT.md

**Interfaces:**
- Consumes: official frozen bundle, selected augmented model, current canonical catalog.
- Produces: evaluate_bundle(bundle, runtime, catalog) -> dict.
- Produces: relation accuracy, raw exact-sequence accuracy, constrained accuracy, pass count, and provenance hashes.

- [ ] **Step 1: Write a failing exact-sequence scoring test**

~~~python
def test_exact_sequence_counts_only_identical_ordered_actions():
    report = evaluate_predictions(
        [{"id": "a", "expected_actions": ["locate(cup)", "wash(cup)"]}],
        {"a": ["wash(cup)", "locate(cup)"]},
    )
    self.assertEqual(report["raw_passes"], 0)
~~~

- [ ] **Step 2: Run it before implementation**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_official_fixed_command_acceptance -v

Expected: import failure for the official evaluator.

- [ ] **Step 3: Implement report creation**

For every case, invoke HybridSemanticActionParser.generate_raw_actions(), record the prediction before calling constrain_semantic_actions(), and compare raw actions with expected_actions by exact ordered-list equality. Write a per-case row with instruction, expected actions, predicted relation, confidence, raw actions, final actions, and pass state.

Write report fields:

~~~json
{
  "cases": 552,
  "raw_passes": 0,
  "raw_exact_sequence_accuracy": 0.0,
  "minimum_raw_passes_at_90_percent": 497,
  "raw_threshold_passed": false,
  "bundle_sha256": "<sha256>",
  "model_sha256": "<sha256>",
  "catalog_sha256": "<sha256>"
}
~~~

- [ ] **Step 4: Verify evaluator tests**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_official_acceptance_data tests.test_official_fixed_command_acceptance -v

Expected: all tests pass.

### Task 3: Run official acceptance and publish verification evidence

**Files:**
- Modify: docs/CURRENT_PROJECT_STATUS_2026-07-04.md
- Create at runtime: official JSON/Markdown report from Task 2.

**Interfaces:**
- Consumes: only the frozen official acceptance bundle, selected model, and catalog.
- Produces: final reproducibility command and a scope-limited status entry.

- [ ] **Step 1: Compile and run the official acceptance command**

Run: .\.venv\Scripts\python.exe evaluate_official_fixed_command_acceptance.py

Expected: report prints cases, raw passes, raw accuracy, threshold result, and report paths.

- [ ] **Step 2: Update status document**

Add the frozen bundle path, model/report hashes, raw pass count, raw exact-sequence rate, 497/552 threshold, and the statement that this validates the fixed accepted command scope only.

- [ ] **Step 3: Final verification**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_official_acceptance_data tests.test_official_fixed_command_acceptance tests.test_hybrid_semantic_action_runtime -v

Run: .\.venv\Scripts\python.exe -m compileall -q tools\official_acceptance_data.py build_official_fixed_command_acceptance.py evaluate_official_fixed_command_acceptance.py

Expected: zero failures and successful compilation.

## Plan Self-review

- Coverage: Task 1 creates isolated fixed cases; Task 2 records strict raw scoring and hashes; Task 3 executes and documents third-party evidence.
- Data boundary: no task opens the old human final; all cross-dataset comparisons use hashes or already-authorized non-final files.
- Metric boundary: raw ordered action equality is the formal pass metric; constrained output is supplementary.
