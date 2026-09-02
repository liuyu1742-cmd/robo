# Semantic Parser Augmentation and Assistant Self-test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Broaden trainable natural-language coverage without using the locked human final test, then run a separately labelled assistant-generated self-test.

**Architecture:** A dedicated augmentation module creates operation-explicit Chinese phrases from the existing 138 relation records. It partitions distinct phrase families into training, development, and assistant self-test, rejects all hash overlaps with the locked human-final manifest, and reuses the existing character n-gram relation classifier and canonical action planner. The assistant self-test is generated only after the new checkpoint is written and is reported as development evidence, never as human evidence.

**Tech Stack:** Python 3.11, PyTorch, JSON, SHA-256, unittest.

## Global Constraints

- Never open, copy, train on, tune on, or rerun `datasets/human_only_independent_action_retest_414.json`.
- Use only `datasets/human_only_final_instruction_hashes.json` for exclusion checks against the old human final.
- Every labelled phrase must state an operation; do not use generic “处理一下/弄一下” phrasing.
- Keep training, development, and assistant self-test phrase hashes mutually disjoint.
- Assistant self-test results are not independent human-test results and must never support a “human accuracy >90%” claim.

---

### Task 1: Separate augmented phrase families and enforce hash isolation

**Files:**
- Create: `tools/semantic_action_augmentation.py`
- Create: `tests/test_semantic_action_augmentation.py`
- Modify: `tools/semantic_action_data.py`

**Interfaces:**
- Consumes: benchmark records with `relation_key`, `acceptance_task`, `object`, `target`, and `actions`.
- Consumes: `normalized_instruction_hash()` and `assert_no_blind_overlap()` from `tools.semantic_action_data`.
- Produces: `build_augmented_split(records, split: str, variants_per_relation: int) -> list[dict]`.
- Produces: `assert_disjoint_splits(*splits: Sequence[Mapping]) -> None`.

- [ ] **Step 1: Write failing split-isolation tests**

```python
def test_augmented_splits_are_mutually_disjoint():
    records = [single_reading_lamp_record()]
    train = build_augmented_split(records, "train", 4)
    dev = build_augmented_split(records, "dev", 2)
    assistant = build_augmented_split(records, "assistant_selftest", 2)
    assert_disjoint_splits(train, dev, assistant)
    self.assertEqual({row["split"] for row in assistant}, {"assistant_selftest"})

def test_assistant_selftest_contains_operation_language():
    row = build_augmented_split([single_reading_lamp_record()], "assistant_selftest", 1)[0]
    self.assertNotIn("处理", row["instruction"])
    self.assertNotIn("弄一下", row["instruction"])
```

- [ ] **Step 2: Run the test and verify it fails before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_semantic_action_augmentation -v
```

Expected: import failure because `tools.semantic_action_augmentation` does not yet exist.

- [ ] **Step 3: Implement explicit, non-overlapping phrase pools**

Create `tools/semantic_action_augmentation.py` with three separate template collections. The train templates use short requests, the dev templates use time/cause and word-order variations, and the assistant self-test templates use distinct daily-language forms. For a shared object with the same first operation in multiple task families, append a concise task-context clause exactly as `build_semantic_examples()` already does.

The returned row format must be:

```python
{
    "id": "augmented_<relation>__<split>__v<index>",
    "relation_key": relation,
    "acceptance_task": task,
    "object": obj,
    "target": target,
    "instruction": phrase,
    "source": "augmented_semantic_<split>_v1",
    "split": split,
    "variant": index + 1,
}
```

Implement `assert_disjoint_splits()` by hashing normalized instructions and raising `ValueError("overlap between semantic data splits")` when a hash appears in more than one split.

- [ ] **Step 4: Run the split tests and existing hash tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_semantic_action_augmentation tests.test_semantic_action_data -v
```

Expected: all tests pass.

### Task 2: Build expanded training/development artifacts without human-final access

**Files:**
- Create: `build_augmented_semantic_action_dataset.py`
- Create: `tests/test_build_augmented_semantic_action_dataset.py`
- Create at runtime: `datasets/semantic_action_augmented_train.json`
- Create at runtime: `datasets/semantic_action_augmented_dev.json`
- Create at runtime: `datasets/semantic_action_augmented_data_report.json`

**Interfaces:**
- Consumes: `build_augmented_split()`, `assert_disjoint_splits()`, `assert_no_blind_overlap()`.
- Consumes: only the hash manifest `datasets/human_only_final_instruction_hashes.json` for old-final exclusion.
- Produces: augmented training and development JSON files; does not create assistant self-test files in this task.

- [ ] **Step 1: Write a failing builder test with temporary files**

```python
def test_builder_marks_old_human_final_as_hash_only(tmp_path):
    report = build_dataset(catalog_path, hash_manifest_path, train_path, dev_path)
    self.assertTrue(report["old_human_final_hash_only"])
    self.assertGreater(report["train_examples"], 0)
    self.assertGreater(report["dev_examples"], 0)
```

- [ ] **Step 2: Run it before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_build_augmented_semantic_action_dataset -v
```

Expected: import failure for `build_augmented_semantic_action_dataset`.

- [ ] **Step 3: Implement the builder**

Implement `build_dataset(catalog_path, hash_manifest_path, train_output, dev_output) -> dict`. Generate at least 32 augmented train phrases and 8 dev phrases per relation. Hash-check each split against the human-final manifest and against each other. Write only train/dev phrase text to their corresponding files. Write a report with counts, relation coverage, and `old_human_final_hash_only: true`.

- [ ] **Step 4: Build and verify the actual artifacts**

Run:

```powershell
.\.venv\Scripts\python.exe build_augmented_semantic_action_dataset.py
```

Expected: 138 relations covered in both outputs, no overlap exception, and no opening of the old final bundle.

### Task 3: Train and select the expanded semantic classifier on the new development split

**Files:**
- Modify: `train_semantic_action_parser.py`
- Create: `models/semantic_action_augmented/best.pt`
- Create: `models/semantic_action_augmented/dev_report.json`
- Modify: `evaluate_hybrid_semantic_development.py`
- Create: `models/semantic_action_augmented/development_report.json`

**Interfaces:**
- Consumes: augmented train/dev JSON only.
- Produces: checkpoint and reports with `blind_final_data_read: false` / `frozen_human_final_data_read: false`.

- [ ] **Step 1: Write a failing argument test**

```python
def test_train_cli_accepts_explicit_train_dev_and_model_paths():
    args = parse_args([
        "--train", "train.json", "--dev", "dev.json",
        "--model-output", "out.pt", "--report-output", "report.json",
    ])
    self.assertEqual(args.model_output, Path("out.pt"))
```

- [ ] **Step 2: Run it before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_train_semantic_action_parser -v
```

Expected: failing import or missing `parse_args(argv)` support.

- [ ] **Step 3: Make CLI parsing testable and train to a new output directory**

Refactor `parse_args()` to accept an optional argument list. Do not alter the old checkpoint. Train with:

```powershell
.\.venv\Scripts\python.exe train_semantic_action_parser.py `
  --train datasets\semantic_action_augmented_train.json `
  --dev datasets\semantic_action_augmented_dev.json `
  --model-output models\semantic_action_augmented\best.pt `
  --report-output models\semantic_action_augmented\dev_report.json `
  --epochs 80
```

- [ ] **Step 4: Run raw and constrained development evaluation**

Run:

```powershell
.\.venv\Scripts\python.exe evaluate_hybrid_semantic_development.py `
  --dev datasets\semantic_action_augmented_dev.json `
  --model models\semantic_action_augmented\best.pt `
  --report models\semantic_action_augmented\development_report.json
```

Expected: JSON contains relation, raw sequence, final sequence, and confidence metrics, with `frozen_human_final_data_read: false`.

### Task 4: Freeze an assistant-authored self-test only after model selection

**Files:**
- Create: `build_assistant_semantic_selftest.py`
- Create: `tests/test_assistant_semantic_selftest.py`
- Create at runtime: `datasets/assistant_semantic_selftest_138.json`
- Create at runtime: `datasets/assistant_semantic_selftest_hashes.json`

**Interfaces:**
- Consumes: selected augmented model path only as provenance; it does not consume model predictions while writing test text.
- Consumes: catalog records and the old human-final hash manifest.
- Produces: one assistant-authored instruction per relation and a hash-only manifest.

- [ ] **Step 1: Write a failing coverage and provenance test**

```python
def test_selftest_has_one_assistant_authored_case_per_relation():
    bundle = build_assistant_selftest(catalog, human_final_hashes)
    self.assertEqual(len(bundle["cases"]), 138)
    self.assertEqual(bundle["evidence_level"], "assistant_generated_development_selftest")
```

- [ ] **Step 2: Run it before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_assistant_semantic_selftest -v
```

Expected: import failure because the self-test builder does not exist.

- [ ] **Step 3: Implement and freeze the assistant self-test**

Generate exactly one phrase for each relation from the assistant-selftest template family. Assert it differs by hash from train, dev, and the old human final. Store expected relation and canonical actions only inside this assistant-development artifact. Write metadata:

```json
{
  "evidence_level": "assistant_generated_development_selftest",
  "not_independent_human_evidence": true,
  "cases": 138
}
```

- [ ] **Step 4: Run the builder once after the augmented model is selected**

Run:

```powershell
.\.venv\Scripts\python.exe build_assistant_semantic_selftest.py
```

Expected: 138 cases, no train/dev/old-final hash overlaps, and an explicit non-human evidence label.

### Task 5: Evaluate the assistant self-test and update project status

**Files:**
- Create: `evaluate_assistant_semantic_selftest.py`
- Create: `models/semantic_action_augmented/assistant_selftest_report.json`
- Create: `docs/ASSISTANT_SEMANTIC_SELFTEST_REPORT.md`
- Modify: `docs/CURRENT_PROJECT_STATUS_2026-07-04.md`

**Interfaces:**
- Consumes: frozen assistant self-test, selected augmented checkpoint, canonical catalog.
- Produces: raw/final sequence scores and an explicit evidence disclaimer.

- [ ] **Step 1: Write a failing report-label test**

```python
def test_markdown_never_labels_assistant_selftest_as_human_blind_test():
    text = markdown_report(sample_report())
    self.assertIn("助手生成自测", text)
    self.assertIn("不能替代独立人工终测", text)
```

- [ ] **Step 2: Run it before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_assistant_semantic_selftest_evaluation -v
```

Expected: import failure because the evaluator does not exist.

- [ ] **Step 3: Implement raw/final scoring**

Reuse `HybridSemanticActionParser`, `canonical_actions_for_relation`, and `constrain_semantic_actions`. Report relation accuracy, raw exact sequence accuracy, final exact sequence accuracy, low-confidence cases, model SHA-256, and self-test SHA-256. Do not open the old human-final bundle.

- [ ] **Step 4: Run tests, evaluation, and status update**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_semantic_action_augmentation `
  tests.test_build_augmented_semantic_action_dataset `
  tests.test_train_semantic_action_parser `
  tests.test_assistant_semantic_selftest `
  tests.test_assistant_semantic_selftest_evaluation -v
.\.venv\Scripts\python.exe evaluate_assistant_semantic_selftest.py
```

Expected: all tests pass, report identifies itself as assistant-generated self-test, and the status document records it as development evidence only.

## Plan Self-review

- Spec coverage: Tasks 1–2 implement isolated data, Task 3 trains and validates, Task 4 freezes the assistant self-test, and Task 5 scores it and documents the evidence limit.
- Data separation: every task uses the existing human-final hash manifest only; no task opens the locked human-final bundle.
- Type consistency: all generated rows retain the existing `relation_key` and canonical catalog fields consumed by `HybridSemanticActionParser`.
- Scope: this plan improves the parser and its internal evidence only; it intentionally does not change the one-time human-final result or make a new human-performance claim.
