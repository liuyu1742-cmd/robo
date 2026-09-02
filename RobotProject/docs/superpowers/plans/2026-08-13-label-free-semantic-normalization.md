# Label-Free Semantic Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one shared label-free Chinese semantic normalizer and train/evaluate a global 254-class model with overall, scene, and object top-1 all at least 90%.

**Architecture:** A focused normalizer converts surface synonyms and common action/state language into shared standard Chinese concepts without label IDs, object IDs, scene compositions, or complete intents. The same function transforms train, dev, and runtime inputs before the existing global TF-IDF model. Reports preserve raw model-only and normalized-model metrics separately.

**Tech Stack:** Python 3.11, PyTorch, unittest, existing project synonym/task resources.

## Global Constraints

- No feature or resource may contain a catalogue label ID or map a scene object combination to a scene answer.
- No complete dev text, dev scene goal, or dev-only alias may enter model vocabulary or normalization resources.
- Normalization never returns a label and uses the same deterministic rules for train/dev/runtime.
- The normalized model scores all 254 classes globally.

---

### Task 1: Label-Free Normalizer

**Files:**
- Create: `tools/semantic_text_normalization.py`
- Modify: `tests/test_detailed_action_semantic_data.py`

**Interfaces:**
- Produces: `normalize_for_semantic_model(text: str) -> str` and `normalization_resource_terms() -> tuple[str, ...]`.

- [ ] Write tests asserting shared synonym normalization, absence of labels/dev texts, and deterministic output.
- [ ] Run the focused tests and confirm missing API failures.
- [ ] Implement longest-match object concept normalization plus shared action/state normalization using only fixed project-level resources.
- [ ] Run the focused tests and confirm green.

### Task 2: Normalized Global Training and Evaluation

**Files:**
- Modify: `tools/semantic_action_model.py`
- Modify: `tools/detailed_action_semantic_evaluation.py`
- Modify: `train_detailed_action_semantic_parser.py`
- Modify: `tests/test_detailed_action_semantic_data.py`

**Interfaces:**
- Consumes: `normalize_for_semantic_model`.
- Produces: raw model-only metrics and normalized-model metrics from the same 254-class scorer.

- [ ] Add failing tests for checkpoint normalization metadata and three-group normalized acceptance.
- [ ] Apply the same normalizer before vocabulary fitting, centroid fitting, evaluation, and runtime prediction.
- [ ] Train experiments, preserving raw baseline and normalized results separately.
- [ ] Retain only a configuration with overall/scene/object top-1 each at least 90%; otherwise report exact blocker.

### Task 3: Artifacts and Verification

**Files:**
- Regenerate: `models/detailed_action_semantic/best.pt`
- Regenerate: `models/detailed_action_semantic/training_report.json`
- Modify: `.superpowers/sdd/task-2-report.md`

- [ ] Rebuild deterministic artifacts with seed 20260813.
- [ ] Run Task 2, v1 compatibility, Task 1 regressions, and compile checks.
- [ ] Independently retrain and compare vocabulary, tensors, and metrics.
- [ ] Document model-only and normalized-model metrics without conflation.
