# Hybrid Semantic Action Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a leak-checked Chinese semantic relation classifier and evaluate the frozen human-only retest exactly once after model selection is frozen.

**Architecture:** A data generator produces relation-labelled training phrases from the canonical 138 records, then rejects normalized phrase overlap against both blind bundles. A PyTorch character n-gram relation classifier predicts a canonical relation; the existing benchmark supplies its defined action sequence. Evaluation records the raw hybrid plan before any action constraints.

**Tech Stack:** Python 3.11, PyTorch, JSON, existing benchmark and Wilson-scoring helpers.

## Global Constraints

- Never read `instruction` or `actions` from `datasets/human_only_independent_action_retest_414.json` during training, validation, model selection or error analysis. Training may read only `datasets/human_only_final_instruction_hashes.json`, which contains irreversible normalized-text hashes created at freeze time.
- Do not train from `datasets/independent_blind_action_test_414.json`.
- Current scope is 15 tasks, 123 unique objects and 138 task-object operation relations.
- Report “hybrid semantic action parser” metrics separately from pure Transformer and constraint-corrected metrics.
- Do not claim ≥90% before a single frozen human-only evaluation reports raw exact sequence accuracy ≥0.90.

---

### Task 1: Generate leak-checked semantic training data

**Files:**
- Create: `tools/semantic_action_data.py`
- Create: `tests/test_semantic_action_data.py`
- Create at runtime: `datasets/semantic_action_train.json`, `datasets/semantic_action_dev.json`, `datasets/semantic_action_data_report.json`, `datasets/human_only_final_instruction_hashes.json`

**Interfaces:**
- `build_semantic_examples(records: Sequence[Mapping], variants_per_relation: int = 40) -> list[dict]`
- `write_instruction_hash_manifest(bundle_path: Path, output_path: Path) -> None`
- `assert_no_blind_overlap(examples: Sequence[Mapping], blind_hash_paths: Sequence[Path]) -> None`
- `split_by_relation(examples: Sequence[Mapping], dev_variant: int = 40) -> tuple[list[dict], list[dict]]`

- [ ] **Step 1: Write failing data-contract tests**

```python
def test_generator_creates_40_unique_examples_per_relation():
    records = [{"relation_key": "appliance_management::reading_lamp", "acceptance_task": "appliance_management", "object": "reading_lamp", "target": "none"}]
    rows = build_semantic_examples(records, variants_per_relation=40)
    assert len(rows) == 40
    assert len({row["instruction"] for row in rows}) == 40
    assert {row["relation_key"] for row in rows} == {"appliance_management::reading_lamp"}

def test_overlap_check_rejects_a_blind_instruction(tmp_path):
    path = tmp_path / "blind_hashes.json"
    path.write_text(json.dumps({"hashes": [normalized_instruction_hash("睡前把阅读灯关了")]}), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        assert_no_blind_overlap([{"instruction": "睡前把阅读灯关了"}], [path])
```

- [ ] **Step 2: Verify the test fails**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_semantic_action_data -v `

Expected: FAIL because the semantic data module is absent.

- [ ] **Step 3: Implement fixed phrase families and isolation check**

Generate forty deterministic variants per relation from object aliases, operation-specific templates, politeness prefixes and time/context suffixes. Store `id`, `relation_key`, `acceptance_task`, `object`, `target`, `instruction`, and `source="generated_semantic_train_v1"`. Normalize by stripping whitespace and casefolding; reject any duplicate within training data and any overlap with blind-instruction hashes.

`write_instruction_hash_manifest` is the one-time freeze utility: it reads the frozen bundle once, writes only SHA-256 hashes and count, then all training code accepts only the hash file. It must not write plaintext instructions.

- [ ] **Step 4: Generate train/dev files and verify their audit**

Run: ` .\.venv\Scripts\python.exe build_semantic_action_dataset.py --variants-per-relation 40 `

Expected: 5,520 examples total, 138 dev rows (one variant per relation), 5,382 train rows, a 414-hash final manifest, and zero blind overlap.

- [ ] **Step 5: Commit**

This workspace is not a Git repository. Record completion in this plan; no commit command is available.

### Task 2: Train a character n-gram relation classifier

**Files:**
- Create: `tools/semantic_action_model.py`
- Create: `train_semantic_action_parser.py`
- Create: `tests/test_semantic_action_model.py`
- Create at runtime: `models/semantic_action_parser_v1.pt`, `datasets/semantic_action_train_report.json`

**Interfaces:**
- `CharNgramRelationModel(vocab_size: int, relation_count: int) -> nn.Module`
- `encode_instruction(instruction: str, char_vocab: Mapping[str, int], ngram_size: int = 2) -> Tensor`
- `predict_relation(model, instruction, metadata) -> SemanticPrediction`

- [ ] **Step 1: Write failing model tests**

```python
def test_encoder_is_deterministic_and_retains_chinese_character_ngrams():
    vocab = build_char_vocab(["睡前把阅读灯关了"])
    first = encode_instruction("睡前把阅读灯关了", vocab)
    second = encode_instruction("睡前把阅读灯关了", vocab)
    assert torch.equal(first, second)
    assert first.numel() > 0

def test_prediction_returns_one_known_relation():
    prediction = SemanticPrediction("appliance_management::reading_lamp", 0.95, 0.80)
    assert prediction.relation_key == "appliance_management::reading_lamp"
    assert prediction.confidence > prediction.runner_up_confidence
```

- [ ] **Step 2: Verify the model tests fail**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_semantic_action_model -v `

Expected: FAIL because the model helpers do not exist.

- [ ] **Step 3: Implement and train**

Use a bag of character bigram IDs, `nn.EmbeddingBag(mode="mean")`, dropout 0.1 and one linear head over sorted relation keys. Train with cross entropy, fixed seed 20260720, batch size 64, maximum 30 epochs, early stopping patience 5 using only the generated development split. Save only model weights, char vocabulary, relation keys, seed, dataset SHA-256, threshold and development metrics.

- [ ] **Step 4: Verify training on the generated development split**

Run: ` .\.venv\Scripts\python.exe train_semantic_action_parser.py --epochs 30 --device cuda `

Expected: checkpoint and report created; report contains relation accuracy, raw planned action sequence accuracy, and clarification rate. This is a development result only.

- [ ] **Step 5: Commit**

This workspace is not a Git repository. Record completion in this plan; no commit command is available.

### Task 3: Run raw hybrid planning without constraints and preserve audit data

**Files:**
- Create: `tools/semantic_action_runtime.py`
- Create: `evaluate_hybrid_semantic_action_parser.py`
- Create: `tests/test_semantic_action_runtime.py`

**Interfaces:**
- `resolve_hybrid_instruction(instruction: str, checkpoint: Path, threshold: float) -> HybridResolution`
- `raw_actions_for_resolution(resolution: HybridResolution, benchmark: Sequence[Mapping]) -> list[str]`
- `evaluate_hybrid_bundle(bundle_path: Path, checkpoint: Path, *, allow_frozen_final: bool = False) -> dict`

- [ ] **Step 1: Write failing runtime tests**

```python
def test_raw_actions_are_lookup_from_predicted_relation_not_constraints():
    resolution = HybridResolution("cleaning::carpet", 0.98, 0.01, False)
    actions = raw_actions_for_resolution(resolution, [{"relation_key": "cleaning::carpet", "actions": ["locate(carpet)", "clean(carpet)"]}])
    assert actions == ["locate(carpet)", "clean(carpet)"]

def test_low_confidence_returns_clarification_without_actions():
    resolution = HybridResolution(None, 0.40, 0.38, True)
    assert resolution.needs_clarification is True
    assert raw_actions_for_resolution(resolution, []) == []
```

- [ ] **Step 2: Verify runtime tests fail**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_semantic_action_runtime -v `

Expected: FAIL because the runtime module is absent.

- [ ] **Step 3: Implement restricted evaluation modes**

Default evaluator accepts only the generated development JSON and records raw relation/action prediction, confidence and clarification. Add explicit `--run-frozen-human-final` flag; this is the only code path permitted to read the frozen human-only bundle and must write a one-time final report with bundle fingerprint, checkpoint fingerprint and timestamp.

- [ ] **Step 4: Verify development evaluation**

Run: ` .\.venv\Scripts\python.exe evaluate_hybrid_semantic_action_parser.py --bundle datasets/semantic_action_dev.json `

Expected: report contains raw relation accuracy, raw exact action sequence accuracy, clarification rate and no constrained score substitution.

- [ ] **Step 5: Commit**

This workspace is not a Git repository. Record completion in this plan; no commit command is available.

### Task 4: Freeze configuration, run final test once, and update status honestly

**Files:**
- Modify: `docs/CURRENT_PROJECT_STATUS_2026-07-04.md`
- Create at runtime: `datasets/human_only_hybrid_final_report.json`, `docs/HUMAN_ONLY_HYBRID_FINAL_REPORT.md`
- Modify: `tests/test_independent_blind_evaluation.py`

**Interfaces:**
- Consumes: frozen human bundle, saved checkpoint and fixed threshold.
- Produces: immutable final report with exact sequence score and Wilson interval.

- [ ] **Step 1: Write failing score-separation test**

```python
def test_final_report_labels_hybrid_raw_score_without_constraint_substitution():
    report = {"metrics": {"raw_exact_sequence_accuracy": 0.91, "final_exact_sequence_accuracy": 0.95}, "architecture": "hybrid_semantic_action_parser_v1"}
    markdown = hybrid_markdown_report(report)
    assert "混合语义动作解析系统" in markdown
    assert "原始" in markdown
    assert "受约束" in markdown
```

- [ ] **Step 2: Verify it fails**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation -v `

Expected: FAIL because hybrid report rendering is absent.

- [ ] **Step 3: Run the frozen final evaluation once**

Run: ` .\.venv\Scripts\python.exe evaluate_hybrid_semantic_action_parser.py --bundle datasets/human_only_independent_action_retest_414.json --run-frozen-human-final `

Expected: exactly one report writes raw exact action sequence accuracy, final constrained accuracy, Wilson intervals, relation accuracy, clarification rate and fingerprints. Do not re-run after inspecting errors.

- [ ] **Step 4: Update status from actual output**

If raw exact ≥0.90, write that the hybrid semantic action parser passes the specified threshold. Otherwise write the actual score and failure distribution, without substituting the final constrained score.

- [ ] **Step 5: Focused verification**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_semantic_action_data tests.test_semantic_action_model tests.test_semantic_action_runtime tests.test_independent_blind_evaluation -v `

Run: ` .\.venv\Scripts\python.exe -m compileall -q tools build_semantic_action_dataset.py train_semantic_action_parser.py evaluate_hybrid_semantic_action_parser.py `

Expected: tests PASS and compilation exits with code 0.

- [ ] **Step 6: Commit**

This workspace is not a Git repository. Record completion in this plan; no commit command is available.
