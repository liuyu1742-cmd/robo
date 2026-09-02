# 独立口语数据集与 BGE 离线编码器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 4,964 条独立标注口语数据，下载并离线部署 `BAAI/bge-small-zh-v1.5`，训练新的细分动作语义模型，并通过一次性 independent test 与 Blind v5 验证真实泛化能力。

**Architecture:** 统一 254 标签目录保持不变。三个互不复用上下文的标注任务分别生成 train/dev/independent test；本地 BGE 编码器配合意图头、执行门控头和监督对比损失训练。`DetailedActionParser` 保持公开接口不变，默认加载 BGE checkpoint，候选约束与模型评分共同路由；字符模型仅作为显式回退。模型冻结后才运行 independent test，代码冻结后再由新代理创建 Blind v5。

**Tech Stack:** Python 3.11、PyTorch 2.7.1、Transformers 4.40.1、Hugging Face Hub、safetensors、CUDA 12.6、unittest、JSON、SHA-256。

## Global Constraints

- 数据固定为 4,964 条：4,064 条标签正例 + 900 条安全/边界样本。
- 254 标签不增删、不改语义：49 scene + 205 object。
- train/dev/independent test 的标注作者必须隔离；训练实现者不得读取 independent test 原文。
- 禁止读取 Blind v2/v3/v4 原文生成数据；禁止把完整测试命令映射到标签。
- 模型只从本地目录加载，运行时必须 `local_files_only=True`，不得自动联网。
- 只下载官方 safetensors、配置、tokenizer 与许可证；固定 40 位 revision，记录逐文件 SHA-256。
- Independent test 与 Blind v5 各只运行一次；未达标则停止并如实报告。
- 通过门槛：scene/object/overall top-1 各 >=95%，九类安全/边界各 100%，拒识无动作泄漏。
- 旧入口继续禁止细分动作；新入口计时边界不变。
- 当前目录不是有效 Git 仓库，所有计划中的提交步骤以报告文件和 SHA manifest 替代，禁止伪造 commit。

---

### Task 1: 建立 v2 数据 schema、构建器与隔离门禁

**Files:**
- Create: `tools/detailed_action_spoken_dataset.py`
- Create: `tests/test_detailed_action_spoken_dataset.py`
- Create: `datasets/detailed_action_spoken_v2/annotation_guidelines.md`
- Consume: `meta/detailed_action_entry_catalog.json`

**Interfaces:**
- Produces: `validate_split(records, split, catalog) -> list[str]`
- Produces: `validate_dataset(train, dev, test, catalog) -> list[str]`
- Produces: `write_dataset(train, dev, test, output_dir) -> dict`
- Record keys: `sample_id`, `text`, `label_id`, `result_type`, `annotation_source`, `object_mentions`, `operation_evidence`, `safety_class`, `expected_executable`。

- [ ] **Step 1: 写 schema 与隔离失败测试**

```python
def test_rejects_cross_split_text_skeleton_and_label_conflicts():
    errors = validate_dataset(train_fixture(), dev_fixture_with_leak(), test_fixture(), catalog())
    self.assertTrue(any("cross-split" in error for error in errors))

def test_requires_exact_label_and_safety_counts():
    errors = validate_dataset(incomplete_train(), incomplete_dev(), incomplete_test(), catalog())
    self.assertTrue(any("254" in error or "count" in error for error in errors))
```

- [ ] **Step 2: 运行红灯**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_spoken_dataset -v`

Expected: FAIL because module does not exist.

- [ ] **Step 3: 实现严格 schema 和跨 split 门禁**

实现规范化全文唯一性、标签合法性、split/source 一致性、正例操作证据、拒识无标签、九类安全计数、长 n-gram 与表达骨架隔离。门禁错误必须包含 sample ID 和冲突 split。

- [ ] **Step 4: 运行绿灯与旧目录回归**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_spoken_dataset tests.test_detailed_action_catalog -v`

Expected: PASS.

---

### Task 2: 独立标注 train split

**Files:**
- Create: `datasets/detailed_action_spoken_v2/train.json`
- Create: `datasets/detailed_action_spoken_v2/train_annotation_report.md`
- Consume: catalog、annotation guidelines；禁止读取 dev/test、Blind v2-v4。

**Interfaces:**
- Produces 3,080 records: 2,540 label positives + 540 safety/boundary records（9 类 × 60）。

- [ ] **Step 1: 按标签逐项标注 10 条训练正例**

每个标签覆盖不同语序、动词、目的、状态、指代；object 必须含操作证据，scene 必须表达整体目的。

- [ ] **Step 2: 标注 540 条训练安全/边界样本**

每类 60，正确混合可执行边界正例和应拒识负例，设置 `expected_executable`。

- [ ] **Step 3: 运行 split 校验**

Run: `.\.venv\Scripts\python.exe -m tools.detailed_action_spoken_dataset --validate-split train datasets/detailed_action_spoken_v2/train.json`

Expected: 3,080 valid, 254/254 labels covered, 0 errors.

- [ ] **Step 4: 输出逐标签标注报告与文件 SHA**

报告列每标签数量、九类安全数量、规范化重复数和 SHA-256。

---

### Task 3: 独立标注 dev split

**Files:**
- Create: `datasets/detailed_action_spoken_v2/dev.json`
- Create: `datasets/detailed_action_spoken_v2/dev_annotation_report.md`
- Consume: catalog、annotation guidelines；禁止读取 train/test、Blind v2-v4。

**Interfaces:**
- Produces 942 records: 762 label positives + 180 safety/boundary records（9 类 × 20）。

- [ ] **Step 1: 每标签独立写 3 条 dev 正例**

禁止机械改写 canonical command，必须包含独立口语结构。

- [ ] **Step 2: 每安全类独立写 20 条 dev 样本**

- [ ] **Step 3: 运行 dev split 校验并输出报告**

Run: `.\.venv\Scripts\python.exe -m tools.detailed_action_spoken_dataset --validate-split dev datasets/detailed_action_spoken_v2/dev.json`

Expected: 942 valid, 254/254 labels covered, 0 errors.

---

### Task 4: 独立标注并封存 independent test

**Files:**
- Create: `datasets/detailed_action_spoken_v2/independent_test.json`
- Create: `datasets/detailed_action_spoken_v2/independent_test.sha256`
- Create: `datasets/detailed_action_spoken_v2/independent_test_annotation_report.md`
- Consume: catalog、annotation guidelines；禁止读取 train/dev、runtime、模型、Blind v2-v4。

**Interfaces:**
- Produces 942 records: 762 label positives + 180 safety/boundary records。

- [ ] **Step 1: 每标签独立写 3 条 test 正例并逐项核对语义**

- [ ] **Step 2: 每安全类独立写 20 条 test 样本**

- [ ] **Step 3: 校验 schema、覆盖、唯一性并永久冻结**

先计算 canonical cases SHA，再写完整文件 SHA sidecar；冻结后作者和实现者不得修改。

- [ ] **Step 4: 记录封存证据**

报告记录作者、冻结时间、catalog SHA、cases SHA、file SHA 和 0 train/dev context 声明。

---

### Task 5: 合并审计 4,964 条数据并生成 manifest

**Files:**
- Create: `datasets/detailed_action_spoken_v2/manifest.json`
- Create: `datasets/detailed_action_spoken_v2/dataset_audit_report.md`
- Modify: `tests/test_detailed_action_spoken_dataset.py`

**Interfaces:**
- Consumes the three frozen splits.
- Produces a manifest with counts, catalog SHA, split SHA, annotation sources, safety counts, and isolation audit.

- [ ] **Step 1: 写磁盘产物逐行一致和篡改失败测试**

```python
def test_disk_dataset_is_exactly_4964_and_isolation_clean():
    dataset = load_dataset(DATASET_DIR)
    self.assertEqual(sum(len(rows) for rows in dataset.values()), 4964)
    self.assertEqual(validate_dataset(**dataset, catalog=catalog()), [])
```

- [ ] **Step 2: 运行红灯并记录实际跨 split 冲突**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_spoken_dataset -v`

Expected: FAIL until independent annotations are reconciled by their original authors.

- [ ] **Step 3: 把冲突清单分别退回原标注作者修订**

训练实现者不得自行重写 dev/test；每次修订更新作者报告和 SHA。

- [ ] **Step 4: 生成 manifest 并运行绿灯**

Expected: exactly train 3,080 / dev 942 / test 942; total 4,964; 0 isolation errors.

---

### Task 6: 安全下载并固定 BGE 模型

**Files:**
- Create: `tools/download_detailed_action_encoder.py`
- Create: `tests/test_detailed_action_encoder_download.py`
- Generate: `models/pretrained/BAAI_bge-small-zh-v1.5/`
- Generate: `models/pretrained/BAAI_bge-small-zh-v1.5/manifest.json`
- Generate: `models/pretrained/BAAI_bge-small-zh-v1.5/LICENSE`

**Interfaces:**
- Produces: `resolve_revision(repo_id) -> str`
- Produces: `download_encoder(repo_id, revision, destination) -> dict`
- Allowed files: safetensors, JSON configs, tokenizer/vocab, pooling config, README, LICENSE.

- [ ] **Step 1: 写 staging、allowlist、40位revision和SHA失败测试**

网络调用用受控 fake API；测试断言 pickle/bin 不进入正式目录、校验失败不覆盖现有模型。

- [ ] **Step 2: 运行红灯**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_encoder_download -v`

- [ ] **Step 3: 实现下载器**

使用 `HfApi().model_info(repo_id).sha` 解析 40 位 revision，再用 `snapshot_download(..., revision=sha, allow_patterns=...)` 下载到 staging。验证 `model.safetensors`、tokenizer/config 完整后生成 manifest，再移动到正式目录。

- [ ] **Step 4: 请求网络权限并执行真实下载**

Run: `.\.venv\Scripts\python.exe tools\download_detailed_action_encoder.py --repo BAAI/bge-small-zh-v1.5 --output models/pretrained/BAAI_bge-small-zh-v1.5`

Expected: official revision pinned; safetensors and tokenizer present; manifest SHA verification PASS.

- [ ] **Step 5: 离线加载 smoke**

Run with offline environment: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and `local_files_only=True`.

Expected: tokenizer/model load on CPU and encode Chinese text without network.

---

### Task 7: 实现 BGE 双头模型与训练

**Files:**
- Create: `tools/detailed_action_bge_model.py`
- Create: `train_detailed_action_bge.py`
- Create: `tests/test_detailed_action_bge_model.py`
- Generate: `models/detailed_action_bge/best.pt`
- Generate: `models/detailed_action_bge/training_report.json`

**Interfaces:**
- Produces: `DetailedActionBgeClassifier.forward(input_ids, attention_mask) -> dict`
- Produces: `load_detailed_action_bge(checkpoint, encoder_dir, device) -> tuple`
- Checkpoint stores format, encoder revision/SHA, catalog SHA, dataset manifest SHA, ordered labels, thresholds, state dict, metrics.

- [ ] **Step 1: 写 pooling、双头、checkpoint审计和离线加载失败测试**

- [ ] **Step 2: 运行红灯**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_bge_model -v`

- [ ] **Step 3: 实现冻结 encoder 的最小双头模型**

用 attention-mask mean pooling + L2 normalization；意图头 254 类，门控头 2 类。

- [ ] **Step 4: Stage 1 训练分类头**

Run: `.\.venv\Scripts\python.exe train_detailed_action_bge.py --stage frozen --seed 20260814 --encoder models/pretrained/BAAI_bge-small-zh-v1.5`

Expected: training loss decreases; dev metrics and confusion written.

- [ ] **Step 5: Stage 2 解冻最后两层联合训练**

Run: `.\.venv\Scripts\python.exe train_detailed_action_bge.py --stage unfreeze-last-2 --seed 20260814 --resume models/detailed_action_bge/stage1.pt`

Expected: select best checkpoint using dev only; report scene/object/overall and nine safety classes.

- [ ] **Step 6: 校准阈值并验证可复现性**

同 seed 重跑短验证，标签顺序、数据 SHA、最优指标和参数摘要一致；不得读取 independent test。

---

### Task 8: 集成 BGE 到 DetailedActionParser 与 CLI

**Files:**
- Modify: `tools/detailed_action_runtime.py`
- Modify: `detailed_action_test.py`
- Create/Modify: `tests/test_detailed_action_runtime_bge.py`
- Modify: `tests/test_detailed_action_cli.py`
- Verify: legacy isolation tests.

**Interfaces:**
- `DetailedActionParser(..., semantic_backend="bge")` default.
- Explicit fallback: `semantic_backend="character"`.
- Missing default BGE raises actionable error; no silent fallback.

- [ ] **Step 1: 写默认BGE、显式字符回退、缺模型失败和不触网测试**

- [ ] **Step 2: 运行红灯**

- [ ] **Step 3: 接入全局模型评分、候选重排和门控阈值**

成功输出仍只从 catalog 复制 detailed actions；澄清不含动作。

- [ ] **Step 4: 运行 runtime/CLI/legacy 回归**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_runtime_bge tests.test_detailed_action_cli tests.test_legacy_detailed_action_isolation -v`

Expected: PASS; old entries contain no detailed markers.

- [ ] **Step 5: 性能证据**

分别记录 cold load 与 warm inference；CLI `elapsed_ns` 保持原计时公式，报告写盘不计入。

---

### Task 9: 一次性运行 independent test

**Files:**
- Create: `tools/evaluate_detailed_action_independent_test.py`
- Create: `outputs/detailed_action_bge/independent_test_evidence.json`
- Create: `outputs/detailed_action_bge/independent_test_report.md`

**Interfaces:**
- Evaluator validates frozen test SHA before import/inference.
- Produces full per-case result, confidence, margin, timing, and aggregate metrics.

- [ ] **Step 1: 写 evaluator 完整性和指标计算测试**

- [ ] **Step 2: 冻结模型、阈值、runtime和其SHA**

- [ ] **Step 3: 只运行一次 independent test**

Expected: 942 records processed once, 0 exceptions; evidence records model/data/runtime SHA.

- [ ] **Step 4: 验收门禁**

若 scene/object/overall 任一 <95% 或任一安全类 <100%，状态为 `CHANGES_REQUIRED`，停止进入 Blind v5；不得查看原句后调参并重跑。

---

### Task 10: 全新 Blind v5 与最终验证

**Files:**
- Create: `tests/fixtures/detailed_action_runtime_blind_v5.json`
- Create: `tests/fixtures/detailed_action_runtime_blind_v5.sha256`
- Create: `outputs/detailed_action_bge/blind_v5_evidence.json`
- Create: `outputs/detailed_action_bge/blind_v5_report.md`
- Create: `.superpowers/sdd/bge-final-verification.md`

**Interfaces:**
- Blind author may read catalog/spec only before freeze; may import parser only after writing sidecar.

- [ ] **Step 1: 新代理在完全隔离上下文中编写并冻结 Blind v5**

至少 49 scene、205 object、120 safety/boundary、24 model probes；与 train/dev/test/canonical 全文重合为 0。

- [ ] **Step 2: 冻结后仅运行一次黑盒评测**

- [ ] **Step 3: 独立审查反作弊与指标**

检查 runtime 不含 Blind v5 片段/ID/答案映射，fixture SHA 未变，模型实际影响 confidence/margin/ranking。

- [ ] **Step 4: 全量回归和编译**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_spoken_dataset tests.test_detailed_action_encoder_download tests.test_detailed_action_bge_model tests.test_detailed_action_runtime_bge tests.test_detailed_action_runtime tests.test_detailed_action_cli tests.test_legacy_detailed_action_isolation tests.test_independent_blind_evaluation tests.test_natural_language_vla_bedtime_routing tests.test_multi_device_coordination tests.test_vla_action_template_routing -v`

Run: `.\.venv\Scripts\python.exe -m compileall -q detailed_action_test.py train_detailed_action_bge.py tools tests`

Expected: all listed tests PASS and compileall exits 0. Independent test/Blind v5 指标使用已冻结 evidence 读取，不在回归命令中重跑。

- [ ] **Step 5: 最终状态**

只有 independent test 与 Blind v5 均满足三组 >=95%、九类安全 100%、无动作泄漏，才可声明完成；否则输出固定失败证据和下一步数据/模型建议。

## Plan Self-Review

- Spec coverage: 数据规模、独立作者、数据隔离、模型下载与许可证、双头训练、运行时默认切换、阈值校准、independent test、Blind v5、旧入口隔离和计时均有明确任务。
- Placeholder scan: 无 TBD/TODO/“以后实现”；所有失败路径、计数和验收阈值均已确定。
- Interface consistency: 254 ordered labels、catalog SHA、dataset manifest SHA、encoder revision 从数据构建贯穿 checkpoint、runtime 与 evidence。
- Scope: 任务按可独立审查产物拆分；标注任务与训练实现任务隔离。
- Repository constraint: 无有效 Git metadata，因此不包含不可执行的 commit 步骤，以报告与 SHA manifest 保留审计链。
