# 独立细分动作测试入口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可从 PyCharm 直接运行的口语化细分动作入口，并把细分动作从所有旧入口中彻底隔离，同时恢复冻结盲测入口的默认可运行性。

**Architecture:** 从已审核的全物体联动目录生成稳定的 254 标签细分动作目录（49 个联动场景 + 205 个物体操作），用候选词约束和本地字符级分类模型共同路由口语指令。新入口是细分动作的唯一展示入口；旧入口只保留原动作。冻结盲测默认校验冻结 JSON 内部结构，只有显式提供外部 CSV 时才做 SHA-256 复核。

**Tech Stack:** Python 3.11、PyTorch、`unittest`、Tkinter、JSON、`time.perf_counter_ns()`。

## Global Constraints

- 新入口成功结果不得包含原动作、动作模板、`final_actions` 或 `expected_actions`。
- 所有旧入口不得返回或显示 `detailed_action_sequence` 或“额外细分动作”。
- 低置信度、候选过近、同名物体用途不明确或否定冲突时必须拒绝猜测，最多列出三个候选且不输出动作。
- 计时从用户提交后开始，到完整结果格式化结束；不包含用户等待、弹窗停留和报告写盘。
- 不关闭冻结盲测完整性校验；显式错误 CSV 仍必须失败。
- 不新增网络服务或在线模型依赖。

---

### Task 1: 生成并验证独立细分动作目录

**Files:**
- Create: `tools/detailed_action_catalog.py`
- Create: `meta/detailed_action_entry_catalog.json`
- Create: `tests/test_detailed_action_catalog.py`
- Consume: `tools/build_all_object_coordination_catalog.py`

**Interfaces:**
- Produces: `build_detailed_action_catalog() -> dict`、`load_detailed_action_catalog(path: Path = DEFAULT_CATALOG) -> dict`、`validate_detailed_action_catalog(catalog: Mapping) -> list[str]`。
- Catalog labels: `scene:<scenario_id>` and `object:<source>:<object_id>`，合计 254 个唯一标签。

- [ ] **Step 1: 写目录覆盖和隔离失败测试**

```python
class DetailedActionCatalogTest(unittest.TestCase):
    def test_catalog_has_49_scenes_and_205_object_labels(self):
        catalog = build_detailed_action_catalog()
        labels = catalog["labels"]
        self.assertEqual(sum(x["label_type"] == "coordination_scene" for x in labels), 49)
        self.assertEqual(sum(x["label_type"] == "object_detail" for x in labels), 205)
        self.assertEqual(len({x["label_id"] for x in labels}), 254)

    def test_object_labels_only_expose_detailed_actions(self):
        for row in build_detailed_action_catalog()["labels"]:
            self.assertNotIn("original_actions", row)
            self.assertNotIn("final_actions", row)
            if row["label_type"] == "object_detail":
                self.assertTrue(row["detailed_actions"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_detailed_action_catalog -v`

Expected: FAIL because `tools.detailed_action_catalog` does not exist.

- [ ] **Step 3: 实现目录构建与严格校验**

```python
def build_detailed_action_catalog() -> dict:
    source = build_catalog()
    labels = []
    for scenario in source["scenarios"]:
        rows = [x for x in source["object_details"] if x["scenario_id"] == scenario["scenario_id"]]
        labels.append({
            "label_id": f"scene:{scenario['scenario_id']}",
            "label_type": "coordination_scene",
            "canonical_command": scenario["command"],
            "intent_key": scenario["intent_key"],
            "objects": [object_payload(row) for row in rows],
        })
    for row in source["object_details"]:
        labels.append({
            "label_id": f"object:{row['source']}:{row['object_id']}",
            "label_type": "object_detail",
            "task": row["task"],
            "object_id": row["object_id"],
            "object_name_zh": row["object_name_zh"],
            "operation": row["operation"],
            "detailed_actions": list(row["detailed_actions"]),
        })
    catalog = {"format": "detailed_action_entry_catalog_v1", "labels": labels}
    errors = validate_detailed_action_catalog(catalog)
    if errors:
        raise ValueError("；".join(errors))
    return catalog
```

- [ ] **Step 4: 导出 JSON 并运行目录测试**

Run: `python -m tools.detailed_action_catalog --output meta/detailed_action_entry_catalog.json`

Run: `python -m unittest tests.test_detailed_action_catalog -v`

Expected: 49 scenes, 205 objects, 254 unique labels; all tests PASS.

---

### Task 2: 构建专用口语训练数据并训练模型

**Files:**
- Create: `tools/detailed_action_semantic_data.py`
- Create: `train_detailed_action_semantic_parser.py`
- Create: `tests/test_detailed_action_semantic_data.py`
- Generate: `datasets/detailed_action_semantic_train.json`
- Generate: `datasets/detailed_action_semantic_dev.json`
- Generate: `models/detailed_action_semantic/best.pt`
- Reuse: `tools/semantic_action_model.py`

**Interfaces:**
- Produces: `build_detailed_action_examples(catalog: Mapping) -> list[dict]` and a checkpoint whose label set exactly equals catalog label IDs.
- Training example shape: `{"text": str, "label_id": str, "source": str}`.

- [ ] **Step 1: 写训练数据完整性与不泄漏测试**

```python
class DetailedActionSemanticDataTest(unittest.TestCase):
    def test_every_label_has_multiple_unique_phrases(self):
        examples = build_detailed_action_examples(build_detailed_action_catalog())
        counts = Counter(row["label_id"] for row in examples)
        self.assertEqual(len(counts), 254)
        self.assertTrue(all(value >= 4 for value in counts.values()))

    def test_normalized_text_never_crosses_labels(self):
        owner = {}
        for row in build_detailed_action_examples(build_detailed_action_catalog()):
            key = normalize_text(row["text"])
            self.assertNotIn(key, owner) if key not in owner else self.assertEqual(owner[key], row["label_id"])
            owner[key] = row["label_id"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_detailed_action_semantic_data -v`

Expected: FAIL because the data builder does not exist.

- [ ] **Step 3: 生成受控口语样本**

```python
def phrase_variants(label: Mapping) -> list[str]:
    if label["label_type"] == "coordination_scene":
        base = label["canonical_command"]
        return [base, f"麻烦{base}", base.replace("帮我", "请帮我"), base.rstrip("了") + "吧"]
    name = label["object_name_zh"]
    operation = operation_phrase(label["task"], label["operation"])
    return [
        f"帮我{operation}{name}",
        f"麻烦把{name}{operation}",
        f"我想处理一下{name}",
        f"请给出{name}的详细操作步骤",
    ]
```

同义词来自现有 `OBJECT_SYNONYMS`；重复文本只能属于同一标签，否则构建失败。将规范化文本按稳定哈希切为 train/dev，禁止跨集合重复。

- [ ] **Step 4: 训练本地字符级模型并保存标签审计信息**

Run: `python train_detailed_action_semantic_parser.py --epochs 40 --seed 20260813`

Expected: writes `models/detailed_action_semantic/best.pt`; checkpoint format is `semantic_action_character_ngram_v1`; checkpoint metadata includes `catalog_sha256`, `train_examples`, `dev_examples`, and all 254 labels.

- [ ] **Step 5: 验证模型与目录标签完全一致**

Run: `python -m unittest tests.test_detailed_action_semantic_data -v`

Expected: all tests PASS and no train/dev normalized-text overlap.

---

### Task 3: 实现候选约束、模型路由和拒绝猜测

**Files:**
- Create: `tools/detailed_action_runtime.py`
- Create: `tests/test_detailed_action_runtime.py`
- Consume: `meta/detailed_action_entry_catalog.json`
- Consume: `models/detailed_action_semantic/best.pt`

**Interfaces:**
- Produces: `DetailedActionParser.resolve(instruction: str) -> dict` and `format_detailed_action_result(result: Mapping) -> str`。
- Result types: `object_detailed_action`, `coordination_detailed_action`, `clarification_required`。

- [ ] **Step 1: 写代表性口语路由和歧义失败测试**

```python
class DetailedActionRuntimeTest(unittest.TestCase):
    def test_colloquial_scene_and_object_commands(self):
        parser = DetailedActionParser.for_test()
        bedtime = parser.resolve("我准备休息了，调一下睡眠环境")
        self.assertEqual(bedtime["result_type"], "coordination_detailed_action")
        self.assertEqual({x["object_id"] for x in bedtime["objects"]}, {"air_conditioner", "electric_curtain", "bedroom_lamp"})
        floor = parser.resolve("地板有点脏，帮我彻底弄干净")
        self.assertTrue(any("地面" in x["object_name_zh"] or "地板" in x["object_name_zh"] for x in floor["objects"]))

    def test_ambiguous_bowl_request_refuses_to_guess(self):
        result = DetailedActionParser.for_test().resolve("处理一下碗")
        self.assertEqual(result["result_type"], "clarification_required")
        self.assertFalse(result.get("detailed_actions"))
        self.assertLessEqual(len(result["candidates"]), 3)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_detailed_action_runtime -v`

Expected: FAIL because runtime is absent.

- [ ] **Step 3: 实现候选生成与模型重排**

```python
def resolve(self, instruction: str) -> dict:
    normalized = normalize_text(instruction)
    candidates = self.candidate_labels(normalized)
    predictions = self.predict(instruction)
    ranked = rerank(predictions, candidates)
    if not ranked or ranked[0]["confidence"] < self.min_confidence:
        return clarification_result(instruction, "模型置信度不足", ranked[:3])
    if len(ranked) > 1 and ranked[0]["confidence"] - ranked[1]["confidence"] < self.min_margin:
        return clarification_result(instruction, "候选结果过于接近", ranked[:3])
    return self.build_success_result(instruction, ranked[0])
```

候选规则必须处理否定词；当显式物体名对应多个标签且没有操作/场景词可消歧时，直接要求澄清。

- [ ] **Step 4: 验证语义边界**

Run: `python -m unittest tests.test_detailed_action_runtime tests.test_all_object_coordination_catalog -v`

Expected: breakfast never maps to books; floor commands contain floor objects; bedtime is exactly three devices; ambiguity produces no action sequence.

---

### Task 4: 建立 PyCharm 可直接运行的新入口并加入可信计时

**Files:**
- Create: `detailed_action_test.py`
- Create: `tests/test_detailed_action_cli.py`
- Generate: `outputs/detailed_action_test/latest_result.json`

**Interfaces:**
- Produces: `run_detailed_instruction(args, instruction: str) -> dict`、`timed_detailed_instruction(args, instruction: str) -> tuple[dict, str]`、`main() -> None`。

- [ ] **Step 1: 写字段隔离、计时公式和 GUI 回退失败测试**

```python
class DetailedActionCliTest(unittest.TestCase):
    def test_success_contains_only_detailed_actions_and_valid_timing(self):
        result, text = timed_detailed_instruction(fake_args(), "帮我把地面清理干净")
        self.assertIn("detailed_action_sequence", json.dumps(result, ensure_ascii=False))
        self.assertNotIn("original_actions", result)
        self.assertNotIn("final_actions", result)
        self.assertEqual(result["elapsed_ns"], result["ended_perf_counter_ns"] - result["started_perf_counter_ns"])
        self.assertAlmostEqual(result["elapsed_seconds"], result["elapsed_ns"] / 1_000_000_000)
        self.assertIn("运行耗时", text)

    def test_popup_failure_falls_back_to_terminal(self):
        with patch("detailed_action_test._get_popup_instruction", side_effect=RuntimeError("no display")), patch("builtins.input", return_value="帮我准备早餐"):
            self.assertEqual(main_for_test([]), 0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_detailed_action_cli -v`

Expected: FAIL because entry file does not exist.

- [ ] **Step 3: 实现计时边界和输出格式**

```python
def timed_detailed_instruction(args, instruction: str) -> tuple[dict, str]:
    started_ns = time.perf_counter_ns()
    result = parser_from_args(args).resolve(instruction)
    text_without_time = format_detailed_action_result(result)
    ended_ns = time.perf_counter_ns()
    elapsed_ns = ended_ns - started_ns
    result.update({
        "started_perf_counter_ns": started_ns,
        "ended_perf_counter_ns": ended_ns,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_ns / 1_000_000_000,
    })
    text = text_without_time + f"\n运行耗时：{result['elapsed_seconds']:.9f} 秒"
    return result, text
```

报告写盘和 `_show_popup_result()` 必须发生在函数返回之后。

- [ ] **Step 4: 验证命令行和模拟 PyCharm 无参数路径**

Run: `python detailed_action_test.py --instruction "我要睡觉了" --no-gui`

Run: `python detailed_action_test.py --instruction "帮我准备早餐" --no-gui`

Expected: each prints only detailed actions and timing, and writes valid JSON.

---

### Task 5: 从所有旧入口删除细分动作返回与显示

**Files:**
- Modify: `tools/vla_instruction_planner.py`
- Modify: `tools/manual_instruction_entry.py`
- Modify: `manual_instruction_test.py`
- Modify: `tools/action_parser_demo_runner.py`
- Modify: `tools/run_decision_planning_benchmark.py`
- Modify: `tools/update_decision_test_docx.py`
- Modify: affected tests under `tests/`
- Inspect only and modify if applicable: `final_demo.py`

**Interfaces:**
- Old VLA planner continues returning `template_steps` and `final_actions`, but never `detailed_action_sequence`.
- `build_manual_result(...)` no longer accepts a detailed-action argument.

- [ ] **Step 1: 写全局旧入口隔离失败测试**

```python
class LegacyDetailedActionIsolationTest(unittest.TestCase):
    def test_old_vla_and_manual_results_never_expose_detailed_actions(self):
        vla = plan_vla_instruction("把牙膏放到洗手台的杯中")
        self.assertNotIn("detailed_action_sequence", vla)
        self.assertNotIn("额外细分动作", format_vla_instruction_result(vla))
        manual = run_instruction(fake_args(), "打开阅读灯")
        self.assertNotIn("detailed_action_sequence", manual)

    def test_old_entry_source_has_no_detailed_action_access(self):
        forbidden = ("detailed_action_sequence", "get_vla_detailed_action_sequence")
        for path in LEGACY_ENTRY_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertTrue(all(token not in text for token in forbidden), path)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_legacy_detailed_action_isolation -v`

Expected: FAIL on current VLA/manual/demo paths.

- [ ] **Step 3: 删除旧字段、导入和格式化分支**

```python
def plan_vla_instruction(instruction: str) -> dict[str, Any] | None:
    template = select_vla_action_template(instruction)
    if template is None:
        return None
    actions = template_actions(template)
    return {
        "result_type": "vla_task_decision",
        "instruction": instruction,
        "resolved": resolved_payload(template),
        "final_actions": actions,
        "template_steps": list(template.steps),
        "completed": True,
        "completion_status": "通过",
        "completion_explanation": "已匹配审核前既有VLA动作模板",
    }
```

从旧构造器、失败结果、演示详情、决策报告和文档生成逻辑中删除细分字段；需要细分结果的功能改为显式调用 `tools.detailed_action_runtime`，不得从旧结果中读取。

- [ ] **Step 4: 运行旧入口与隔离回归测试**

Run: `python -m unittest tests.test_legacy_detailed_action_isolation tests.test_manual_instruction_runtime tests.test_manual_instruction_entry tests.test_natural_language_vla_bedtime_routing tests.test_action_parser_demo_runner -v`

Expected: old entries still return original/template actions, with no detailed-action field or text.

---

### Task 6: 修复冻结盲测默认 PyCharm 运行路径

**Files:**
- Modify: `evaluate_independent_blind_action_test.py`
- Modify: `tests/test_independent_blind_evaluation.py`

**Interfaces:**
- `load_frozen_bundle(bundle_path: Path, benchmark_path: Path, source_csv_path: Path | None = None) -> dict`。
- `--source-csv` default is `None`; explicit paths retain SHA-256 enforcement.

- [ ] **Step 1: 写默认无CSV和显式错误CSV失败测试**

```python
def test_default_bundle_load_does_not_require_external_csv(self):
    bundle = load_frozen_bundle(DEFAULT_BUNDLE, DEFAULT_BENCHMARK)
    self.assertEqual(len(bundle["cases"]), 414)

def test_explicit_wrong_csv_still_fails_with_hash_evidence(self):
    with self.assertRaisesRegex(ValueError, "expected=.*actual=.*path="):
        load_frozen_bundle(bundle_path, benchmark_path, wrong_csv)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_independent_blind_evaluation -v`

Expected: default call/signature test FAIL.

- [ ] **Step 3: 实现可选外部指纹复核**

```python
def load_frozen_bundle(bundle_path, benchmark_path, source_csv_path=None):
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    expected_relations = sorted(str(record["relation_key"]) for record in benchmark)
    validate_frozen_blind_bundle(bundle, expected_relations)
    if source_csv_path is not None:
        actual = hashlib.sha256(Path(source_csv_path).read_bytes()).hexdigest()
        expected = bundle["source_csv_sha256"]
        if actual != expected:
            raise ValueError(f"blind source CSV fingerprint mismatch: expected={expected}; actual={actual}; path={Path(source_csv_path).resolve()}")
    return bundle
```

- [ ] **Step 4: 运行无参数 smoke test 和显式指纹测试**

Run: `python evaluate_independent_blind_action_test.py --device cpu --beam-width 1 --max-actions 8`

Run: `python -m unittest tests.test_independent_blind_evaluation -v`

Expected: no default fingerprint error; explicit wrong CSV test still PASS by observing rejection.

---

### Task 7: 全量验证与证据输出

**Files:**
- Generate: `outputs/detailed_action_test/verification_object.json`
- Generate: `outputs/detailed_action_test/verification_coordination.json`
- Generate: `outputs/detailed_action_test/verification_ambiguity.json`

**Interfaces:**
- No new runtime interfaces; this task proves the completed behavior.

- [ ] **Step 1: 运行定向测试集**

Run: `python -m unittest tests.test_detailed_action_catalog tests.test_detailed_action_semantic_data tests.test_detailed_action_runtime tests.test_detailed_action_cli tests.test_legacy_detailed_action_isolation tests.test_independent_blind_evaluation -v`

Expected: all tests PASS.

- [ ] **Step 2: 运行相关既有回归测试**

Run: `python -m unittest tests.test_all_object_coordination_catalog tests.test_manual_instruction_runtime tests.test_manual_instruction_entry tests.test_natural_language_vla_bedtime_routing tests.test_multi_device_coordination tests.test_vla_action_template_routing tests.test_action_parser_demo_runner tests.test_decision_planning_benchmark -v`

Expected: all tests PASS after expectations are updated to the new isolation contract.

- [ ] **Step 3: 编译检查**

Run: `python -m compileall -q detailed_action_test.py evaluate_independent_blind_action_test.py manual_instruction_test.py final_demo.py tools tests`

Expected: exit code 0.

- [ ] **Step 4: 生成三类可审计证据**

Run: `python detailed_action_test.py --instruction "帮我准备早餐" --no-gui --report outputs/detailed_action_test/verification_object.json`

Run: `python detailed_action_test.py --instruction "我要睡觉了" --no-gui --report outputs/detailed_action_test/verification_coordination.json`

Run: `python detailed_action_test.py --instruction "处理一下碗" --no-gui --report outputs/detailed_action_test/verification_ambiguity.json`

Expected: first two contain detailed-only actions and valid timing; third is `clarification_required`, contains at most three candidates, and contains no action sequence.

- [ ] **Step 5: 复核项目没有旧入口旁路**

Run: `rg -n "detailed_action_sequence|额外细分动作|get_vla_detailed_action_sequence" manual_instruction_test.py final_demo.py tools/vla_instruction_planner.py tools/manual_instruction_entry.py tools/action_parser_demo_runner.py`

Expected: no matches. Matches are allowed only in the new detailed-action modules, catalog builders, dedicated tests, and historical report tooling explicitly migrated to the new service.

## Plan Self-Review

- Spec coverage: catalog, model, candidate constraints, ambiguity rejection, timing, GUI fallback, old-entry isolation, and frozen blind-test repair each have an implementation and verification task.
- Placeholder scan: no deferred requirements or unspecified error handling remain.
- Interface consistency: catalog label IDs flow into training checkpoint labels and `DetailedActionParser`; CLI consumes only the parser result; legacy paths never consume the new catalog.
- Repository note: this directory is not currently a valid Git work tree, so the normally required per-task commits cannot be performed unless repository metadata is restored.
