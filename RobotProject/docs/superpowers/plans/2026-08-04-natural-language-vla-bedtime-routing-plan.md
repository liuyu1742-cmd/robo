# Natural-Language VLA and Bedtime Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three screenshot instructions and approved bedtime paraphrases pass through the desktop/CLI entry while preserving existing VLA, multi-device, and ordinary-task behavior.

**Architecture:** Introduce one shared VLA instruction planner used by both the interactive entry and benchmark code. Route instructions in the order multi-device, VLA, then ordinary 15-task parsing; extend bedtime recognition with positive paraphrases and explicit negation protection.

**Tech Stack:** Python 3.11, `dataclasses`, JSON, existing VLA catalogues, `unittest`.

## Global Constraints

- Do not modify the approved VLA workbook-derived templates or detailed action catalogue.
- Do not modify model weights, the three coordinated-device targets, or historical benchmark evidence.
- The two VLA screenshot phrases must pass without loading the ordinary semantic model.
- Positive bedtime paraphrases must invoke air conditioner, electric curtain, and bedroom lamp coordination.
- Negative bedtime phrases must not invoke multi-device coordination.
- `C:\RobotProject\RobotProject` is not an active Git repository, so commit steps cannot be executed in this environment.

---

### Task 1: Add Failing Natural-Language Routing Tests

**Files:**
- Create: `tests/test_natural_language_vla_bedtime_routing.py`

**Interfaces:**
- Consumes: `manual_instruction_test.run_instruction(args, instruction)`, `manual_instruction_test.format_result_text(result)`, and `tools.multi_device_coordination.is_multi_device_instruction(text)`.
- Produces: regression coverage for the exact screenshot phrases, approved positive paraphrases, and negative phrases.

- [ ] **Step 1: Write the failing VLA tests**

Create a temporary no-model argument helper and assert both phrases return completed VLA decisions:

```python
def test_screenshot_vla_phrases_pass_without_generic_parser_failure(self):
    examples = (
        ("把牙膏放到洗手台的杯中", "vla_081"),
        ("把数码相机安装到三脚架上", "vla_046"),
    )
    for instruction, object_id in examples:
        with self.subTest(instruction=instruction), tempfile.TemporaryDirectory() as directory:
            try:
                result = app.run_instruction(_args(directory), instruction)
            except Exception as exc:
                self.fail(f"入口不应拒绝VLA指令：{exc}")
            self.assertEqual(result["result_type"], "vla_task_decision")
            self.assertEqual(result["resolved"]["object"], object_id)
            self.assertTrue(result["detailed_action_sequence"])
            self.assertTrue(result["completed"])
            self.assertIn("完成情况：通过", app.format_result_text(result))
```

- [ ] **Step 2: Write the failing bedtime tests**

```python
def test_screenshot_and_approved_bedtime_phrases_trigger_three_devices(self):
    for instruction in ("我要睡觉了", "我想睡觉了", "准备睡觉", "该睡觉了"):
        with self.subTest(instruction=instruction), tempfile.TemporaryDirectory() as directory:
            try:
                result = app.run_instruction(_args(directory), instruction)
            except Exception as exc:
                self.fail(f"入口不应拒绝就寝指令：{exc}")
            self.assertEqual(result["result_type"], "multi_device_coordination")
            self.assertEqual(
                set(result["devices"]),
                {"air_conditioner", "electric_curtain", "bedroom_lamp"},
            )
            self.assertTrue(result["completed"])
            self.assertIn("完成情况：通过", app.format_result_text(result))

def test_negative_bedtime_phrases_do_not_trigger_coordination(self):
    for instruction in ("我不想睡觉", "今天不睡觉", "不用准备睡觉环境", "不要进入睡眠模式"):
        with self.subTest(instruction=instruction):
            self.assertFalse(is_multi_device_instruction(instruction))
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_natural_language_vla_bedtime_routing
```

Expected: failures state that the two VLA phrases and positive sleep phrases are rejected, and that embedded negative environment phrases are incorrectly recognized.

### Task 2: Extract a Shared VLA Instruction Planner

**Files:**
- Create: `tools/vla_instruction_planner.py`
- Modify: `tools/run_decision_planning_benchmark.py`
- Test: `tests/test_natural_language_vla_bedtime_routing.py`
- Test: `tests/test_decision_planning_benchmark.py`

**Interfaces:**
- Produces: `plan_vla_instruction(instruction: str) -> dict[str, Any] | None` and `format_vla_instruction_result(result: Mapping[str, Any]) -> str`.
- Consumes: `select_vla_action_template`, `template_actions`, and `get_vla_detailed_action_sequence`.

- [ ] **Step 1: Implement the shared planner**

Create `tools/vla_instruction_planner.py` with a planner that returns `None` when no VLA object matches, raises `ValueError("VLA物体缺少审核后的细分动作：<id>")` when approved details are missing, and otherwise returns:

```python
{
    "result_type": "vla_task_decision",
    "instruction": instruction,
    "resolved": {
        "task": template.task_id,
        "object": template.object_id,
        "object_name_zh": template.object_name_zh,
        "operation": template.source_operation_label,
    },
    "final_actions": template_actions(template),
    "template_steps": list(template.steps),
    "template_selection": {
        "task_id": template.task_id,
        "object_id": template.object_id,
        "object_name_zh": template.object_name_zh,
        "source_operation_label": template.source_operation_label,
        "steps": list(template.steps),
    },
    "detailed_action_sequence": list(detailed.steps),
    "completed": True,
    "completion_status": "通过",
    "completion_explanation": "已匹配审核后的VLA动作模板和额外细分动作序列",
    "real_output": output_text,
}
```

The formatter must include the original instruction, VLA task/object, preset template, numbered detailed sequence, `完成情况：通过`, and the completion explanation.

- [ ] **Step 2: Replace the benchmark-private planner**

Import `plan_vla_instruction` in `tools/run_decision_planning_benchmark.py`, delete the duplicated private `_plan_vla_instruction`, and change:

```python
if vla_result := plan_vla_instruction(instruction):
    return vla_result
```

- [ ] **Step 3: Run focused planner and benchmark tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_decision_planning_benchmark tests.test_natural_language_vla_bedtime_routing
```

Expected: benchmark tests pass; the exact VLA entry tests remain RED until Task 3 routes them through the shared planner.

### Task 3: Route the Interactive Entry Through VLA Before Generic Parsing

**Files:**
- Modify: `manual_instruction_test.py`
- Modify: `tools/manual_instruction_entry.py`
- Test: `tests/test_manual_instruction_runtime.py`
- Test: `tests/test_natural_language_vla_bedtime_routing.py`

**Interfaces:**
- Consumes: `plan_vla_instruction(instruction)` and `format_vla_instruction_result(result)` from Task 2.
- Produces: multi-device → VLA → ordinary-parser routing in `run_instruction`.

- [ ] **Step 1: Add the early VLA route**

Immediately after the existing multi-device branch in `run_instruction`, add:

```python
if vla_result := plan_vla_instruction(instruction):
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(vla_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return vla_result
```

Remove the now-duplicated post-generic-parser VLA selection block and its unused imports; leave the ordinary model path unchanged.

- [ ] **Step 2: Add VLA formatting dispatch**

At the beginning of `format_result_text`, after the multi-device branch, add:

```python
if result.get("result_type") == "vla_task_decision":
    from tools.vla_instruction_planner import format_vla_instruction_result

    return format_vla_instruction_result(result)
```

- [ ] **Step 3: Run VLA runtime tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_natural_language_vla_bedtime_routing tests.test_manual_instruction_runtime
```

Expected: both screenshot VLA phrases pass, existing garbage-bin VLA tests still pass, and bedtime-positive tests remain RED until Task 4.

### Task 4: Extend Bedtime Recognition With Negation Protection

**Files:**
- Modify: `tools/multi_device_coordination.py`
- Test: `tests/test_multi_device_coordination.py`
- Test: `tests/test_natural_language_vla_bedtime_routing.py`

**Interfaces:**
- Consumes: normalized instruction text from `_compact(text)`.
- Produces: `is_multi_device_instruction(text) -> bool` recognizing approved positive phrases while rejecting explicit negations.

- [ ] **Step 1: Add positive and negative phrase constants**

Extend `BEDTIME_TERMS` with `我要睡觉了`, `我想睡觉了`, `准备睡觉`, and `该睡觉了`. Add:

```python
BEDTIME_NEGATIVE_TERMS = (
    "不想睡觉",
    "不睡觉",
    "不用准备睡觉环境",
    "不要进入睡眠模式",
)
```

- [ ] **Step 2: Apply negation before positive matching**

Change `is_multi_device_instruction` to:

```python
compact = _compact(text)
if any(_compact(term) in compact for term in BEDTIME_NEGATIVE_TERMS):
    return False
return any(_compact(term) in compact for term in BEDTIME_TERMS)
```

- [ ] **Step 3: Run routing and multi-device tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_natural_language_vla_bedtime_routing tests.test_multi_device_coordination
```

Expected: all positive and negative routing tests pass; existing multi-device behavior remains unchanged.

### Task 5: Full Verification

**Files:**
- Verify all files changed in Tasks 1–4.

**Interfaces:**
- Consumes: the complete project test suite relevant to instruction routing.
- Produces: fresh evidence that all requested and prior behaviors pass.

- [ ] **Step 1: Run the three screenshot instructions through the actual CLI**

Run each command:

```powershell
.\.venv\Scripts\python.exe manual_instruction_test.py --instruction "把牙膏放到洗手台的杯中" --no-gui --no-model
.\.venv\Scripts\python.exe manual_instruction_test.py --instruction "把数码相机安装到三脚架上" --no-gui --no-model
.\.venv\Scripts\python.exe manual_instruction_test.py --instruction "我要睡觉了" --no-gui --no-model
```

Expected: each output contains `完成情况：通过`; the VLA outputs include the correct object IDs and the sleep output includes all three devices.

- [ ] **Step 2: Run the regression suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_natural_language_vla_bedtime_routing tests.test_manual_instruction_runtime tests.test_multi_device_coordination tests.test_decision_planning_benchmark tests.test_manual_instruction_entry tests.test_manual_instruction_cli tests.test_vla_detailed_action_sequences
```

Expected: all tests pass with no errors or warnings.

- [ ] **Step 3: Compile changed Python files**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q manual_instruction_test.py tools\vla_instruction_planner.py tools\manual_instruction_entry.py tools\multi_device_coordination.py tools\run_decision_planning_benchmark.py tests\test_natural_language_vla_bedtime_routing.py
```

Expected: exit code 0 and no compiler output.
