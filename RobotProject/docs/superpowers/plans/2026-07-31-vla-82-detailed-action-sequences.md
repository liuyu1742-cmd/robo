# VLA 82 Detailed Action Sequences Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the 82 approved fine-grained action sequences as a second, independent runtime output without changing the existing VLA template action sequence.

**Architecture:** Store the approved workbook content in a versioned JSON catalogue keyed by the existing `vla_001`–`vla_082` template IDs. A focused loader validates one detailed sequence per existing template. The manual instruction runtime keeps `final_actions` and `template_selection.steps` unchanged, and adds `detailed_action_sequence` as a separate list rendered under its own heading.

**Tech Stack:** Python 3.11, JSON, dataclasses, unittest, existing manual instruction runtime.

## Global Constraints

- Treat all 82 rows in `VLA_88物体_额外细分动作序列_审核草案.xlsx` as approved exactly as written, per the user's explicit instruction.
- Do not modify the source workbook, any video, annotation, dataset, model checkpoint, or existing standard action sequence.
- The new detailed sequence must be structurally separate from `final_actions`, `expected_actions`, and `template_selection.steps`.
- Cover exactly the same 82 `object_id` values as `load_vla_action_templates()`.
- Preserve legacy behavior when no VLA template is selected.

---

## File Structure

- Create `meta/vla_82_detailed_action_sequences.json`: approved detailed steps keyed by the existing VLA template IDs.
- Create `tools/vla_detailed_action_sequences.py`: load, validate, and retrieve detailed sequences.
- Create `tests/test_vla_detailed_action_sequences.py`: catalogue coverage and validation tests.
- Modify `manual_instruction_test.py`: attach the selected detailed sequence to runtime results.
- Modify `tools/manual_instruction_entry.py`: serialize and render the independent sequence.
- Modify `tests/test_manual_instruction_runtime.py`: integration and regression coverage.
- Modify `docs/PROJECT_STRUCTURE.md`: document the second output and its source-only workbook relationship.

### Task 1: Add the approved detailed-sequence catalogue

**Files:**
- Create: `meta/vla_82_detailed_action_sequences.json`
- Create: `tools/vla_detailed_action_sequences.py`
- Create: `tests/test_vla_detailed_action_sequences.py`

**Interfaces:**
- Consumes: `tools.vla_action_templates.load_vla_action_templates()`
- Produces:
  - `VlaDetailedActionSequence(object_id: str, task_id: str, object_name_zh: str, source_operation_label: str, steps: tuple[str, ...])`
  - `load_vla_detailed_action_sequences() -> dict`
  - `validate_vla_detailed_action_sequences(catalogue: Mapping[str, object]) -> list[str]`
  - `get_vla_detailed_action_sequence(object_id: str) -> VlaDetailedActionSequence | None`

- [ ] **Step 1: Write failing catalogue tests**

```python
class VlaDetailedActionSequencesTest(unittest.TestCase):
    def test_catalogue_matches_existing_82_template_ids(self):
        detailed = load_vla_detailed_action_sequences()
        original = load_vla_action_templates()
        self.assertEqual(validate_vla_detailed_action_sequences(detailed), [])
        self.assertEqual(
            {row["object_id"] for row in detailed["sequences"]},
            {row["object_id"] for row in original["candidates"]},
        )

    def test_each_sequence_is_independent_and_fine_grained(self):
        for row in load_vla_detailed_action_sequences()["sequences"]:
            self.assertGreaterEqual(len(row["steps"]), 4)
            self.assertEqual(len(row["steps"]), len(set(row["steps"])))
            self.assertTrue(all(step.strip() for step in row["steps"]))
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vla_detailed_action_sequences -v
```

Expected: FAIL because `tools.vla_detailed_action_sequences` does not exist.

- [ ] **Step 3: Snapshot the approved workbook rows into JSON**

Create versioned JSON with this top-level structure and all 82 approved rows:

```json
{
  "format": "vla_82_detailed_action_sequences_v1",
  "source": "VLA_88物体_额外细分动作序列_审核草案.xlsx",
  "approval": "user-approved-2026-07-31",
  "sequences": [
    {
      "object_id": "vla_001",
      "task_id": "indoor_cleaning",
      "object_name_zh": "垃圾桶",
      "source_operation_label": "多罐汽水投放：客厅→厨房垃圾桶",
      "steps": [
        "识别垃圾桶位置、开口/桶盖类型、内袋是否到位及剩余容量",
        "扫描客厅与通道中的饮料罐，确认数量、是否漏液，并规划无障碍搬运路径",
        "逐罐从罐身中部稳定抓取，保持开口朝上，低速运至厨房",
        "在桶口上方低高度释放，避免投掷和碰撞桶沿，重复至全部投入",
        "复核地面无遗漏、桶外无洒漏，必要时擦拭并更换内袋，最后关闭桶盖"
      ]
    }
  ]
}
```

- [ ] **Step 4: Implement strict loading and validation**

```python
@dataclass(frozen=True)
class VlaDetailedActionSequence:
    object_id: str
    task_id: str
    object_name_zh: str
    source_operation_label: str
    steps: tuple[str, ...]


def validate_vla_detailed_action_sequences(catalogue: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    rows = catalogue.get("sequences", [])
    original = load_vla_action_templates()["candidates"]
    expected_ids = {row["object_id"] for row in original}
    actual_ids = {row.get("object_id") for row in rows}
    if actual_ids != expected_ids:
        errors.append("detailed sequence IDs do not match the VLA 82 template IDs")
    for index, row in enumerate(rows):
        steps = row.get("steps", [])
        if len(steps) < 4 or len(steps) != len(set(steps)) or any(not str(step).strip() for step in steps):
            errors.append(f"invalid detailed steps at sequence {index}")
    return errors
```

- [ ] **Step 5: Run the catalogue tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vla_detailed_action_sequences -v
```

Expected: all catalogue tests PASS.

### Task 2: Add the independent runtime result field

**Files:**
- Modify: `manual_instruction_test.py`
- Modify: `tools/manual_instruction_entry.py`
- Modify: `tests/test_manual_instruction_runtime.py`

**Interfaces:**
- Consumes: `get_vla_detailed_action_sequence(template.object_id)`
- Produces: `result["detailed_action_sequence"] -> list[str]`
- Preserves: `result["final_actions"]`, `result["expected_actions"]`, and `result["template_selection"]["steps"]`

- [ ] **Step 1: Write a failing integration test**

```python
def test_manual_runtime_adds_detailed_sequence_without_changing_original(self):
    result = run_instruction(self.no_model_args, "把客厅里的多个汽水罐扔进厨房垃圾桶")
    self.assertEqual(
        result["final_actions"],
        [
            "template_step(拿起汽水罐)",
            "template_step(移动至厨房垃圾桶)",
            "template_step(投放汽水罐)",
            "template_step(确认已放入)",
        ],
    )
    self.assertEqual(result["template_selection"]["steps"][0], "拿起汽水罐")
    self.assertEqual(
        result["detailed_action_sequence"][0],
        "识别垃圾桶位置、开口/桶盖类型、内袋是否到位及剩余容量",
    )
    self.assertNotIn(result["detailed_action_sequence"][0], result["final_actions"])
```

- [ ] **Step 2: Verify the new assertion fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_runtime.ManualInstructionRuntimeTest.test_manual_runtime_adds_detailed_sequence_without_changing_original -v
```

Expected: FAIL because `detailed_action_sequence` is absent.

- [ ] **Step 3: Load the approved sequence after template selection**

In `manual_instruction_test.run_instruction`, keep the existing template overlay unchanged and add:

```python
detailed_action_sequence: list[str] = []
if template is not None:
    detailed = get_vla_detailed_action_sequence(template.object_id)
    detailed_action_sequence = list(detailed.steps) if detailed is not None else []
```

Pass it to `build_manual_result(...)`. Add an empty list to `build_failure_result(...)`.

- [ ] **Step 4: Serialize the field without affecting scoring**

Extend `build_manual_result` with:

```python
detailed_action_sequence: list[str] | None = None,
```

and include:

```python
"detailed_action_sequence": list(detailed_action_sequence or []),
```

Do not reference this field from `completed`, `scored_actions`, constraints, or expected-sequence comparison.

- [ ] **Step 5: Render the second sequence as a separate block**

In `format_result_text`, add:

```python
detailed_steps = result.get("detailed_action_sequence") or []
if detailed_steps:
    visible_lines.extend(["", "额外细分动作序列："])
    visible_lines.extend(
        f"{index}. {step}" for index, step in enumerate(detailed_steps, start=1)
    )
```

Keep the existing `预设动作模板`, `建议步骤`, and technical `最终采用动作序列` blocks unchanged.

- [ ] **Step 6: Run focused runtime tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_runtime -v
```

Expected: all runtime tests PASS.

### Task 3: Document and verify the end-to-end output

**Files:**
- Modify: `docs/PROJECT_STRUCTURE.md`
- Test: `tests/test_vla_detailed_action_sequences.py`
- Test: `tests/test_manual_instruction_runtime.py`

**Interfaces:**
- Produces user-visible heading: `额外细分动作序列：`
- Produces JSON field: `detailed_action_sequence`

- [ ] **Step 1: Document the independent-output contract**

Add to `docs/PROJECT_STRUCTURE.md`:

```markdown
- `meta/vla_82_detailed_action_sequences.json`: the user-approved 82-row fine-grained sequence catalogue.
- Runtime JSON exposes it as `detailed_action_sequence`; CLI/GUI renders it under `额外细分动作序列`.
- This sequence is independent of and does not modify `final_actions`, `expected_actions`, or the original VLA template `steps`.
```

- [ ] **Step 2: Run the focused suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vla_detailed_action_sequences tests.test_manual_instruction_runtime -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run a garbage-can CLI smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe manual_instruction_test.py --no-model --no-gui --instruction "把客厅里的多个汽水罐扔进厨房垃圾桶" --report outputs/vla_detailed_garbage_smoke.json
```

Expected:
- Existing `预设动作模板` and `最终采用动作序列` remain unchanged.
- A separate `额外细分动作序列` block contains five approved steps.
- The report JSON contains `detailed_action_sequence`.

- [ ] **Step 4: Run a cooking CLI smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe manual_instruction_test.py --no-model --no-gui --instruction "用平底锅烹调切好的卷心菜和辣椒" --report outputs/vla_detailed_cooking_smoke.json
```

Expected: the independent sequence includes ingredient preparation, `160–180 ℃` heat control, seasoning, shutdown, and cleanup.

- [ ] **Step 5: Run regression verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_entry tests.test_manual_instruction_runtime tests.test_vla_detailed_action_sequences -v
.\.venv\Scripts\python.exe -m compileall -q manual_instruction_test.py tools tests
```

Expected: all tests PASS and compileall exits with code 0.

## Self-Review

- Spec coverage: all 82 approved rows are versioned, validated, selected by the existing template ID, serialized, displayed, and tested.
- Separation requirement: no task modifies `final_actions`, `expected_actions`, template `steps`, model checkpoints, datasets, or scoring.
- Type consistency: all runtime layers use `list[str]`; the immutable catalogue record exposes `tuple[str, ...]`.
- Placeholder scan: no deferred implementation or ambiguous fallback remains.
