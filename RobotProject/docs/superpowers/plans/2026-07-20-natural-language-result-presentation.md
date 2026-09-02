# Natural-language Result Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Present manual-instruction results in natural Chinese while retaining inspectable technical details.

**Architecture:** Keep `build_manual_result` and all parser/model/scoring data unchanged. Add pure display helpers in `tools.manual_instruction_entry`, then make `format_result_text` render a short natural-language section followed by an explicit technical-details section.

**Tech Stack:** Python 3.11 and standard-library `unittest`.

## Global Constraints

- Do not change parsing, model generation, action constraints, standard actions, or scoring.
- Preserve the user’s original instruction verbatim in the visible result.
- Default content must use Chinese descriptions; internal IDs remain available only in technical details.
- Unknown action primitives must remain visible instead of being silently mistranslated.

---

### Task 1: Add pure Chinese display helpers

**Files:**
- Modify: `tests/test_manual_instruction_entry.py`
- Modify: `tools/manual_instruction_entry.py`

**Interfaces:**
- Produces: `describe_resolved_intent(resolved: Mapping) -> str`
- Produces: `describe_actions(actions: Sequence[str]) -> list[str]`

- [ ] **Step 1: Write failing helper tests**

```python
def test_display_helpers_translate_reading_lamp_actions(self):
    resolved = {"task": "appliance_management", "object": "reading_lamp", "operation": "turn_off"}
    self.assertEqual(describe_resolved_intent(resolved), "关闭阅读灯")
    self.assertEqual(
        describe_actions([
            "locate(reading_lamp)",
            "turn_off(reading_lamp)",
            "confirm_state(reading_lamp)",
        ]),
        ["找到阅读灯", "关闭阅读灯", "确认已经关闭"],
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_entry -v `

Expected: FAIL because the two display helpers do not yet exist.

- [ ] **Step 3: Implement the smallest display-only translation layer**

```python
def describe_resolved_intent(resolved: Mapping[str, str]) -> str:
    object_name = OBJECT_NAMES_ZH.get(str(resolved["object"]), str(resolved["object"]))
    verb = OPERATION_NAMES_ZH.get(str(resolved["operation"]), str(resolved["operation"]))
    return f"{verb}{object_name}"

def describe_actions(actions: Sequence[str]) -> list[str]:
    return [_describe_action(action) for action in actions]
```

Implement `_describe_action` for `locate`, `turn_on`, `turn_off`, `confirm_state`, `grasp`, `move`, `wash`, `water`, `place`, and `inspect`; return the original action string for unknown syntax or primitive.

- [ ] **Step 4: Verify the helper test passes**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_entry -v `

Expected: PASS with the three reading-lamp actions rendered in Chinese.

- [ ] **Step 5: Commit**

This workspace is not a Git repository. Record completion in this plan; no commit command is available.

### Task 2: Render the natural result first and technical data second

**Files:**
- Modify: `tests/test_manual_instruction_entry.py`
- Modify: `tools/manual_instruction_entry.py`

**Interfaces:**
- Consumes: unchanged result dictionary from `build_manual_result`.
- Produces: `format_result_text(result: dict) -> str` with a user-facing section and `技术详情（供检查）` section.

- [ ] **Step 1: Write the failing presentation test**

```python
def test_popup_text_keeps_user_words_and_separates_technical_details(self):
    resolved = resolve_manual_instruction("睡前把阅读灯关了")
    actions = ["locate(reading_lamp)", "turn_off(reading_lamp)", "confirm_state(reading_lamp)"]
    text = format_result_text(build_manual_result("睡前把阅读灯关了", resolved, actions, actions, model_used=True))
    visible, technical = text.split("技术详情（供检查）：", maxsplit=1)
    self.assertIn("你的指令：睡前把阅读灯关了", visible)
    self.assertIn("我理解为：关闭阅读灯。", visible)
    self.assertIn("建议步骤：找到阅读灯 → 关闭阅读灯 → 确认已经关闭", visible)
    self.assertNotIn("reading_lamp", visible)
    self.assertIn("reading_lamp", technical)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_entry -v `

Expected: FAIL because current formatting places technical IDs in the first block.

- [ ] **Step 3: Replace only the formatter output structure**

```python
visible_lines = [
    f"你的指令：{result['instruction']}",
    f"我理解为：{describe_resolved_intent(result['resolved'])}。",
    f"建议步骤：{' → '.join(describe_actions(result['final_actions']))}",
    f"完成情况：{result['completion_status']}",
]
technical_lines = ["技术详情（供检查）：", ...]
return "\n".join([*visible_lines, "", *technical_lines])
```

Populate technical lines with the existing task/object/operation/target, original model actions, final actions, expected actions and constraint fields. Do not alter the result dictionary.

- [ ] **Step 4: Verify presentation and runtime regression tests pass**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_entry tests.test_manual_instruction_runtime tests.test_action_constraints -v `

Expected: PASS; popup formatting retains raw technical values after the separator.

- [ ] **Step 5: Commit**

This workspace is not a Git repository. Record completion in this plan; no commit command is available.

### Task 3: Verify the complete focused workflow without changing score behavior

**Files:**
- Modify: `docs/CURRENT_PROJECT_STATUS_2026-07-04.md`

**Interfaces:**
- Consumes: the implemented display helpers and existing test suite.
- Produces: a concise status note stating that natural-language presentation does not change raw or final scores.

- [ ] **Step 1: Write a documentation contract test**

```python
def test_status_says_natural_presentation_does_not_change_scoring(self):
    status = (ROOT / "docs/CURRENT_PROJECT_STATUS_2026-07-04.md").read_text(encoding="utf-8")
    self.assertIn("自然语言展示不改变解析、原始模型分数或最终系统分数", status)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_independent_blind_evaluation -v `

Expected: FAIL until the status note is added.

- [ ] **Step 3: Append the status note**

Add a dated note stating that popup/report display now preserves the user’s instruction and uses Chinese action descriptions, while raw generation, final constrained actions and scores remain unchanged.

- [ ] **Step 4: Run focused verification**

Run: ` .\.venv\Scripts\python.exe -m unittest tests.test_manual_instruction_entry tests.test_manual_instruction_runtime tests.test_action_constraints tests.test_manual_instruction_resolution tests.test_independent_blind_evaluation -v `

Run: ` .\.venv\Scripts\python.exe -m compileall -q tools manual_instruction_test.py evaluate_independent_blind_action_test.py `

Expected: all focused tests PASS and compilation exits with code 0.

- [ ] **Step 5: Commit**

This workspace is not a Git repository. Record completion in this plan; no commit command is available.
