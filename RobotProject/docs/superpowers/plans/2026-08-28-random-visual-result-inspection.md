# Random Visual Result Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove visual acceptance wording, summarize failure reasons by count, and show the selected row's full actions or failure explanation in the upper detail panel.

**Architecture:** Add pure presentation helpers to `detailed_action_random_visual_test.py` so reason classification and detail formatting are testable without Tk. Keep a stable Treeview item-id-to-result mapping in `RandomVisualApp`; both live events and row-selection events call one detail renderer. Existing evidence and model logic remain unchanged.

**Tech Stack:** Python 3.11, Tkinter/ttk, unittest, existing random detailed-action result dictionaries.

## Global Constraints

- Remove “验收”“达标”“未达标” from the visual interface only.
- Preserve acceptance data in JSON/CSV/Markdown automation evidence.
- Passing rows show the complete returned detailed actions.
- Failing rows show their exact failure reason; any returned actions are marked “非最终采用”.
- Filters must not break the selected Treeview row's mapping to its full result.
- Do not change command generation, inference, timing, accuracy, or old entry behavior.

---

### Task 1: Failure reason presentation helpers

**Files:**
- Modify: `detailed_action_random_visual_test.py`
- Modify: `tests/test_detailed_action_random_visual_test.py`

**Interfaces:**
- Consumes: result dictionaries emitted by `RandomDetailedActionResult.to_dict()`.
- Produces: `failure_reason_zh(row: dict[str, Any]) -> str`, `format_failure_counts(rows: list[dict[str, Any]]) -> str`, and `result_detail(row: dict[str, Any]) -> dict[str, str]`.

- [ ] **Step 1: Write failing helper tests**

```python
def test_failure_reason_prefers_specific_clarification_and_counts_categories(self):
    rows = [
        {"passed": False, "clarification_reason": "BGE 执行门控未达到校准阈值", "failure_reason": "result_type_mismatch", "error": ""},
        {"passed": False, "clarification_reason": "", "failure_reason": "label_mismatch", "error": ""},
    ]
    self.assertEqual(failure_reason_zh(rows[0]), "执行门控未通过")
    self.assertEqual(format_failure_counts(rows), "执行门控未通过 1 条｜标签不匹配 1 条")

def test_result_detail_shows_actions_for_pass_and_reason_for_failure(self):
    passed = {"passed": True, "detailed_actions": ["步骤一", "步骤二"], "failure_reason": "", "clarification_reason": "", "error": ""}
    failed = {"passed": False, "detailed_actions": [], "failure_reason": "label_mismatch", "clarification_reason": "", "error": ""}
    self.assertIn("步骤一 → 步骤二", result_detail(passed)["actions"])
    self.assertEqual(result_detail(failed)["failure"], "标签不匹配")
```

- [ ] **Step 2: Run tests and verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: import errors for the three new helper functions.

- [ ] **Step 3: Implement minimal helpers**

Add ordered reason matching for parser exception, BGE gate, object-operation incompatibility, instance/location insufficiency, label mismatch, result-type mismatch and fallback clarification. Build the failure summary with `collections.Counter`, preserving first-seen category order. Build detail text without modifying the input row.

- [ ] **Step 4: Run helper tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: all tests pass.

### Task 2: Stable Treeview selection and detail rendering

**Files:**
- Modify: `detailed_action_random_visual_test.py`
- Modify: `tests/test_detailed_action_random_visual_test.py`

**Interfaces:**
- Consumes: `result_detail(row)` from Task 1.
- Produces: `RandomVisualApp.row_by_iid: dict[str, dict[str, Any]]`, `_show_row(row)`, and `_on_tree_select(event=None)`.

- [ ] **Step 1: Add a failing stable-mapping test**

Use a small fake Treeview whose `selection()` returns `("RND-00002",)` and fake StringVar objects with `set()`. Construct the app with `object.__new__(RandomVisualApp)`, assign `row_by_iid`, and assert `_on_tree_select()` renders the row mapped to `RND-00002`, not its list position.

- [ ] **Step 2: Run the selection test and verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: missing `row_by_iid`/`_on_tree_select` behavior.

- [ ] **Step 3: Implement row mapping and selection binding**

Initialize `row_by_iid = {}`. Insert every Treeview item with `iid=row["case_id"]`, store the complete row under the same key, bind `<<TreeviewSelect>>` to `_on_tree_select`, and rebuild the mapping in `_refresh_table`. Refactor live result handling to call `_show_row(row)` so live and selected displays are identical.

- [ ] **Step 4: Run selection tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: all tests pass, including a filtered/rebuilt map case.

### Task 3: Summary layout and failure detail fields

**Files:**
- Modify: `detailed_action_random_visual_test.py`
- Modify: `tests/test_detailed_action_random_visual_test.py`

**Interfaces:**
- Consumes: `format_failure_counts(self.rows)` and `result_detail(row)`.
- Produces: `failure_counts_var` and `failure_detail_var` shown in the GUI.

- [ ] **Step 1: Add failing UI-definition assertions**

Read the entry source in the test and assert the summary labels no longer contain `("验收", "acceptance")`; assert `failure_counts_var`, `failure_detail_var`, and the Treeview selection binding exist.

- [ ] **Step 2: Run test and verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: source assertions fail before the layout change.

- [ ] **Step 3: Modify the GUI**

Remove `acceptance` from `summary_vars`, the summary-card tuple, and summary event updates. Add a “失败原因统计” label under the summary cards and a “失败原因” row in the upper detail panel. Update counts on every result and once on summary completion. `_show_row` places complete actions for passing rows and the exact reason for failing rows.

- [ ] **Step 4: Run UI-definition and helper tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: all tests pass.

### Task 4: Regression and visual smoke verification

**Files:**
- Verify: `detailed_action_random_visual_test.py`
- Verify: `tools/detailed_action_random_runner.py`
- Verify: `tools/detailed_action_random_output.py`

**Interfaces:**
- Consumes: all completed UI behavior.
- Produces: verified entry without changes to inference/evidence behavior.

- [ ] **Step 1: Run focused and isolation regression**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test tests.test_detailed_action_random_runner tests.test_detailed_action_random_output tests.test_legacy_detailed_action_isolation`

Expected: zero failures.

- [ ] **Step 2: Compile changed modules**

Run: `.\.venv\Scripts\python.exe -m compileall -q detailed_action_random_visual_test.py tools\detailed_action_random_runner.py tools\detailed_action_random_output.py`

Expected: exit code 0.

- [ ] **Step 3: Run a CUDA smoke batch**

Run: `.\.venv\Scripts\python.exe detailed_action_random_visual_test.py --no-gui --count 10 --seed 20260904 --device cuda`

Expected: ten completed rows, evidence files written, and unchanged accuracy/timing output behavior.

- [ ] **Step 4: Inspect the GUI manually**

Launch `run_detailed_action_random_visual_test.bat`, run a small batch, click one passing and one failing row, then change the filter and click again. Confirm the upper panel tracks the selected row, actions are complete, failure reasons are specific, and no visual “验收/达标” label remains.

