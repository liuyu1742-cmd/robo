# All-Object Multi-Device Coordination Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one verified Excel workbook containing realistic multi-object coordination commands, per-object original and fine-grained action parsing, and formula-driven coverage auditing for all 123 original objects and 82 VLA objects.

**Architecture:** A Python catalogue builder reads the two authoritative object catalogues and existing action sources, assigns every object to a curated real-life coordination scenario, and emits deterministic JSON. A JavaScript workbook builder uses `@oai/artifact-tool` to turn that JSON into three formatted, filterable, auditable worksheets with formulas and conditional formatting.

**Tech Stack:** Python 3.11, JSON, `unittest`, bundled Node.js, `@oai/artifact-tool`, XLSX.

## Global Constraints

- Cover 123 unique original objects and 82 VLA object IDs, for 205/205 total coverage.
- Generate 45–60 realistic coordination commands; each command includes 3–8 objects and at least two task categories.
- Preserve original action sequences; fine-grained sequences are separate additional outputs.
- Reuse all 82 approved VLA detailed sequences without modification.
- Give every original object a non-empty 4–8-step fine-grained sequence appropriate to task, material, state, and safety.
- Include the exact columns `测试场景工况`, `测试操作步骤`, and `预期输出结果` in the per-object sheet.
- Initialize every review status as `待审核`.
- Do not modify project runtime code, model weights, VLA source data, or historical benchmark evidence.
- Produce one final workbook under `outputs/019fb5d7-4c5a-7e23-bb89-f7ec997f7068/`.
- The current project directory is not an active Git repository; commit steps are not applicable.

---

### Task 1: Build and Validate the Coordination Catalogue

**Files:**
- Create: `tools/build_all_object_coordination_catalog.py`
- Create: `tests/test_all_object_coordination_catalog.py`
- Generate: `outputs/019fb5d7-4c5a-7e23-bb89-f7ec997f7068/all_object_coordination_catalog.json`

**Interfaces:**
- Consumes: `meta/compliance_catalog_15x120.json`, `meta/vla_82_detailed_action_sequences.json`, `tools.vla_action_templates.load_vla_action_templates()`, and `tools.manual_instruction_entry.load_benchmark_records()`.
- Produces: `build_catalog() -> dict` with keys `summary`, `scenarios`, `object_details`, and `coverage`.

- [ ] **Step 1: Write failing catalogue tests**

Tests must assert exact source counts, scenario constraints, unique coverage, action separation, approved VLA detail equality, non-empty traceability fields, and initial review status:

```python
catalog = build_catalog()
self.assertEqual(catalog["summary"]["original_unique_objects"], 123)
self.assertEqual(catalog["summary"]["vla_objects"], 82)
self.assertEqual(catalog["summary"]["total_objects"], 205)
self.assertGreaterEqual(len(catalog["scenarios"]), 45)
self.assertLessEqual(len(catalog["scenarios"]), 60)
self.assertTrue(all(3 <= row["object_count"] <= 8 for row in catalog["scenarios"]))
self.assertTrue(all(row["task_count"] >= 2 for row in catalog["scenarios"]))
self.assertEqual(len({(row["source"], row["object_id"]) for row in catalog["object_details"]}), 205)
self.assertTrue(all(row["original_actions"] for row in catalog["object_details"]))
self.assertTrue(all(4 <= len(row["detailed_actions"]) <= 8 for row in catalog["object_details"]))
self.assertTrue(all(row["review_status"] == "待审核" for row in catalog["object_details"]))
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_all_object_coordination_catalog
```

Expected: import failure because `tools.build_all_object_coordination_catalog` does not exist.

- [ ] **Step 3: Implement authoritative object extraction**

Read original task memberships, deduplicate by object ID while preserving all owning tasks, choose a canonical executable task/action record, and read all VLA IDs with approved original and detailed sequences. Fail with a count-specific `ValueError` unless counts are exactly 123, 82, and 205.

- [ ] **Step 4: Implement scenario blueprints and assignment**

Create 45–60 curated scenarios across cleaning, cooking, storage, delivery, laundry, bedroom, work, bathroom, appliance, security, elderly care, entertainment, installation, and waste/maintenance. Assign objects deterministically by semantic task/object groups, then validate 3–8 objects and at least two task categories per scenario. Add related support objects where required to make a scenario cross-task, without changing unique coverage counting.

- [ ] **Step 5: Implement per-object action detail generation**

For original objects, preserve existing standard actions in `original_actions` and generate 4–8 additional steps from task/operation-specific rule templates. For VLA objects, preserve `template_steps` as original actions and copy the approved detailed steps exactly. Populate subtask instruction, prerequisites, control parameters, phase, mode, dependencies, target state, conflict rule, test conditions, test steps, expected output, qualification, and review fields.

- [ ] **Step 6: Run tests and confirm GREEN**

Run the Task 1 unittest command and expect all tests to pass.

- [ ] **Step 7: Export deterministic JSON**

Run:

```powershell
.\.venv\Scripts\python.exe tools\build_all_object_coordination_catalog.py
```

Expected: prints 45–60 scenario count, `123/123`, `82/82`, and `205/205`, then writes the JSON file.

### Task 2: Build the Three-Sheet Review Workbook

**Files:**
- Create: `tools/build_all_object_coordination_workbook.mjs`
- Generate: `outputs/019fb5d7-4c5a-7e23-bb89-f7ec997f7068/全物体多设备协调联动命令与动作解析审核表.xlsx`

**Interfaces:**
- Consumes: `all_object_coordination_catalog.json` from Task 1.
- Produces: one XLSX workbook with sheets `多设备联动命令表`, `逐物体动作解析测试表`, and `覆盖审核表`.

- [ ] **Step 1: Initialize the artifact-tool workspace**

Use `codex_app__load_workspace_dependencies`, create a writable builder directory, and create a Windows junction named `node_modules` pointing to the bundled dependency directory. Import `Workbook` and `SpreadsheetFile` only from `@oai/artifact-tool`.

- [ ] **Step 2: Build the command sheet**

Write title and coverage summary rows, then a filterable table with scenario ID, category, name, natural-language command, task/object counts, original/VLA object lists, execution phases, modes, triggers, safety/conflict rules, expected overall output, qualification, and review fields. Freeze headers, wrap long text, use category/status colors, and apply review-status validation values `待审核`, `通过`, and `需修改`.

- [ ] **Step 3: Build the per-object parsing sheet**

Write at least 205 unique object rows with the screenshot-aligned columns `测试场景工况`, `测试操作步骤`, and `预期输出结果`, plus source, task, ID, Chinese name, subtask instruction, operation, original actions, additional detailed actions, prerequisites, parameters, phase/mode, dependencies, target state, conflict rule, qualification, and review fields. Color original/VLA rows differently and preserve line breaks in numbered action sequences.

- [ ] **Step 4: Build the formula-driven audit sheet**

List all 205 unique source/ID pairs and use formulas referencing `逐物体动作解析测试表` to calculate appearance count, required-field presence, and coverage status. Display summary formulas for original, VLA, and total coverage; conditionally format missing or incomplete rows red and complete rows green.

- [ ] **Step 5: Export the workbook**

Save exactly one workbook at the specified output path. Do not export alternate versions.

### Task 3: Verify Data, Formulas, and Visual Layout

**Files:**
- Verify: `outputs/019fb5d7-4c5a-7e23-bb89-f7ec997f7068/全物体多设备协调联动命令与动作解析审核表.xlsx`

**Interfaces:**
- Consumes: final workbook from Task 2.
- Produces: evidence that counts, formulas, and all three rendered sheets are correct and legible.

- [ ] **Step 1: Inspect key ranges**

Use `workbook.inspect` on representative header/summary/data ranges of all three sheets. Confirm scenario count, 205 detail rows, formulas, review statuses, and representative original/VLA actions.

- [ ] **Step 2: Scan formula errors**

Use a regex match inspection for `#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A`; expected result is zero unexpected errors.

- [ ] **Step 3: Render every sheet**

Render the title/header plus representative data ranges for all three sheets at readable scale. Inspect images for clipped headers, unreadable text, excessive widths, missing colors, or broken formulas.

- [ ] **Step 4: Apply focused formatting fixes**

Patch only affected widths, heights, wrapping, freezes, or colors; re-export and re-render changed sheets until all key content is legible.

- [ ] **Step 5: Run final catalogue verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_all_object_coordination_catalog
.\.venv\Scripts\python.exe -m compileall -q tools\build_all_object_coordination_catalog.py tests\test_all_object_coordination_catalog.py
```

Expected: all catalogue tests pass and compileall exits 0.
