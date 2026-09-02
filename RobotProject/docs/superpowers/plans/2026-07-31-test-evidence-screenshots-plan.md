# Test Evidence Screenshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate six browser screenshots from verified benchmark JSON and actual timing source code, then embed them with evidence explanations into the updated DOCX report.

**Architecture:** A focused Python generator reads the existing benchmark, 31 evidence JSON files, and the actual benchmark timing source, verifies timing arithmetic and SHA-256, and writes one self-contained local HTML dashboard. The in-app browser captures six fixed dashboard sections as PNGs. The DOCX updater embeds those PNGs and explanatory paragraphs after the existing evidence table.

**Tech Stack:** Python 3.11, `json`, `hashlib`, HTML/CSS/SVG, in-app browser screenshots, `python-docx`, `unittest`.

## Global Constraints

- Screenshot values must be derived from `outputs/decision_planning_benchmark_20260731.json`.
- Every evidence hash and timing equation must validate before HTML generation.
- No external JavaScript, font, image, or network dependency.
- Generate exactly six PNG screenshots.
- Preserve `C:\Users\sjtu101\Desktop\3.docx`.
- Keep the existing test-case and result table headers unchanged.
- Explain each image and the experimental result in Chinese.

---

### Task 1: Define Verifiable Dashboard Output

**Files:**
- Create: `tests/test_decision_evidence_visuals.py`
- Create: `tools/build_decision_evidence_dashboard.py`

**Interfaces:**
- Consumes: benchmark JSON and evidence directory.
- Produces: `build_dashboard(benchmark_path, evidence_directory, output_html) -> dict`.

- [ ] **Step 1: Write the failing test**

Test that the generated HTML contains six uniquely identified screenshot sections, all 30 run values, the replan value, three multi-device subtasks, the timing formula, the actual timing code, and no external resource URL.

- [ ] **Step 2: Run test to verify it fails**

Run:
`.\.venv\Scripts\python.exe -m unittest tests.test_decision_evidence_visuals`

Expected: import failure because `tools.build_decision_evidence_dashboard` does not exist.

- [ ] **Step 3: Implement evidence verification and HTML generation**

Implement:

```python
def build_dashboard(
    benchmark_path: Path,
    evidence_directory: Path,
    output_html: Path,
) -> dict:
    """Verify evidence and write a self-contained HTML dashboard."""
```

The function must recompute every `elapsed_ns`, `elapsed_seconds`, and output SHA-256 before writing HTML.

- [ ] **Step 4: Run test to verify it passes**

Run the Task 1 unittest command and expect `OK`.

### Task 2: Capture Six Browser Screenshots

**Files:**
- Generate: `outputs/decision_evidence_dashboard/index.html`
- Generate: `outputs/decision_evidence_screenshots/01_test_overview.png`
- Generate: `outputs/decision_evidence_screenshots/02_vla_run_evidence.png`
- Generate: `outputs/decision_evidence_screenshots/03_multi_device_run_evidence.png`
- Generate: `outputs/decision_evidence_screenshots/04_multi_device_subtasks.png`
- Generate: `outputs/decision_evidence_screenshots/05_timing_chart.png`
- Generate: `outputs/decision_evidence_screenshots/06_timing_code_and_output.png`

**Interfaces:**
- Consumes: six HTML sections with IDs `shot-overview`, `shot-vla`, `shot-multi`, `shot-subtasks`, `shot-chart`, and `shot-code`.
- Produces: six non-empty PNGs captured directly from those elements.

- [ ] **Step 1: Generate the dashboard**

Run the dashboard generator against the final benchmark and 31 evidence files.

- [ ] **Step 2: Serve it locally**

Start a hidden, workspace-scoped HTTP server bound to `127.0.0.1`.

- [ ] **Step 3: Capture each section**

Open the local page in the in-app browser and capture each target section at full element bounds.

- [ ] **Step 4: Inspect every PNG**

Verify dimensions, text legibility, no clipping, correct case IDs, correct averages, and correct three-subtask output.

### Task 3: Embed Screenshots and Explanations in DOCX

**Files:**
- Modify: `tools/update_decision_test_docx.py`
- Modify: `tests/test_decision_test_docx_output.py`
- Regenerate: `outputs/3_项目测试更新版.docx`

**Interfaces:**
- Consumes: six verified PNG paths.
- Produces: section `3.2.6.6 图像化测试证据与结果解释`.

- [ ] **Step 1: Write the failing DOCX structural test**

Require six inline shapes, the new section heading, six figure titles, source explanations, timing interpretation, and the software-only execution boundary statement.

- [ ] **Step 2: Run the test and confirm failure**

Expected: no `3.2.6.6` section and zero added screenshot images.

- [ ] **Step 3: Implement minimal DOCX embedding**

Insert each image at readable width, followed by a Chinese explanation containing its data source, reading method, supported conclusion, and evidence-file relationship.

- [ ] **Step 4: Regenerate and test**

Run the DOCX updater and all targeted tests.

### Task 4: Final Verification

**Files:**
- Verify all files above.

- [ ] **Step 1: Recompute evidence integrity**

Confirm 31/31 evidence files satisfy timing arithmetic and SHA-256.

- [ ] **Step 2: Run regression tests**

Run all multi-device, VLA, benchmark, visual, runtime, and DOCX output tests.

- [ ] **Step 3: Compile**

Run `compileall` on all changed Python files.

- [ ] **Step 4: Render DOCX if available**

Use the packaged renderer. If LibreOffice remains unavailable, report the limitation and rely on direct inspection of all six PNGs plus structural DOCX validation.
