# Group Cleaning Rows in Task Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every row assigned to the unified `cleaning` task appear as one uninterrupted block in the household task-object workbook.

**Architecture:** The workbook remains the sole edited artifact. Its existing rows are read, sorted by a fixed task order with `cleaning` first, and written back with their original cell values and formatting retained. No image, video, annotation, metadata, or filesystem path is modified.

**Tech Stack:** Node.js; `@oai/artifact-tool`; Excel `.xlsx`.

## Global Constraints

- Modify only `C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx`.
- Keep the existing sheet, headers, table styling, filters, numeric values, and object paths.
- Do not change any task, object, image, video, annotation, or metadata value.
- Place all `task_id = cleaning` records contiguously at the beginning of the data rows.

---

### Task 1: Reorder the workbook rows and verify the result

**Files:**
- Modify: `C:\RobotProject\RobotProject\docs\家庭服务任务物体数据表.xlsx`
- Create: `C:\RobotProject\spreadsheet_work\group_cleaning_rows.mjs`

**Interfaces:**
- Consumes: sheet `任务物体数据表`, range `A1:I139`.
- Produces: the same sheet and row count, with rows 2 through 24 all having `task_id = cleaning`.

- [x] **Step 1: Import and inspect the existing workbook**

Run the artifact-tool builder to read the workbook and inspect `任务物体数据表!A1:I139`; confirm there are 138 data rows and 23 rows whose first column is `cleaning`.

- [x] **Step 2: Reorder values without changing row contents**

Use a stable sort: all records whose first cell is `cleaning` first; the remaining records ordered by their existing first appearance of `task_id`; records within a task retain their original order. Copy the original formatted data rows to a temporary in-workbook-safe matrix and write the reordered 138 values back into `A2:I139` only.

- [x] **Step 3: Verify data integrity and grouping**

Inspect `A1:I30`, confirm rows 2–24 are `cleaning` and row 25 is a different task. Compare each row’s nine cell values before and after as a multiset, confirm 138 records both before and after, and scan for spreadsheet formula errors.

- [x] **Step 4: Render visual output and export**

Render the start of the sheet at normal viewing scale, visually confirm the header and the single uninterrupted cleaning block are legible, then save the modified workbook at its original path.
