# Coordination Fan-out Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every coordination benchmark invoke all scene object operations and report parent, child, and complete coordination timing after an excluded warm-up.

**Architecture:** Extend the runtime case generator with deterministic object subcommands, then extend the batch runner with nested child-call evidence while preserving existing result fields. `run_batch` performs one excluded warm-up after parser construction. Summary/output/UI consume the new timing fields without changing the model, command datasets, or old entries.

**Tech Stack:** Python 3.11, dataclasses, `time.perf_counter_ns`, Tkinter, JSON/JSONL/CSV, unittest, local BGE/CUDA.

## Global Constraints

- Every successful coordination scene invokes every object returned by that scene, never only the first two.
- A scene with fewer than two objects fails.
- Child commands use catalogue structure and runtime grammar only; never read train, dev, or frozen independent-test text.
- Formal result timing excludes model construction, warm-up, command generation, UI refresh, and output writes.
- Coordination total time equals parent resolve time plus all child resolve times.
- Child failures propagate to the coordination result, but remaining children still run.
- Existing result fields remain available for compatibility.

---

### Task 1: Deterministic child-operation command generation

**Files:**
- Modify: `tools/detailed_action_random_cases.py`
- Modify: `tests/test_detailed_action_random_cases.py`

**Interfaces:**
- Consumes: a catalogue object label and stable integer seed.
- Produces: `generate_object_subcommand(object_label: str, seed: int, catalogue: Mapping[str, Any] | None = None) -> RandomDetailedActionCase`.

- [ ] **Step 1: Add failing tests**

```python
def test_child_subcommand_is_deterministic_and_matches_object_label(self):
    first = generate_object_subcommand("object:vla:vla_080", 17)
    second = generate_object_subcommand("object:vla:vla_080", 17)
    self.assertEqual(first, second)
    self.assertEqual(first.case_type, "object")
    self.assertEqual(first.expected_label, "object:vla:vla_080")

def test_child_subcommand_rejects_scene_and_unknown_labels(self):
    with self.assertRaises(ValueError):
        generate_object_subcommand("scene:MDC-001", 1)
    with self.assertRaises(KeyError):
        generate_object_subcommand("object:missing", 1)
```

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_cases`

Expected: import failure for `generate_object_subcommand`.

- [ ] **Step 3: Implement minimal generator**

Index object entries by label, select the requested entry, seed a local `random.Random`, reuse `_spoken_object`, and return a `RandomDetailedActionCase` whose ID is `CHILD-{seed}`. Do not load any dataset split.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_cases`

Expected: all case tests pass.

### Task 2: Nested child calls and exact timing

**Files:**
- Modify: `tools/detailed_action_random_runner.py`
- Modify: `tests/test_detailed_action_random_runner.py`

**Interfaces:**
- Consumes: parent `RandomDetailedActionCase`, parser, catalogue, and `generate_object_subcommand`.
- Produces: `ChildOperationResult`, plus `scene_parse_elapsed_ns`, `child_operation_count`, `child_operation_total_elapsed_ns`, and `child_operations` on `RandomDetailedActionResult`.

- [ ] **Step 1: Add a failing N+1 call/timing test**

Use a fake parser returning one scene with three object labels, then three correct object results. Use a fake clock yielding parent `100→1100`, then children `2000→4000`, `5000→8000`, and `9000→13000`. Assert four parser calls, three child records, parent time `1000`, child total `9000`, and coordination `elapsed_ns == 10000`.

- [ ] **Step 2: Add failing propagation tests**

Assert a wrong second child label changes the parent failure reason to `child_operation_failed` while the third child still runs. Assert a one-object scene becomes `coordination_requires_multiple_objects` and performs no child call.

- [ ] **Step 3: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_runner`

Expected: missing nested timing fields and call behavior.

- [ ] **Step 4: Implement nested execution**

Add:

```python
@dataclass(frozen=True)
class ChildOperationResult:
    index: int
    instruction: str
    expected_label: str
    actual_result_type: str
    actual_label: str
    passed: bool
    failure_reason: str
    elapsed_ns: int
    detailed_actions: tuple[str, ...]
```

Time the parent first. Only after a correct scene result, generate all child commands outside timed regions and call `parser.resolve()` once per object. Compute coordination `elapsed_ns` as the parent duration plus child durations. Keep independent behavior unchanged.

- [ ] **Step 5: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_runner`

Expected: all runner tests pass.

### Task 3: Excluded parser warm-up

**Files:**
- Modify: `detailed_action_random_visual_test.py`
- Modify: `tests/test_detailed_action_random_visual_test.py`

**Interfaces:**
- Consumes: constructed keyword-enhanced parser and a deterministic object subcommand.
- Produces: `warm_up_parser(parser: Any) -> dict[str, Any]`; emits a `warmup` event but no formal result row.

- [ ] **Step 1: Add a failing warm-up ordering test**

Use a recording fake parser and `run_batch`. Assert the first resolve call is the warm-up, formal `result` events still equal requested count, and summary completed count excludes warm-up.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: call count/order assertion fails.

- [ ] **Step 3: Implement warm-up**

After parser construction, generate one object subcommand with fixed seed `20260828`, call `resolve()` once, and emit `{type: "warmup", instruction, result_type}`. Do this before `run_random_cases`; never append it to output results.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: requested counts remain unchanged and warm-up is first.

### Task 4: Summary and evidence output

**Files:**
- Modify: `tools/detailed_action_random_runner.py`
- Modify: `tools/detailed_action_random_output.py`
- Modify: `tests/test_detailed_action_random_runner.py`
- Modify: `tests/test_detailed_action_random_output.py`

**Interfaces:**
- Consumes: nested timing fields from Task 2.
- Produces: scene summary keys `average_scene_parse_elapsed_ms`, `average_child_operation_elapsed_ms`, `average_child_total_elapsed_ms`; nested JSONL and flattened CSV evidence.

- [ ] **Step 1: Add failing summary/output assertions**

Create two scene results with known parent/child times. Assert the three new averages. Finalize an output bundle and assert CSV contains `scene_parse_elapsed_ms`, `child_operation_count`, `child_operation_total_elapsed_ms`, and `child_operations`, while JSONL preserves child dictionaries.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_runner tests.test_detailed_action_random_output`

Expected: missing summary and CSV fields.

- [ ] **Step 3: Implement summaries and serialization**

Aggregate only scene rows. The per-child average divides total child time by total child-call count; the child-total average divides by scene-row count. `to_dict()` recursively serializes child dataclasses. CSV writes a readable string per child: `index|instruction|expected|actual|passed|elapsed_ms`.

- [ ] **Step 4: Update Markdown report timing explanation**

State that coordination time is N+1 model calls and list all four scene timing metrics. Preserve old keys and acceptance metadata.

- [ ] **Step 5: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_runner tests.test_detailed_action_random_output`

Expected: all tests pass.

### Task 5: GUI child evidence and timing cards

**Files:**
- Modify: `detailed_action_random_visual_test.py`
- Modify: `tests/test_detailed_action_random_visual_test.py`

**Interfaces:**
- Consumes: scene summary fields and row `child_operations`.
- Produces: cards for scene parse average, child-call average and coordination total average; selected scene detail includes every child command/time/action.

- [ ] **Step 1: Add failing presentation tests**

Pass a scene row with two child-operation dictionaries into `result_detail()`. Assert the actions text includes both child commands, both elapsed values and both action sequences. Inspect `_build` source and assert the new timing labels exist and obsolete “联动平均耗时” does not.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: missing child detail and labels.

- [ ] **Step 3: Implement presentation**

Rename `scene_avg` to `coordination_total_avg`, add `scene_parse_avg` and `child_operation_avg`, and populate them from summary. For scene rows, `result_detail()` renders one numbered child block per child with command, label, status, time and detailed actions.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_detailed_action_random_visual_test`

Expected: all visual tests pass.

### Task 6: Regression and real CUDA evidence

**Files:**
- Verify: all changed files
- Produce: `outputs/detailed_action_random_visual_test/run_*`

**Interfaces:**
- Consumes: completed implementation.
- Produces: auditable real batch proving child call counts and timing arithmetic.

- [ ] **Step 1: Run focused/full regression and compile**

Run all random visual/case/runner/output/BGE/isolation tests, then compile changed modules. Expected: zero failures and exit code 0.

- [ ] **Step 2: Run a CUDA batch**

Run: `.\.venv\Scripts\python.exe detailed_action_random_visual_test.py --no-gui --count 20 --seed 20260905 --device cuda`

Expected: ten coordination rows, every successful coordination row has at least two child calls, and all timing arithmetic validates from JSONL.

- [ ] **Step 3: Validate evidence mathematically**

For each scene row assert `elapsed_ns == scene_parse_elapsed_ns + child_operation_total_elapsed_ns` and `child_operation_count == len(child_operations)`. Print independent and coordination averages from `summary.json`; coordination must use the complete N+1 timing field rather than the parent-only field.

- [ ] **Step 4: Perform hidden Tkinter smoke check**

Select a scene row and assert the upper detail text contains every child operation. Confirm the new timing cards populate from summary.

