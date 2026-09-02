# Natural Colloquial Command Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 500 reproducible, natural Chinese household commands without mechanical “按……方式操作” wording or implausible location/action combinations, while keeping real full-chain accuracy above 80%.

**Architecture:** Add a focused command-quality module that validates forbidden phrasing and semantic location conflicts. Replace global random-location concatenation with task-aware legacy phrasing and rule-based VLA operation naturalization; the case generator rejects invalid candidates before adding them to the frozen test set. Existing runner, GUI, scoring, and seven audit artifacts remain unchanged.

**Tech Stack:** Python 3.11, `random.Random`, regular expressions, `unittest`, existing legacy/VLA catalogues and full-chain model runtime.

## Global Constraints

- Default count remains exactly `500` and default seed remains exactly `20260813`.
- Same catalogue, count, and seed must produce identical unique instructions.
- No generated instruction may contain “按……方式操作”, “按照……方式”, “具体是”, “场景中”, “口语变体”, or “第几种说法”.
- Bedrooms, studies, and living rooms may be object source locations but not the execution location for washing clothes, washing dishes, or cooking.
- Bedding may not be placed on desks, in kitchens, at doors, or in hallways.
- VLA source/destination paths must not be overwritten by random locations.
- Existing videos, task/object labels, standard sequences, and VLA source records remain unchanged.
- The 500-case semantic match rate must remain at least 90%.
- The final real 500-case full-chain accuracy must be strictly greater than 80%.
- Existing GUI presentation and seven audit artifacts remain compatible.

---

### Task 1: Command Quality Validator

**Files:**
- Create: `tools/action_parser_command_quality.py`
- Test: `tests/test_action_parser_command_quality.py`

**Interfaces:**
- Consumes: one generated Chinese instruction and optional `task`, `object_id`, `operation` metadata.
- Produces: `command_quality_violations(instruction: str, *, task: str, object_id: str, operation: str) -> list[str]` and `is_acceptable_command(...) -> bool`.

- [ ] **Step 1: Write failing forbidden-phrase and conflict tests**

```python
def test_rejects_mechanical_operation_wording():
    assert "mechanical_wording" in command_quality_violations(
        "按关闭烤箱门的方式操作厨房里的烤箱",
        task="cooking_heating", object_id="vla_012", operation="关闭烤箱门",
    )

def test_rejects_bedroom_as_laundry_execution_location():
    assert "location_action_conflict" in command_quality_violations(
        "在卧室里清洗羊毛衫",
        task="laundry", object_id="wool_garment", operation="wash",
    )

def test_accepts_source_to_execution_location_sentence():
    assert not command_quality_violations(
        "把卧室衣柜里的羊毛衫拿到洗衣机里洗一下",
        task="laundry", object_id="wool_garment", operation="wash",
    )

def test_rejects_bedding_on_desktop():
    assert "object_location_conflict" in command_quality_violations(
        "把桌面上的被子整理好",
        task="bedroom_service", object_id="quilt", operation="organize",
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_command_quality -v`

Expected: import failure because `tools.action_parser_command_quality` does not exist.

- [ ] **Step 3: Implement exact validator rules**

```python
FORBIDDEN_PATTERNS = (
    r"按.+的方式操作", r"按照.+方式", r"具体是", r"场景中",
    r"口语变体", r"第\d+种说法",
)

def command_quality_violations(instruction, *, task, object_id, operation):
    violations = []
    if any(re.search(pattern, instruction) for pattern in FORBIDDEN_PATTERNS):
        violations.append("mechanical_wording")
    if task == "laundry" and re.search(r"(?:在|就到?)(?:卧室|书房|客厅)里(?:清洗|洗)", instruction):
        violations.append("location_action_conflict")
    if object_id in BEDDING_OBJECTS and any(term in instruction for term in DESKTOP_OR_INVALID_BEDDING_LOCATIONS):
        violations.append("object_location_conflict")
    return violations
```

Use task-specific regular expressions that distinguish “卧室里的物体” from “在卧室里执行清洗”; export a boolean wrapper that returns `not violations`.

- [ ] **Step 4: Run validator tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_command_quality -v`

Expected: all validator tests pass.

### Task 2: Task-Aware Legacy Command Generation

**Files:**
- Modify: `tools/action_parser_demo_cases.py`
- Modify: `tests/test_action_parser_demo_cases.py`

**Interfaces:**
- Consumes: legacy relation metadata from `_legacy_relations()` and validator from Task 1.
- Produces: natural `_legacy_instruction(relation, variant, rng)` strings with plausible sources/destinations.

- [ ] **Step 1: Write failing natural legacy examples and 500-case scans**

```python
def test_laundry_moves_item_to_washer_instead_of_washing_in_bedroom():
    relation = legacy_relation("laundry", "wool_garment")
    values = {_legacy_instruction(relation, i, random.Random(i)) for i in range(20)}
    assert all("洗衣机" in value for value in values)
    assert not any(re.search(r"在?卧室里(?:清洗|洗)", value) for value in values)

def test_500_commands_have_no_quality_violations():
    cases = generate_demo_cases(500, 20260813)
    assert all(not command_quality_violations(
        case.instruction, task=case.expected_task,
        object_id=case.expected_object, operation=case.expected_operation,
    ) for case in cases)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_cases -v`

Expected: failures showing global `LOCATIONS` creates implausible combinations.

- [ ] **Step 3: Replace global location concatenation with task-aware builders**

Implement builders with explicit semantics:

```python
def _legacy_core(relation, obj, rng):
    task = relation["task"]
    if task == "laundry":
        source = rng.choice(("卧室衣柜里", "换衣篮里", "床边篮子里"))
        return f"把{source}的{obj}拿到洗衣机里洗一下"
    if task == "bedroom_service":
        return f"把床上的{obj}整理好"
    if task == "smart_cooking":
        return f"在厨房用{obj}做饭"
    return TASK_DIRECT_BUILDERS[task](obj, relation["operation"], rng)
```

Remove `LOCATIONS` from legacy generation. Keep request prefixes and polite suffixes only when their concatenation remains grammatical. Before returning, validate the candidate and regenerate a different phrasing when violations are present.

- [ ] **Step 4: Run legacy and coverage tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_cases -v`

Expected: 500 unique deterministic cases, 15 legacy tasks, and zero quality violations.

### Task 3: Natural VLA Operation Conversion

**Files:**
- Modify: `tools/action_parser_demo_cases.py`
- Modify: `tools/vla_action_templates.py`
- Test: `tests/test_action_parser_demo_cases.py`
- Test: `tests/test_vla_action_template_routing.py`

**Interfaces:**
- Consumes: VLA `object_name`, `source_operation_label`, task, source row, and operation-specific override table.
- Produces: `_naturalize_vla_goal(relation) -> str` and natural `_vla_instruction(...)` strings.

- [ ] **Step 1: Write failing VLA naturalization tests**

```python
def test_vla_arrow_path_becomes_direct_request():
    assert _naturalize_vla_goal({
        "object_name": "盘子", "source_operation_label": "台面→橱柜",
    }) == "把台面上的盘子收进橱柜"

def test_complex_onion_instruction_is_natural():
    value = _naturalize_vla_goal(vla_relation("vla_006"))
    assert "用削皮刀把洋葱切成丁" in value
    assert "刀和砧板用完放回水槽" in value
    assert "按" not in value and "具体是" not in value
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_cases -v`

Expected: failures because current templates preserve raw labels and mechanical wording.

- [ ] **Step 3: Implement structured label conversion and reviewed overrides**

Create exact natural overrides for complex/multi-object VLA labels and reusable rules for common forms:

```python
if label.startswith("关闭"):
    return label.replace("关闭", "把", 1) + "关好"
if label == "台面→橱柜":
    return f"把台面上的{object_name}收进橱柜"
if label == "橱柜→台面":
    return f"把橱柜里的{object_name}拿到台面上"
```

For labels containing semicolon-separated secondary objects or three-stage paths, use an `object_id` override dictionary so the final request describes every required movement in natural order. `_vla_instruction` adds only natural request/politeness variants and never adds random locations.

Add the same candidate-specific natural intent phrases to VLA template matching. Duplicate object names are disambiguated with ordinary context rather than task IDs or raw labels; for example, the cooking relation uses “做完饭把烤箱门关好”, while the storage-open/close relation uses “烤箱不用了，把门关好”. The selector scores these reviewed aliases before broad object-name matching, so full-chain routing remains production-realistic.

- [ ] **Step 4: Run all generator and VLA routing tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_action_parser_demo_cases tests.test_vla_action_template_routing -v`

Expected: all tests pass, full operation labels still disambiguate duplicate VLA relations through internal metadata/routing support without exposing mechanical input wording.

### Task 4: Static Quality, Semantic Accuracy, and Real Full-Chain Acceptance

**Files:**
- Verify: `tools/action_parser_command_quality.py`
- Verify: `tools/action_parser_demo_cases.py`
- Runtime output: `outputs/action_parser_demo/<timestamp>_seed20260813_n500/`

**Interfaces:**
- Consumes: completed generator, real parser/model checkpoint, fixed count and seed.
- Produces: new seven-file real result directory with accuracy greater than 80%.

- [ ] **Step 1: Run complete automated tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_project_runtime tests.test_action_parser_command_quality tests.test_action_parser_demo_cases tests.test_action_parser_demo_runner tests.test_action_parser_demo_output tests.test_action_parser_500case_demo tests.test_vla_action_template_routing`

Expected: all tests pass.

- [ ] **Step 2: Scan the frozen 500 commands**

Generate 500 cases with seed `20260813` and assert: 500 unique strings; zero forbidden patterns; zero location conflicts; no benchmark sentence copies; at least 15 legacy tasks and 10 VLA tasks; semantic match count at least 450.

- [ ] **Step 3: Run a real 50-case smoke test**

Run: `.\.venv\Scripts\python.exe action_parser_500case_demo.py --no-gui --count 50 --seed 20260813`

Expected: 50 completed records and no batch-level crash.

- [ ] **Step 4: Run the real 500-case test**

Run: `.\.venv\Scripts\python.exe action_parser_500case_demo.py --no-gui --count 500 --seed 20260813`

Expected: `completed=500`, closed failure totals, seven output artifacts, and `accuracy_percent > 80.0`.

- [ ] **Step 5: If accuracy is not greater than 80%, diagnose before changing rules**

Group failures by category/task/object/template, inspect actual parser/model fields, add a failing regression test for the dominant confirmed cause, implement one root-cause fix, and repeat Steps 1–4. Do not weaken scoring or rewrite expected results.

- [ ] **Step 6: Restart the GUI and verify the visible command list**

Launch `action_parser_500case_demo.py`, preview seed `20260813`, and confirm the current command and result table contain no mechanical wording or implausible location/action pair.
