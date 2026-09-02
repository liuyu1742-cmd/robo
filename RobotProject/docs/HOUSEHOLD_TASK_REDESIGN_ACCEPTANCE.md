# Household task redesign acceptance

## Result

PASS. `run_acceptance_check.py` reports all formal gates true:

- task hierarchy
- unique object scope
- 135 task-object relations
- local reference integrity
- exact canonical instruction benchmark identity and >=90% action accuracy

## Canonical scope

- Acceptance tasks: 15
- Unique objects: 124
- Task-object relations: 135
- Cleaning associations: 24
- Maintenance associations: 8
- Entertainment associations: 6
- Legacy cleaning IDs map to `cleaning` and remain recorded as subtask detail.

## Formal benchmark

The benchmark contains 139 canonical relation cases covering exactly the canonical 15 task IDs and all 124 unique objects. Runtime parsing and planning achieved 139/139 exact identities and action sequences (100%). Missing task IDs and unexpected task IDs are both empty.

## Reference integrity

- Reference directories/rows: 3/38
- Copied reference media or annotations: 0
- Reference paths outside `datasets/midterm_15task_delivery`: 0
- Source totals are not inflated by reference rows.
- `split_leak_check_applicable=false` because reference-only views contain no train/validation/test metadata. No zero split-leak claim is made.

## Regression protection

The full test suite checks the synchronized 124-object scope. Tests explicitly reject stale task-ID sets, an undersized benchmark that omits shared relations, relation-key/digest drift, source-task actions on new-task references, copied reference media, out-of-scope paths, misleading split semantics, and canonical/legacy object-count conflation.

The formal gate requires exactly 135 cases and exact benchmark/report/catalog relation-key equality. All three use digest `30e503e21e8e65fb5af427b32b094edec976c2c5aecf0c71c2a81e41e4306687`. New-task runtime and generated artifacts share one canonical professional action-plan function.
