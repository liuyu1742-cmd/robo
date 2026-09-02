import json
import re
from dataclasses import dataclass
from pathlib import Path

from tools.encode import OBJECT_VOCAB, TASK_TARGETS, TASK_VOCAB


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "meta" / "compliance_catalog_15x120.json").open("r", encoding="utf-8") as file:
    _catalog = json.load(file)
TASK_NAMES_ZH = {task["name_zh"]: task["id"] for task in _catalog["tasks"]}
OBJECT_TO_TASK = {
    obj: task["id"]
    for task in _catalog["tasks"]
    for obj in task["objects"]
}


@dataclass(frozen=True)
class ParsedInstruction:
    task: str
    object: str
    target: str


def _compact(text):
    return text.strip().lower().replace(" ", "").replace("_", "")


def _keyword_maps():
    """Load free-form Chinese keywords lazily to avoid a module import cycle."""
    try:
        from tools.manual_instruction_entry import OBJECT_SYNONYMS, TASK_KEYWORDS
    except Exception:  # noqa: BLE001 - parser still supports canonical ids.
        return {}, {}
    return TASK_KEYWORDS, OBJECT_SYNONYMS


def _score_terms(text, terms):
    compact = _compact(text)
    matched = [
        term for term in terms
        if term and _compact(term) and _compact(term) in compact
    ]
    return sum(max(2, len(_compact(term))) for term in matched)


def _match_tasks(instruction, normalized):
    task_keywords, _ = _keyword_maps()
    candidates = {}

    # Chinese display names and canonical English ids are still supported, but
    # user text can also trigger a task through natural keywords such as
    # "洗杯子", "递水", or "关窗".
    for name_zh, task_id in TASK_NAMES_ZH.items():
        score = _score_terms(instruction, [name_zh])
        if score:
            candidates[task_id] = max(candidates.get(task_id, 0), score)
    for task in TASK_VOCAB:
        if re.search(
            rf"(?<![a-z0-9_]){re.escape(task.lower())}(?![a-z0-9_])",
            normalized,
        ):
            candidates[task] = max(candidates.get(task, 0), len(task) + 2)
        score = _score_terms(instruction, task_keywords.get(task, []))
        if score:
            candidates[task] = max(candidates.get(task, 0), score)

    if not candidates:
        return []
    best_score = max(candidates.values())
    return [task for task, score in candidates.items() if score == best_score]


def _match_objects(instruction, normalized, task=None):
    _, object_synonyms = _keyword_maps()
    candidates = {}
    for obj in OBJECT_VOCAB:
        score = 0
        if obj.lower() in normalized:
            score = max(score, len(obj) + 2)
        score = max(score, _score_terms(instruction, object_synonyms.get(obj, [])))
        if score:
            if task and OBJECT_TO_TASK.get(obj) == task:
                score += 1000
            candidates[obj] = score

    if not candidates:
        return []
    best_score = max(candidates.values())
    return [obj for obj, score in candidates.items() if score == best_score]


def parse_instruction(instruction):
    normalized = instruction.strip().lower()
    task_matches = list(dict.fromkeys(_match_tasks(instruction, normalized)))
    object_matches = _match_objects(instruction, normalized)
    if len(task_matches) > 1 and object_matches:
        object_task = OBJECT_TO_TASK.get(object_matches[0])
        if object_task in task_matches:
            task_matches = [object_task]
    task = task_matches[0] if len(task_matches) == 1 else None
    if task:
        object_matches = _match_objects(instruction, normalized, task)
    if len(task_matches) != 1:
        raise ValueError(f"Expected one task in instruction, found: {task_matches}")
    if not object_matches:
        raise ValueError("No supported object found in instruction")
    return ParsedInstruction(
        task=task,
        object=object_matches[0],
        target=TASK_TARGETS[task],
    )
