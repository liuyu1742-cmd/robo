"""Safety-preserving keyword rescue above the BGE detailed-action parser."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from tools.detailed_action_catalog import load_detailed_action_catalog
from tools.detailed_action_random_cases import _TASK_VERBS


_RESCUABLE_REASONS = {
    "BGE 执行门控未达到校准阈值",
    "短对象指令置信余量不足，请补充位置或用途",
    "对象与操作或用途不兼容",
    "识别到对象但缺少可执行操作或用途",
}


def _compact(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _task_terms(task: str) -> tuple[str, ...]:
    verb, tail = _TASK_VERBS.get(task, ("", ""))
    terms = {verb, tail, verb.replace("一下", ""), tail.removesuffix("好")}
    return tuple(_compact(term) for term in terms if _compact(term))


class KeywordEnhancedDetailedActionParser:
    """Rescue only explicit, catalogue-compatible object commands.

    All semantic safety decisions remain owned by the base parser.  Only three
    low-confidence/compatibility clarification reasons are eligible; negation,
    questions, completed statements, ambiguity and out-of-domain requests can
    never be rescued here.
    """

    def __init__(self, base_parser: Any, *, catalogue: Mapping[str, Any] | None = None) -> None:
        self.base_parser = base_parser
        catalog = catalogue if catalogue is not None else load_detailed_action_catalog()
        self.object_entries = [entry for entry in catalog["entries"] if entry["entry_type"] == "object"]

    def resolve(self, instruction: str) -> dict[str, Any]:
        original = self.base_parser.resolve(instruction)
        if original.get("result_type") != "clarification_required":
            result = dict(original)
            result.setdefault("resolution_backend", "bge")
            return result
        if original.get("reason") not in _RESCUABLE_REASONS:
            return dict(original)

        text = _compact(instruction)
        named = [
            entry for entry in self.object_entries
            if _compact(entry["object_name_zh"]) and _compact(entry["object_name_zh"]) in text
        ]
        if not named:
            return dict(original)
        longest = max(len(_compact(entry["object_name_zh"])) for entry in named)
        named = [entry for entry in named if len(_compact(entry["object_name_zh"])) == longest]
        compatible = [
            entry for entry in named
            if any(term in text for term in _task_terms(str(entry.get("task", ""))))
        ]
        if len(compatible) != 1:
            return dict(original)
        entry = compatible[0]
        return {
            "result_type": "object_detailed_action",
            "input": str(instruction),
            "label": entry["label"],
            "confidence": 1.0,
            "margin": 1.0,
            "model_used": True,
            "resolution_backend": "keyword_catalog_rescue",
            "object_id": entry["object_id"],
            "object_name_zh": entry["object_name_zh"],
            "task": entry["task"],
            "detailed_actions": list(entry["detailed_actions"]),
            "control_parameters": entry["control_parameters"],
            "base_clarification_reason": original.get("reason", ""),
        }
