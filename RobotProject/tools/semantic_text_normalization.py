"""Label-free Chinese concept normalization shared by training and inference."""

from __future__ import annotations

import re
import unicodedata

from tools.manual_instruction_entry import OBJECT_SYNONYMS, TASK_KEYWORDS


def _chinese_terms(values):
    return [
        str(value).strip()
        for value in values
        if str(value).strip()
        and any("\u4e00" <= char <= "\u9fff" for char in str(value))
    ]


def _object_replacements() -> tuple[tuple[str, str], ...]:
    replacements = []
    for aliases in OBJECT_SYNONYMS.values():
        terms = _chinese_terms(aliases)
        if not terms:
            continue
        # Use the shortest ordinary Chinese head concept.  This deliberately
        # collapses specific surfaces across catalogue entries (e.g. two cup
        # labels share “杯”), so the token cannot itself encode one answer.
        canonical = min(terms, key=lambda term: (len(term), term))
        replacements.extend((term, canonical) for term in terms if term != canonical)
    return tuple(sorted(set(replacements), key=lambda pair: (-len(pair[0]), pair)))


# Shared lexical concepts.  These are deliberately short and reusable across
# many objects/tasks; none represents a catalogue intent or scene composition.
_ACTION_GROUPS = (
    ("清洁", ("擦净", "擦洗", "刷洗", "洗净", "洗妥", "清洗", "去污", "除尘去渍")),
    ("整理", ("归整", "归置", "归位", "摆齐", "放妥", "安置妥当", "分区放妥")),
    ("摆放", ("布置", "铺陈", "置于", "放到", "摆稳", "放稳")),
    ("准备", ("备妥", "备齐", "配齐", "准备好")),
    ("检查", ("核验", "确认", "巡查", "巡检", "复查", "试验")),
    ("调节", ("调好", "调成", "切换", "进入所需工作模式")),
    ("烹饪", ("下厨", "做饭", "烹调", "烹制", "制作晚饭")),
    ("递送", ("拿来", "拿到", "送到", "取用")),
    ("收纳", ("储藏", "储物", "装入", "分类收好")),
    ("护理", ("洗护", "打理", "照料")),
    ("安装", ("装牢", "悬挂", "固定")),
    ("协同", ("统筹", "协调", "联动")),
)

_STRUCTURE_REPLACEMENTS = (
    ("摆放妥当", "摆放"),
    ("布置好", "摆放"),
    ("统筹起来", "协同"),
    ("相关物品按用途逐一协调", "协同"),
    ("你来安排各步骤", "协同"),
    ("按顺序执行", "协同"),
)


def _lexical_replacements() -> tuple[tuple[str, str], ...]:
    pairs = list(_STRUCTURE_REPLACEMENTS)
    for canonical, variants in _ACTION_GROUPS:
        pairs.extend((variant, canonical) for variant in variants if variant != canonical)
    return tuple(sorted(set(pairs), key=lambda pair: (-len(pair[0]), pair)))


_OBJECT_REPLACEMENTS = _object_replacements()
_LEXICAL_REPLACEMENTS = _lexical_replacements()


def normalize_for_semantic_model(text: str) -> str:
    """Return deterministic label-free standard Chinese concept text."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("把", "").replace("将", "")
    for source, target in _OBJECT_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    for source, target in _LEXICAL_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)
    return normalized


def normalization_resource_terms() -> tuple[str, ...]:
    """Expose auditable source/target terms, never answer identifiers."""
    keyword_terms = [term for values in TASK_KEYWORDS.values() for term in values]
    return tuple(
        sorted(
            {
                *(term for pair in _OBJECT_REPLACEMENTS for term in pair),
                *(term for pair in _LEXICAL_REPLACEMENTS for term in pair),
                *keyword_terms,
            }
        )
    )
