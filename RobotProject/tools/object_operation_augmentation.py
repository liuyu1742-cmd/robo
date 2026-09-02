"""Deterministic train-only augmentation for object-operation compatibility."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping


_NEGATION_PREFIX = re.compile(r"^(?:不得|禁止|不能|不要|不应|不可|请勿)")
_COMMAND_PREFIXES = ("请", "麻烦", "现在")


def _row_id(kind: str, label: str, text: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{label}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"train-ontology-{kind}-{digest}"


def _command_variants(phrase: str, count: int) -> list[str]:
    phrase = phrase.strip().rstrip("。！？!?；;")
    if not phrase or count < 1:
        return []
    return [f"{_COMMAND_PREFIXES[index % len(_COMMAND_PREFIXES)]}{phrase}。" for index in range(count)]


def _forbidden_capability_phrases(entry: Mapping[str, object]) -> list[str]:
    name = str(entry["object_name_zh"])
    phrases: list[str] = []
    for raw in entry.get("forbidden_actions", ()):
        phrase = _NEGATION_PREFIX.sub("", str(raw).strip())
        if phrase and "无关动作" not in phrase:
            phrases.append(phrase)

    # Capability flags provide natural, affirmative mismatches without using
    # held-out examples.  They deliberately describe capability families,
    # rather than memorising any evaluation utterance.
    if entry.get("powered") is False:
        phrases.extend((
            f"打开{name}的电源开关",
            f"把{name}的运行档位调到三档",
            f"让{name}切换到节能运行模式",
            f"用{name}打印一张状态清单",
            f"把{name}的系统升级到最新版本",
        ))
    if entry.get("mechanism_capable") is False:
        phrases.extend((
            f"给{name}设定一条自动巡航路线",
            f"把{name}的运行速度调高一档",
        ))
    if entry.get("container_capable") is False:
        phrases.extend((f"把{name}清空", f"往{name}里装满清水", f"把{name}封口"))
    if entry.get("heat_capable") is False:
        phrases.append(f"把{name}的加热温度设为一百八十度")
    if entry.get("blade_capable") is False:
        phrases.append(f"用{name}切开一块木板")

    return list(dict.fromkeys(phrases))


def build_ontology_augmentation_rows(
    entries: Iterable[Mapping[str, object]], *, templates_per_phrase: int = 2
) -> list[dict]:
    """Build labelled synthetic rows from the train-approved capability ontology.

    No dev/test split is accepted by this interface, making accidental held-out
    text use impossible at the augmentation boundary.
    """
    rows: list[dict] = []
    for entry in sorted(entries, key=lambda item: str(item["label_id"])):
        label = str(entry["label_id"])
        name = str(entry["object_name_zh"])
        positive_phrases = list(dict.fromkeys(str(value) for value in entry.get("operation_phrases", ())))
        for phrase in positive_phrases:
            for text in _command_variants(phrase, templates_per_phrase):
                rows.append({
                    "sample_id": _row_id("positive", label, text),
                    "text": text,
                    "label_id": label,
                    "result_type": "object_detailed_action",
                    "annotation_source": "train_ontology_augmentation_v1",
                    "object_mentions": [name],
                    "operation_evidence": [phrase],
                    "safety_class": None,
                    "expected_executable": True,
                })
        for phrase in _forbidden_capability_phrases(entry):
            for text in _command_variants(phrase, templates_per_phrase):
                rows.append({
                    "sample_id": _row_id("negative", label, text),
                    "text": text,
                    "label_id": None,
                    "result_type": "clarification_required",
                    "annotation_source": "train_ontology_augmentation_v1",
                    "object_mentions": [name],
                    "operation_evidence": [],
                    "safety_class": "object_operation_incompatibility",
                    "expected_executable": False,
                })
    return rows
