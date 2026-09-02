"""Schema and isolation gates for the independent spoken-action v2 dataset.

This module deliberately validates annotations; it never generates annotation
text or reads blind-evaluation fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.detailed_action_catalog import load_detailed_action_catalog


RECORD_FIELDS = frozenset(
    {
        "sample_id",
        "text",
        "label_id",
        "result_type",
        "annotation_source",
        "object_mentions",
        "operation_evidence",
        "safety_class",
        "expected_executable",
    }
)
SPLITS = ("train", "dev", "independent_test")
POSITIVE_QUOTAS = {"train": 10, "dev": 3, "independent_test": 3}
SPLIT_TOTALS = {"train": 3080, "dev": 942, "independent_test": 942}
SAFETY_QUOTAS = {"train": 60, "dev": 20, "independent_test": 20}
SAFETY_CLASSES = frozenset(
    {
        "cancel_negation",
        "restrictive_negation_or_in_word_boundary",
        "information_question",
        "completed_statement",
        "same_name_object_ambiguity",
        "object_operation_incompatibility",
        "out_of_domain",
        "low_confidence_near_candidate",
        "multi_clause_turn_correction",
    }
)
SAFETY_EXECUTABLE_QUOTAS = {
    "cancel_negation": {"train": 0, "dev": 0, "independent_test": 0},
    "restrictive_negation_or_in_word_boundary": {
        "train": 30,
        "dev": 10,
        "independent_test": 10,
    },
    "information_question": {"train": 0, "dev": 0, "independent_test": 0},
    "completed_statement": {"train": 0, "dev": 0, "independent_test": 0},
    "same_name_object_ambiguity": {"train": 0, "dev": 0, "independent_test": 0},
    "object_operation_incompatibility": {"train": 0, "dev": 0, "independent_test": 0},
    "out_of_domain": {"train": 0, "dev": 0, "independent_test": 0},
    "low_confidence_near_candidate": {"train": 0, "dev": 0, "independent_test": 0},
    "multi_clause_turn_correction": {
        "train": 30,
        "dev": 10,
        "independent_test": 10,
    },
}
LONG_NGRAM_LENGTH = 8


def normalize_text(text: str) -> str:
    """Return the Unicode-normalized form used for collision checks."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _error(split: str, sample_id: object, reason: str) -> str:
    return f"split={split} sample_id={sample_id}: {reason}"


def _catalog_labels(catalog: Mapping[str, Any]) -> dict[str, str]:
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        return {}
    return {
        entry["label"]: entry["entry_type"]
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("label"), str)
        and isinstance(entry.get("entry_type"), str)
    }


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def _normalized_spans(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        normalized
        for item in value
        if isinstance(item, str) and (normalized := normalize_text(item))
    ]


def _expected_result_type(entry_type: str) -> str:
    return "scene_coordination" if entry_type == "scene" else "object_detailed_action"


def validate_split(
    records: Sequence[Mapping[str, Any]], split: str, catalog: Mapping[str, Any]
) -> list[str]:
    """Return schema, catalog, provenance, and local-collision errors for a split."""
    if split not in SPLITS:
        return [_error(str(split), "<split-summary>", "unknown split")]
    if not isinstance(records, list):
        return [_error(split, "<split-summary>", "records must be a list")]

    labels = _catalog_labels(catalog)
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_texts: dict[str, tuple[str, object]] = {}
    required_source = f"independent_annotator_{split}"
    for index, record in enumerate(records):
        sample_id = record.get("sample_id", f"<index-{index}>") if isinstance(record, Mapping) else f"<index-{index}>"
        if not isinstance(record, Mapping):
            errors.append(_error(split, sample_id, "schema record must be an object"))
            continue
        record_keys = set(record)
        if record_keys != RECORD_FIELDS:
            missing = sorted(RECORD_FIELDS - record_keys)
            extra = sorted(record_keys - RECORD_FIELDS)
            errors.append(_error(split, sample_id, f"schema fields must match exactly; missing={missing} extra={extra}"))
        if not isinstance(sample_id, str) or not sample_id.strip():
            errors.append(_error(split, sample_id, "schema sample_id must be a non-empty string"))
        elif sample_id in seen_ids:
            errors.append(_error(split, sample_id, f"duplicate sample_id (first seen at index {seen_ids[sample_id]})"))
        else:
            seen_ids[sample_id] = index
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(_error(split, sample_id, "schema text must be a non-empty string"))
            normalized_text = ""
        else:
            normalized_text = normalize_text(text)
            if not normalized_text:
                errors.append(_error(split, sample_id, "schema text normalizes to empty"))
            elif normalized_text in seen_texts:
                prior_id = seen_texts[normalized_text][1]
                errors.append(_error(split, sample_id, f"duplicate normalized text (conflicts with {prior_id})"))
            else:
                seen_texts[normalized_text] = (split, sample_id)
        if record.get("annotation_source") != required_source:
            errors.append(_error(split, sample_id, f"annotation_source must be {required_source}"))
        if not _is_string_list(record.get("object_mentions")):
            errors.append(_error(split, sample_id, "schema object_mentions must be a list of non-empty strings"))
        if not _is_string_list(record.get("operation_evidence")):
            errors.append(_error(split, sample_id, "schema operation_evidence must be a list of non-empty strings"))
        if not isinstance(record.get("expected_executable"), bool):
            errors.append(_error(split, sample_id, "schema expected_executable must be boolean"))
        safety_class = record.get("safety_class")
        if safety_class is not None and safety_class not in SAFETY_CLASSES:
            errors.append(_error(split, sample_id, "safety_class is not recognized"))

        label_id = record.get("label_id")
        result_type = record.get("result_type")
        evidence = record.get("operation_evidence")
        mentions = record.get("object_mentions")
        executable = record.get("expected_executable")
        if label_id is None:
            if result_type != "clarification_required":
                errors.append(_error(split, sample_id, "rejected record result_type must be clarification_required"))
            if executable is not False:
                errors.append(_error(split, sample_id, "rejected record expected_executable must be false"))
            if evidence != []:
                errors.append(_error(split, sample_id, "rejected record operation_evidence must be empty"))
            continue
        if not isinstance(label_id, str) or label_id not in labels:
            errors.append(_error(split, sample_id, "label_id is not in the catalog"))
        elif result_type != _expected_result_type(labels[label_id]):
            errors.append(_error(split, sample_id, "positive record has an incorrect successful result_type"))
        if executable is not True:
            errors.append(_error(split, sample_id, "positive record expected_executable must be true"))
        mention_spans = _normalized_spans(mentions)
        evidence_spans = _normalized_spans(evidence)
        if not mention_spans:
            errors.append(_error(split, sample_id, "positive record requires object_mentions"))
        elif not any(span in normalized_text for span in mention_spans):
            errors.append(_error(split, sample_id, "positive record requires an object_mentions span grounded in text"))
        if not evidence_spans:
            errors.append(_error(split, sample_id, "positive record requires operation_evidence"))
        elif not any(span in normalized_text for span in evidence_spans):
            errors.append(_error(split, sample_id, "positive record requires an operation_evidence span grounded in text"))
    return errors


def _skeleton(record: Mapping[str, Any]) -> str:
    value = normalize_text(str(record["text"]))
    mentions = sorted(
        set(_normalized_spans(record.get("object_mentions"))),
        key=len,
        reverse=True,
    )
    for mention in mentions:
        value = value.replace(mention, "<object>")
    return value


def _long_ngrams(text: str) -> set[str]:
    if len(text) < LONG_NGRAM_LENGTH:
        return set()
    return {
        text[index : index + LONG_NGRAM_LENGTH]
        for index in range(len(text) - LONG_NGRAM_LENGTH + 1)
    }


def validate_dataset(
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
    independent_test: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> list[str]:
    """Return all v2 schema, count, and cross-split isolation failures."""
    raw_by_split = {"train": train, "dev": dev, "independent_test": independent_test}
    valid_top_level = {split: isinstance(records, list) for split, records in raw_by_split.items()}
    errors = [
        error
        for split, records in raw_by_split.items()
        for error in validate_split(records, split, catalog)
    ]
    catalog_labels = _catalog_labels(catalog)
    if len(catalog_labels) != 254:
        errors.append(_error("catalog", "<catalog-summary>", f"expected 254 catalog labels, got {len(catalog_labels)}"))

    all_ids: dict[str, tuple[str, Mapping[str, Any]]] = {}
    all_texts: dict[str, tuple[str, Mapping[str, Any]]] = {}
    core_positive_by_label: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    executable_by_label: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for split, records in raw_by_split.items():
        if not valid_top_level[split]:
            continue
        if len(records) != SPLIT_TOTALS[split]:
            errors.append(_error(split, "<split-summary>", f"expected {SPLIT_TOTALS[split]} records, got {len(records)}"))
        safety_counts = Counter(
            record.get("safety_class")
            for record in records
            if isinstance(record, Mapping) and record.get("safety_class") is not None
        )
        for safety_class in sorted(SAFETY_CLASSES):
            if safety_counts[safety_class] != SAFETY_QUOTAS[split]:
                errors.append(_error(split, "<split-summary>", f"expected {SAFETY_QUOTAS[split]} {safety_class} samples, got {safety_counts[safety_class]}"))
        executable_safety_counts = Counter(
            record.get("safety_class")
            for record in records
            if isinstance(record, Mapping)
            and record.get("safety_class") is not None
            and record.get("expected_executable") is True
        )
        rejection_safety_counts = Counter(
            record.get("safety_class")
            for record in records
            if isinstance(record, Mapping)
            and record.get("safety_class") is not None
            and record.get("expected_executable") is False
        )
        for safety_class in sorted(SAFETY_CLASSES):
            expected_executable = SAFETY_EXECUTABLE_QUOTAS[safety_class][split]
            expected_rejections = SAFETY_QUOTAS[split] - expected_executable
            if executable_safety_counts[safety_class] != expected_executable:
                errors.append(_error(split, "<split-summary>", f"safety_class {safety_class} requires {expected_executable} executable samples, got {executable_safety_counts[safety_class]}"))
            if rejection_safety_counts[safety_class] != expected_rejections:
                errors.append(_error(split, "<split-summary>", f"safety_class {safety_class} requires {expected_rejections} rejection samples, got {rejection_safety_counts[safety_class]}"))
        for record in records:
            if not isinstance(record, Mapping):
                continue
            sample_id = record.get("sample_id", "<missing>")
            if isinstance(sample_id, str) and sample_id in all_ids:
                other_split, other = all_ids[sample_id]
                errors.append(_error(split, sample_id, f"duplicate global sample_id (conflicts with {other_split}/{other.get('sample_id')})"))
            elif isinstance(sample_id, str):
                all_ids[sample_id] = (split, record)
            text = record.get("text")
            if isinstance(text, str) and normalize_text(text):
                normalized = normalize_text(text)
                if normalized in all_texts:
                    other_split, other = all_texts[normalized]
                    errors.append(_error(split, sample_id, f"duplicate global normalized text (conflicts with {other_split}/{other.get('sample_id')})"))
                else:
                    all_texts[normalized] = (split, record)
            label = record.get("label_id")
            if isinstance(label, str) and label in catalog_labels and record.get("expected_executable") is True:
                executable_by_label[label][split].append(record)
                if record.get("safety_class") is None:
                    core_positive_by_label[label][split].append(record)

    for label in sorted(catalog_labels):
        core_label_splits = core_positive_by_label[label]
        for split, expected in POSITIVE_QUOTAS.items():
            if not valid_top_level[split]:
                continue
            actual = len(core_label_splits[split])
            if actual != expected:
                errors.append(_error(split, "<split-summary>", f"label {label} requires {expected} positive samples, got {actual}"))
        split_records = [
            (split, record)
            for split, records in executable_by_label[label].items()
            for record in records
        ]
        for index, (left_split, left) in enumerate(split_records):
            for right_split, right in split_records[index + 1 :]:
                if left_split == right_split:
                    continue
                if _skeleton(left) == _skeleton(right):
                    errors.append(_error(right_split, right.get("sample_id", "<missing>"), f"expression skeleton overlaps {left_split}/{left.get('sample_id')} for label {label}"))
                overlap = _long_ngrams(normalize_text(str(left.get("text", "")))) & _long_ngrams(normalize_text(str(right.get("text", ""))))
                if overlap:
                    errors.append(_error(right_split, right.get("sample_id", "<missing>"), f"long n-gram overlap with {left_split}/{left.get('sample_id')} for label {label}: {sorted(overlap)[0]}"))
    return errors


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_dataset(
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
    independent_test: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Validate and write split JSON plus a byte-accurate SHA-256 manifest."""
    catalog = load_detailed_action_catalog()
    errors = validate_dataset(train, dev, independent_test, catalog)
    if errors:
        raise ValueError("dataset validation failed:\n" + "\n".join(errors))
    destination = Path(output_dir)
    split_data = {"train": train, "dev": dev, "independent_test": independent_test}
    manifest: dict[str, Any] = {"format": "detailed_action_spoken_v2", "splits": {}}
    payloads = {split: _json_bytes(records) for split, records in split_data.items()}
    for split, payload in payloads.items():
        _atomic_write(destination / f"{split}.json", payload)
        manifest["splits"][split] = {
            "records": len(split_data[split]),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    _atomic_write(destination / "manifest.json", _json_bytes(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate detailed-action spoken v2 data.")
    parser.add_argument("--validate-split", nargs=2, metavar=("SPLIT", "PATH"))
    args = parser.parse_args(argv)
    if args.validate_split is None:
        parser.error("--validate-split is required")
    split, path_string = args.validate_split
    try:
        raw_records = json.loads(Path(path_string).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"could not read split JSON: {error}\n")
    errors = validate_split(raw_records, split, load_detailed_action_catalog())
    if errors:
        print("INVALID")
        print("\n".join(errors))
        return 1
    print(f"VALID split={split} records={len(raw_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
