"""Offline Qwen candidate generation for the Task 2C semantic pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".superpowers" / "sdd" / "bge-task-2c-pilot"
REPORT_PATH = PROJECT_ROOT / ".superpowers" / "sdd" / "bge-task-2c-pilot-report.md"
SINGLE_DRAFT_OUTPUT_DIR = (
    PROJECT_ROOT / ".superpowers" / "sdd" / "bge-task-2d-drafts"
)
SINGLE_DRAFT_REPORT_PATH = (
    PROJECT_ROOT / ".superpowers" / "sdd" / "bge-task-2d-single-draft-report.md"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained" / "Qwen2.5-7B-Instruct"
CATALOG_PATH = PROJECT_ROOT / "meta" / "detailed_action_entry_catalog.json"
GUIDELINES_PATH = (
    PROJECT_ROOT / "datasets" / "detailed_action_spoken_v2" / "annotation_guidelines.md"
)

CORE_LABELS = (
    "object:original:blender",
    "object:original:electric_kettle",
    "object:original:induction_cooker",
    "object:vla:vla_011",
    "object:original:battery",
    "object:original:first_aid_kit",
    "object:original:remote_control",
    "object:original:window",
    "scene:MDC-009",
    "scene:MDC-028",
    "scene:MDC-045",
    "scene:MDC-047",
)

SINGLE_DRAFT_LABELS = CORE_LABELS[:10]

REQUIREMENT_CORRECTIONS = [
    {
        "brief_label": "object:original:stove",
        "pilot_label": "object:vla:vla_011",
        "reason": "the reviewed 254-label catalog contains only object:vla:vla_011 for 炉灶",
    }
]


SAFETY_CLASSES = (
    "cancel_negation",
    "restrictive_negation_or_in_word_boundary",
    "information_question",
    "completed_statement",
    "same_name_object_ambiguity",
    "object_operation_incompatibility",
    "out_of_domain",
    "low_confidence_near_candidate",
    "multi_clause_turn_correction",
)

_MIXED_SAFETY_CLASSES = {
    "restrictive_negation_or_in_word_boundary",
    "multi_clause_turn_correction",
}

_SAFETY_SEMANTICS = {
    "cancel_negation": "取消、否定或撤回先前动作；全部应拒绝执行",
    "restrictive_negation_or_in_word_boundary": "区分真正限制性否定与词内部/非否定用法",
    "information_question": "只询问信息，不发出动作命令",
    "completed_statement": "陈述动作已经完成，不要求再次执行",
    "same_name_object_ambiguity": "同名对象无法唯一定位，需要澄清",
    "object_operation_incompatibility": "对象与所述操作在现实中不兼容",
    "out_of_domain": "请求不属于家居机器人可执行动作域",
    "low_confidence_near_candidate": "接近命令但信息不足以可靠确定动作",
    "multi_clause_turn_correction": "多分句中后一句纠正前一句，区分最终可执行意图与需澄清情况",
}


def safety_mix(safety_class: str, count: int) -> tuple[int, int]:
    """Scale the guidelines' train executable/rejection ratio to ``count``."""

    if safety_class not in SAFETY_CLASSES:
        raise ValueError(f"unknown safety class: {safety_class}")
    if count <= 0:
        raise ValueError("candidate count must be positive")
    executable = count // 2 if safety_class in _MIXED_SAFETY_CLASSES else 0
    return executable, count - executable


def build_safety_prompt(
    safety_class: str,
    count: int,
    executable_contexts: list[dict[str, Any]],
) -> str:
    """Build a safety-class-specific prompt with the guideline mix encoded."""

    executable, rejection = safety_mix(safety_class, count)
    compact_contexts = [
        {
            "label": item["label"],
            "object_name": item.get("object_name_zh"),
            "task": item.get("task"),
            "catalog_actions": item.get("detailed_actions", []),
        }
        for item in executable_contexts
    ]
    if executable:
        index_plan = (
            f"第 1 至 {executable} 项固定 expected_executable=true；"
            f"第 {executable + 1} 至 {count} 项固定 expected_executable=false。"
        )
    else:
        index_plan = f"第 1 至 {count} 项全部固定 expected_executable=false。"
    return f"""你是独立安全边界候选撰写员。只生成 safety_class={safety_class} 的自然口语中文候选。
该类语义：{_SAFETY_SEMANTICS[safety_class]}
可执行候选可用的已审 catalog 上下文：{json.dumps(compact_contexts, ensure_ascii=False, sort_keys=True)}

总共生成 {count} 条，其中恰好 {executable} 条 expected_executable=true，恰好 {rejection} 条 expected_executable=false。可执行项必须使用上面真实 label_id，并在 text 中给出直接对象 span 和直接操作 span。拒绝项的 label_id 必须为 null，operation_evidence 必须为空数组，而且不得暗含任何可执行动作；object_mentions 可为空或保留 text 中直接提及的对象。
严格按数组索引落实配额：{index_plan}

span 是硬性机械约束：object_mentions 和非空 operation_evidence 的每个值，都必须从同一项 text 中逐字复制一个连续片段，不能概括、扩写或换同义词。例如 text="把窗户打开" 时应写 "object_mentions":["窗户"],"operation_evidence":["打开"]。

仅返回严格 JSON 数组，不要 Markdown 或解释。每一项都必须包含以下 6 个字段，不得遗漏 semantic_rationale：
{{"text":"...","label_id":null或真实标签,"expected_executable":true或false,"object_mentions":[],"operation_evidence":[],"semantic_rationale":"简短语义理由"}}
输出前自行核对数组长度恰好为 {count}、每项六字段齐全、索引配额正确；最终绝不输出 Markdown、备注或任何数组外文字。候选必须符合本类语义并自然多样，不得靠增删礼貌词制造变体。"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derive_draft_seed(base_seed: int, label: str, draft_index: int) -> int:
    """Derive a stable torch-compatible seed for one label draft."""

    if draft_index < 0:
        raise ValueError("draft_index must be non-negative")
    digest = hashlib.sha256(
        f"{base_seed}\0{label}\0{draft_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in folded if character.isalnum())


def _candidate_id(batch_label: str, seed: int, index: int, text: str) -> str:
    digest = sha256_text(f"{batch_label}\0{seed}\0{index}\0{text}")[:16]
    return f"candidate-{digest}"


def parse_batch_response(
    raw_response: str,
    *,
    batch_label: str,
    safety_class: str | None,
    count: int,
    seed: int,
    prompt_sha256: str,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
    known_labels: set[str],
) -> list[dict[str, Any]]:
    """Strictly parse and mechanically validate one model response."""

    raw_hash = sha256_text(raw_response)
    try:
        payload = json.loads(raw_response.strip())
    except (json.JSONDecodeError, UnicodeError) as exc:
        payload = None
        batch_error = f"invalid_json:{exc.msg if isinstance(exc, json.JSONDecodeError) else exc}"
    else:
        batch_error = None
        if not isinstance(payload, list):
            batch_error = "top_level_must_be_array"
        elif len(payload) != count:
            batch_error = f"candidate_count_mismatch:expected={count},actual={len(payload)}"

    source_items = payload if isinstance(payload, list) and len(payload) == count else [None] * count
    records: list[dict[str, Any]] = []
    seen_normalized: set[str] = set()
    required = {
        "text",
        "label_id",
        "expected_executable",
        "object_mentions",
        "operation_evidence",
        "semantic_rationale",
    }
    for index, item in enumerate(source_items):
        reasons: list[str] = []
        if batch_error:
            reasons.append(batch_error)
        if not isinstance(item, dict):
            reasons.append("candidate_must_be_object")
            item = {}
        missing = sorted(required - set(item))
        if missing:
            reasons.append("missing_fields:" + ",".join(missing))

        text = item.get("text") if isinstance(item.get("text"), str) else ""
        label_id = item.get("label_id")
        expected_executable = item.get("expected_executable")
        object_mentions = item.get("object_mentions")
        operation_evidence = item.get("operation_evidence")
        rationale = item.get("semantic_rationale")
        normalized = normalize_text(text)

        if not text.strip():
            reasons.append("text_must_be_nonempty_string")
        if not isinstance(rationale, str) or not rationale.strip():
            reasons.append("semantic_rationale_must_be_nonempty_string")
        if not isinstance(expected_executable, bool):
            reasons.append("expected_executable_must_be_boolean")
        if not isinstance(object_mentions, list) or not all(
            isinstance(value, str) and value for value in object_mentions
        ):
            reasons.append("object_mentions_must_be_string_array")
            object_mentions = []
        if not isinstance(operation_evidence, list) or not all(
            isinstance(value, str) and value for value in operation_evidence
        ):
            reasons.append("operation_evidence_must_be_string_array")
            operation_evidence = []

        if safety_class is None:
            if label_id != batch_label or label_id not in known_labels:
                reasons.append("label_mismatch")
            if expected_executable is not True:
                reasons.append("core_candidate_must_be_executable")
        elif expected_executable is True:
            if not isinstance(label_id, str) or label_id not in known_labels:
                reasons.append("executable_safety_label_invalid")
        elif expected_executable is False:
            if label_id is not None:
                reasons.append("rejection_label_must_be_null")
            if operation_evidence != []:
                reasons.append("rejection_operation_evidence_must_be_empty")

        if expected_executable is True:
            normalized_mentions = [normalize_text(value) for value in object_mentions]
            normalized_evidence = [normalize_text(value) for value in operation_evidence]
            if not any(value and value in normalized for value in normalized_mentions):
                reasons.append("missing_direct_object_span")
            if not any(value and value in normalized for value in normalized_evidence):
                reasons.append("missing_direct_operation_span")

        if normalized:
            if normalized in seen_normalized:
                reasons.append("normalized_duplicate_in_batch")
            seen_normalized.add(normalized)

        records.append(
            {
                "candidate_id": _candidate_id(batch_label, seed, index, text),
                "batch_label": batch_label,
                "label_id": label_id,
                "safety_class": safety_class,
                "expected_executable": expected_executable,
                "text": text,
                "object_mentions": object_mentions,
                "operation_evidence": operation_evidence,
                "semantic_rationale": rationale if isinstance(rationale, str) else "",
                "seed": seed,
                "prompt_sha256": prompt_sha256,
                "raw_response_sha256": raw_hash,
                "generation_parameters": dict(generation_parameters),
                "model_hf_revision": model_revisions["hf"],
                "model_modelscope_revision": model_revisions["modelscope"],
                "parse_status": "rejected" if reasons else "parsed",
                "rejection_reasons": reasons,
            }
        )
    if safety_class is not None:
        expected_true, expected_false = safety_mix(safety_class, count)
        actual_true = sum(record["expected_executable"] is True for record in records)
        actual_false = sum(record["expected_executable"] is False for record in records)
        if (actual_true, actual_false) != (expected_true, expected_false):
            for record in records:
                record["rejection_reasons"].append("safety_mix_mismatch")
                record["parse_status"] = "rejected"
    return records


def parse_single_draft_response(
    raw_response: str,
    *,
    label: str,
    kind: str,
    draft_index: int,
    seed: int,
    prompt_sha256: str,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
    known_labels: set[str],
) -> dict[str, Any]:
    """Strictly parse one model response as one draft, never as a batch."""

    required = {
        "text",
        "label_id",
        "expected_executable",
        "object_mentions",
        "operation_evidence",
        "semantic_rationale",
    }
    reasons: list[str] = []
    try:
        payload = json.loads(raw_response.strip())
    except (json.JSONDecodeError, UnicodeError) as exc:
        payload = None
        reasons.append(
            f"invalid_json:{exc.msg if isinstance(exc, json.JSONDecodeError) else exc}"
        )
    if not isinstance(payload, dict):
        reasons.append("top_level_must_be_object")
        item: dict[str, Any] = {}
    else:
        item = payload
        missing = sorted(required - set(item))
        extra = sorted(set(item) - required)
        if missing:
            reasons.append("missing_fields:" + ",".join(missing))
        if extra:
            reasons.append("unexpected_fields:" + ",".join(extra))

    text = item.get("text") if isinstance(item.get("text"), str) else ""
    label_id = item.get("label_id")
    expected_executable = item.get("expected_executable")
    object_mentions = item.get("object_mentions")
    operation_evidence = item.get("operation_evidence")
    rationale = item.get("semantic_rationale")
    if not text.strip():
        reasons.append("text_must_be_nonempty_string")
    if label_id != label or label_id not in known_labels:
        reasons.append("label_mismatch")
    if expected_executable is not True:
        reasons.append("core_candidate_must_be_executable")
    if not isinstance(rationale, str) or not rationale.strip():
        reasons.append("semantic_rationale_must_be_nonempty_string")
    if not isinstance(object_mentions, list) or not all(
        isinstance(value, str) and value for value in object_mentions
    ):
        reasons.append("object_mentions_must_be_string_array")
        object_mentions = []
    if not isinstance(operation_evidence, list) or not all(
        isinstance(value, str) and value for value in operation_evidence
    ):
        reasons.append("operation_evidence_must_be_string_array")
        operation_evidence = []
    nfkc_text = unicodedata.normalize("NFKC", text)
    if not object_mentions:
        reasons.append("missing_direct_object_span")
    for value in object_mentions:
        if unicodedata.normalize("NFKC", value) not in nfkc_text:
            reasons.append(f"object_span_not_contiguous:{value}")
    if not operation_evidence:
        reasons.append("missing_direct_operation_span")
    for value in operation_evidence:
        if unicodedata.normalize("NFKC", value) not in nfkc_text:
            reasons.append(f"operation_span_not_contiguous:{value}")

    return {
        "candidate_id": _candidate_id(label, seed, draft_index, text),
        "batch_label": label,
        "label_id": label_id,
        "kind": kind,
        "safety_class": None,
        "expected_executable": expected_executable,
        "text": text,
        "object_mentions": object_mentions,
        "operation_evidence": operation_evidence,
        "semantic_rationale": rationale if isinstance(rationale, str) else "",
        "draft_index": draft_index,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "raw_response_sha256": sha256_text(raw_response),
        "generation_parameters": dict(generation_parameters),
        "model_hf_revision": model_revisions["hf"],
        "model_modelscope_revision": model_revisions["modelscope"],
        "parse_status": "rejected" if reasons else "parsed",
        "rejection_reasons": reasons,
    }


def load_local_qwen(
    model_dir: Path,
    *,
    verifier: Callable[[Path], dict[str, Any]] | None = None,
    tokenizer_factory: Any = None,
    model_factory: Any = None,
    torch_module: Any = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Verify then load the pinned model with strict offline CUDA settings."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if verifier is None:
        from tools.download_annotation_generator import verify_directory

        verifier = verify_directory
    manifest = verifier(Path(model_dir))

    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA is required; dtype fallback is forbidden")

    if tokenizer_factory is None or model_factory is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_factory = tokenizer_factory or AutoTokenizer
        model_factory = model_factory or AutoModelForCausalLM

    model_path = str(Path(model_dir))
    tokenizer = tokenizer_factory.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = model_factory.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch_module.bfloat16,
        device_map={"": "cuda"},
    )
    if hasattr(model, "eval"):
        model.eval()
    return tokenizer, model, manifest


class QwenLocalClient:
    """Small injectable wrapper around the verified local Qwen snapshot."""

    def __init__(self, tokenizer: Any, model: Any, torch_module: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        generation_parameters: dict[str, Any],
    ) -> str:
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        messages = [
            {
                "role": "system",
                "content": "你只撰写候选，不签发、不评分、不淘汰候选。严格输出请求的 JSON。",
            },
            {"role": "user", "content": prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs = {name: tensor.to("cuda") for name, tensor in encoded.items()}
        kwargs = dict(generation_parameters)
        kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        with self.torch.inference_mode():
            output = self.model.generate(**model_inputs, **kwargs)
        prompt_length = model_inputs["input_ids"].shape[-1]
        completion = output[0][prompt_length:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


@dataclass(frozen=True)
class BatchSpec:
    label: str
    kind: str
    prompt: str
    safety_class: str | None


def build_single_object_prompt(entry: dict[str, Any]) -> str:
    """Build an object-label prompt for exactly one immutable draft."""

    context = {
        "label": entry["label"],
        "object_name": entry["object_name_zh"],
        "task": entry["task"],
        "catalog_actions": entry["detailed_actions"],
    }
    return f"""你是独立草稿撰写员。只为下面这一个对象标签写一条自然的用户口语命令。
标签上下文：{json.dumps(context, ensure_ascii=False, sort_keys=True)}

只写用户口语命令，不输出细分动作序列。一条命令只表达一个用户目标，不能把多个操作步骤拼成一条。必须依据真实物体能力和日常使用方式，不能让固定设施被搬动或归位；catalog_actions 只用于理解边界，不能照抄、改写或串联成命令。text 必须直接提及目标对象，并含机器人可真实执行的操作证据。

object_mentions 和 operation_evidence 的每个值都必须从 text 中逐字复制连续片段。semantic_rationale 是强制字段，用一句话说明为什么 text 属于该标签，绝不能省略。

仅返回一个严格 JSON object，不要 JSON 数组、Markdown、代码围栏、解释或对象外文字。必须严格照此六字段 schema 输出（semantic_rationale 特意放在前面，确保写出）：
{{"text":"...","semantic_rationale":"为什么这条口语命令属于目标标签","label_id":"{entry['label']}","expected_executable":true,"object_mentions":["text 中的对象原文 span"],"operation_evidence":["text 中的操作原文 span"]}}
输出前逐字段检查：text、label_id、expected_executable、object_mentions、operation_evidence、semantic_rationale 六项一个不少；不得在 semantic_rationale 之前结束 JSON object。"""


def build_single_scene_prompt(entry: dict[str, Any]) -> str:
    """Build a scene-label prompt for exactly one whole-goal draft."""

    context = {
        "label": entry["label"],
        "canonical_goal": entry["command"],
        "object_names": [item["object_name_zh"] for item in entry["objects"]],
    }
    return f"""你是独立草稿撰写员。只为下面这一个多设备场景标签写一条自然的用户口语命令。
场景上下文：{json.dumps(context, ensure_ascii=False, sort_keys=True)}

只写用户口语命令，不输出细分动作序列。一条命令只表达一个整体目标，不能把多个操作步骤拼成一条。命令必须表达 canonical_goal 对应的整体生活目标，不机械枚举对象，不照抄 canonical_goal，不要求搬动固定设施，也不把同义对象当成两个对象。

object_mentions 和 operation_evidence 的每个值都必须从 text 中逐字复制连续片段。semantic_rationale 是强制字段，用一句话说明为什么 text 表达该整体生活目标，绝不能省略。

仅返回一个严格 JSON object，不要 JSON 数组、Markdown、代码围栏、解释或对象外文字。必须严格照此六字段 schema 输出（semantic_rationale 特意放在前面，确保写出）：
{{"text":"...","semantic_rationale":"为什么这条口语命令属于目标标签","label_id":"{entry['label']}","expected_executable":true,"object_mentions":["text 中的对象/目标原文 span"],"operation_evidence":["text 中的操作/目标原文 span"]}}
输出前逐字段检查：text、label_id、expected_executable、object_mentions、operation_evidence、semantic_rationale 六项一个不少；不得在 semantic_rationale 之前结束 JSON object。"""


def build_single_draft_specs(
    entries: list[dict[str, Any]],
    *,
    labels: list[str],
) -> list[BatchSpec]:
    """Build label-isolated, non-safety specs for single-draft generation."""

    by_label = {entry["label"]: entry for entry in entries}
    specs: list[BatchSpec] = []
    for label in labels:
        if label.startswith("safety:") or label in SAFETY_CLASSES:
            raise ValueError(
                "single-draft mode rejects safety specs; use the independent safety annotation task"
            )
        entry = by_label.get(label)
        if entry is None:
            raise ValueError(f"catalog label not found: {label}")
        if entry.get("entry_type") == "object":
            kind = "object"
            prompt = build_single_object_prompt(entry)
        elif entry.get("entry_type") == "scene":
            kind = "scene"
            prompt = build_single_scene_prompt(entry)
        else:
            raise ValueError(f"unsupported catalog entry_type for {label}")
        specs.append(BatchSpec(label=label, kind=kind, prompt=prompt, safety_class=None))
    return specs


def build_batch_specs(
    entries: list[dict[str, Any]],
    *,
    labels: list[str] | None,
    count: int,
) -> list[BatchSpec]:
    """Build one independent prompt per requested core label or safety class."""

    by_label = {entry["label"]: entry for entry in entries}
    requested = list(CORE_LABELS) + [f"safety:{value}" for value in SAFETY_CLASSES]
    if labels is not None:
        requested = [
            value if value.startswith("safety:") else (
                f"safety:{value}" if value in SAFETY_CLASSES else value
            )
            for value in labels
        ]
    executable_contexts = [
        by_label[label]
        for label in CORE_LABELS
        if label in by_label and by_label[label].get("entry_type") == "object"
    ]
    specs: list[BatchSpec] = []
    for label in requested:
        if label.startswith("safety:"):
            safety_class = label.split(":", 1)[1]
            if safety_class not in SAFETY_CLASSES:
                raise ValueError(f"unknown safety class: {safety_class}")
            specs.append(
                BatchSpec(
                    label=label,
                    kind="safety",
                    safety_class=safety_class,
                    prompt=build_safety_prompt(safety_class, count, executable_contexts),
                )
            )
            continue
        entry = by_label.get(label)
        if entry is None:
            raise ValueError(f"catalog label not found: {label}")
        if entry.get("entry_type") == "object":
            prompt = build_object_prompt(entry, count)
            kind = "object"
        elif entry.get("entry_type") == "scene":
            prompt = build_scene_prompt(entry, count)
            kind = "scene"
        else:
            raise ValueError(f"unsupported catalog entry_type for {label}")
        specs.append(BatchSpec(label=label, kind=kind, prompt=prompt, safety_class=None))
    return specs


def _batch_stem(label: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
    return f"{readable}-{sha256_text(label)[:10]}"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_artifact_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


_MANIFEST_BASE_KEYS = {
    "manifest_version",
    "mode",
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "expected_calls",
    "completed_calls",
    "parsed_count",
    "rejected_count",
    "drafts_per_label",
    "base_seed",
    "labels",
    "calls",
    "input_sha256",
    "script_sha256",
    "generated_script_sha256",
    "delivered_script_sha256",
    "model_manifest_sha256",
    "model_revisions",
    "generation_parameters",
    "generation_run_path",
    "generation_run_sha256",
}
_MANIFEST_MIGRATION_KEYS = {
    "parser_migration_audit",
    "superseded_audit",
    "migration",
}
_MANIFEST_CALL_KEYS = {
    "label",
    "kind",
    "draft_index",
    "seed",
    "prompt_sha256",
    "raw_response_sha256",
    "parsed_artifact_sha256",
    "generation_parameters",
    "model_revisions",
    "parse_status",
    "rejection_reasons",
    "resumed",
    "raw_path",
    "parsed_path",
    "provenance_path",
    "raw_file_sha256",
    "parsed_file_sha256",
    "provenance_file_sha256",
    "elapsed_seconds",
}
_MANIFEST_LABEL_KEYS = {
    "label",
    "kind",
    "requested",
    "parsed",
    "rejected",
    "elapsed_seconds",
}
_SIDECAR_KEYS = {
    "provenance_version",
    "label",
    "kind",
    "draft_index",
    "seed",
    "prompt_sha256",
    "raw_response_sha256",
    "parsed_artifact_sha256",
    "generation_parameters",
    "model_revisions",
    "input_sha256",
    "model_manifest_sha256",
    "generated_script_sha256",
    "run_contract_sha256",
    "raw_path",
    "raw_file_sha256",
    "parsed_path",
    "parsed_file_sha256",
}
_RAW_SINGLE_DRAFT_KEYS = {
    "label",
    "kind",
    "draft_index",
    "seed",
    "prompt",
    "prompt_sha256",
    "raw_response",
    "raw_response_sha256",
    "generation_parameters",
    "model_revisions",
    "parsed_artifact_sha256",
}
_PARSED_SINGLE_DRAFT_KEYS = {
    "candidate_id",
    "batch_label",
    "label_id",
    "kind",
    "safety_class",
    "expected_executable",
    "text",
    "object_mentions",
    "operation_evidence",
    "semantic_rationale",
    "draft_index",
    "seed",
    "prompt_sha256",
    "raw_response_sha256",
    "generation_parameters",
    "model_hf_revision",
    "model_modelscope_revision",
    "parse_status",
    "rejection_reasons",
}
_COMPLETION_KEYS = {
    "completion_version",
    "manifest_path",
    "manifest_sha256",
    "run_contract_path",
    "run_contract_sha256",
    "expected_call_count",
    "call_identities",
}
_MIGRATION_PLAN_KEYS = {
    "plan_version",
    "previous_manifest_sha256",
    "previous_manifest_snapshot_path",
    "delivered_script_sha256",
    "pairs",
}
_MIGRATION_PAIR_KEYS = {
    "original_raw_path",
    "migrated_raw_path",
    "raw_sha256",
    "original_parsed_path",
    "migrated_parsed_path",
    "parsed_sha256",
}
_MIGRATION_STATE_KEYS = {
    "state_version",
    "status",
    "plan_sha256",
    "previous_manifest_snapshot_path",
    "completed_pairs",
    "final_manifest_sha256",
}
_SUPERSEDED_AUDIT_KEYS = {
    "directory",
    "raw_count",
    "parsed_count",
    "pairs",
    "reason",
}
_MIGRATION_RECORD_KEYS = {
    "migrated_at",
    "previous_manifest_sha256",
    "previous_manifest_snapshot_path",
    "formal_raw_count",
    "formal_parsed_count",
}
_PARSER_MIGRATION_AUDIT_KEYS = {
    "formal_artifact_count",
    "changed_record_count",
    "delivered_parser_parsed",
    "delivered_parser_rejected",
    "formal_artifacts_rewritten",
}
_DELIVERY_RECORD_KEYS = {
    "delivery_version",
    "script_path",
    "delivered_script_sha256",
    "generated_script_sha256",
    "run_contract_sha256",
}


def _require_exact_keys(payload: Any, expected: set[str], description: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError(f"generation provenance mismatch: invalid {description} schema")


def _completion_record(
    manifest_sha256: str,
    run_contract_sha256: str,
    expected_identities: set[tuple[str, int]],
) -> dict[str, Any]:
    return {
        "completion_version": 1,
        "manifest_path": "draft_manifest.json",
        "manifest_sha256": manifest_sha256,
        "run_contract_path": "generation_run.json",
        "run_contract_sha256": run_contract_sha256,
        "expected_call_count": len(expected_identities),
        "call_identities": [
            {"label": label, "draft_index": draft_index}
            for label, draft_index in sorted(expected_identities)
        ],
    }


def _write_completion_record(
    output_dir: Path,
    run_contract_sha256: str,
    expected_identities: set[tuple[str, int]],
) -> Path:
    manifest_sha256 = sha256_file(output_dir / "draft_manifest.json")
    path = output_dir / "completion" / f"manifest-{manifest_sha256}.json"
    _write_json_immutable(
        path,
        _completion_record(manifest_sha256, run_contract_sha256, expected_identities),
    )
    return path


def _validate_completion_record(
    output_dir: Path,
    manifest_sha256: str,
    run_contract_sha256: str,
    expected_identities: set[tuple[str, int]],
) -> None:
    path = output_dir / "completion" / f"manifest-{manifest_sha256}.json"
    if not path.is_file():
        raise RuntimeError("generation provenance mismatch: missing manifest completion anchor")
    try:
        text = path.read_text(encoding="utf-8")
        record = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("generation provenance mismatch: invalid completion anchor") from exc
    _require_exact_keys(record, _COMPLETION_KEYS, "completion anchor")
    expected = _completion_record(
        manifest_sha256, run_contract_sha256, expected_identities
    )
    if record != expected or text != _json_artifact_text(expected):
        raise RuntimeError("generation provenance mismatch: completion anchor differs")


def _delivery_record(
    delivered_script_sha256: str,
    generated_script_sha256: str,
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "delivery_version": 1,
        "script_path": "tools/generate_spoken_annotation_candidates.py",
        "delivered_script_sha256": delivered_script_sha256,
        "generated_script_sha256": generated_script_sha256,
        "run_contract_sha256": run_contract_sha256,
    }


def _write_delivery_record(
    output_dir: Path,
    delivered_script_sha256: str,
    generated_script_sha256: str,
    run_contract_sha256: str,
) -> Path:
    path = output_dir / "delivery" / f"script-{delivered_script_sha256}.json"
    _write_json_immutable(
        path,
        _delivery_record(
            delivered_script_sha256,
            generated_script_sha256,
            run_contract_sha256,
        ),
    )
    return path


def _validate_delivery_record(
    output_dir: Path,
    delivered_script_sha256: str,
    generated_script_sha256: str,
    run_contract_sha256: str,
) -> None:
    path = output_dir / "delivery" / f"script-{delivered_script_sha256}.json"
    try:
        text = path.read_text(encoding="utf-8")
        record = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("generation provenance mismatch: invalid delivery record") from exc
    _require_exact_keys(record, _DELIVERY_RECORD_KEYS, "delivery record")
    expected = _delivery_record(
        delivered_script_sha256,
        generated_script_sha256,
        run_contract_sha256,
    )
    if record != expected or text != _json_artifact_text(expected):
        raise RuntimeError("generation provenance mismatch: delivery record differs")


def _write_json_immutable(path: Path, payload: Any) -> None:
    """Create a JSON artifact once; matching bytes are reusable, others are fatal."""

    expected = _json_artifact_text(payload).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    try:
        with path.open("xb") as stream:
            stream.write(expected)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(f"immutable artifact mismatch: {path}")


def _write_bytes_immutable(path: Path, expected: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    try:
        with path.open("xb") as stream:
            stream.write(expected)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(f"immutable artifact mismatch: {path}")


def _recover_previous_manifest_bytes(
    manifest: dict[str, Any], current_bytes: bytes
) -> tuple[str, bytes]:
    """Return the exact pre-migration bytes, including historical CRLF encoding."""

    current_sha256 = hashlib.sha256(current_bytes).hexdigest()
    migration = manifest.get("migration")
    recorded_sha256 = (
        migration.get("previous_manifest_sha256")
        if isinstance(migration, dict)
        else None
    )
    if not isinstance(recorded_sha256, str) or recorded_sha256 == current_sha256:
        return current_sha256, current_bytes
    legacy = dict(manifest)
    legacy["manifest_version"] = 1
    for key in (
        "generated_script_sha256",
        "delivered_script_sha256",
        "generation_run_path",
        "generation_run_sha256",
        "superseded_audit",
        "migration",
        "parser_migration_audit",
    ):
        legacy.pop(key, None)
    legacy["calls"] = [
        {
            key: value
            for key, value in call.items()
            if key
            not in {
                "raw_file_sha256",
                "parsed_file_sha256",
                "provenance_path",
                "provenance_file_sha256",
            }
        }
        for call in manifest.get("calls", [])
    ]
    reconstructed = _json_artifact_text(legacy).replace("\n", "\r\n").encode("utf-8")
    if hashlib.sha256(reconstructed).hexdigest() != recorded_sha256:
        raise RuntimeError(
            "cannot reconstruct exact previous manifest bytes from migration record"
        )
    return recorded_sha256, reconstructed


def _single_draft_stem(
    label: str,
    prompt_sha256: str,
    seed: int,
    response_sha256: str,
) -> str:
    return (
        f"{_batch_stem(label)}__prompt-{prompt_sha256}__seed-{seed}"
        f"__response-{response_sha256}"
    )


def _draft_identity_stem(label: str, draft_index: int) -> str:
    return f"{_batch_stem(label)}__draft-{draft_index:03d}"


def _generation_contract(
    *,
    specs: list[BatchSpec],
    drafts_per_label: int,
    base_seed: int,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
    input_hashes: dict[str, str],
    model_manifest_sha256: str,
    script_sha256: str,
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "mode": "single-draft",
        "labels": [
            {
                "label": spec.label,
                "kind": spec.kind,
                "base_prompt_sha256": sha256_text(spec.prompt),
            }
            for spec in specs
        ],
        "drafts_per_label": drafts_per_label,
        "base_seed": base_seed,
        "generation_parameters": dict(generation_parameters),
        "model_revisions": dict(model_revisions),
        "input_sha256": dict(input_hashes),
        "model_manifest_sha256": model_manifest_sha256,
        "generated_script_sha256": script_sha256,
    }


def _load_generation_contract(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"generation provenance mismatch: missing immutable run contract {path}")
    try:
        actual_text = path.read_text(encoding="utf-8")
        actual = json.loads(actual_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"generation provenance mismatch: invalid run contract {path}") from exc
    if actual != expected or actual_text != _json_artifact_text(expected):
        raise RuntimeError("generation provenance mismatch: immutable run contract differs")
    return actual


def _identity_artifacts(output_dir: Path, label: str, draft_index: int) -> list[Path]:
    matches: list[Path] = []
    for directory in ("raw", "parsed"):
        for path in (output_dir / directory).glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"generation provenance mismatch: unreadable artifact {path}"
                ) from exc
            payload_label = payload.get("label", payload.get("batch_label"))
            if payload_label == label and payload.get("draft_index") == draft_index:
                matches.append(path)
    return matches


def _load_resume_manifest(
    output_dir: Path,
    contract: dict[str, Any],
    run_contract_sha256: str,
    expected_identities: set[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    path = output_dir / "draft_manifest.json"
    if not path.is_file():
        return {}
    try:
        manifest_text = path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("generation provenance mismatch: invalid draft manifest") from exc
    allowed_keys = (_MANIFEST_BASE_KEYS, _MANIFEST_BASE_KEYS | _MANIFEST_MIGRATION_KEYS)
    if not isinstance(manifest, dict) or set(manifest) not in allowed_keys:
        raise RuntimeError("generation provenance mismatch: invalid draft manifest schema")
    _validate_completion_record(
        output_dir,
        sha256_file(path),
        run_contract_sha256,
        expected_identities,
    )
    expected = {
        "mode": contract["mode"],
        "drafts_per_label": contract["drafts_per_label"],
        "base_seed": contract["base_seed"],
        "generation_parameters": contract["generation_parameters"],
        "model_revisions": contract["model_revisions"],
        "input_sha256": contract["input_sha256"],
        "model_manifest_sha256": contract["model_manifest_sha256"],
        "script_sha256": contract["generated_script_sha256"],
        "generated_script_sha256": contract["generated_script_sha256"],
        "generation_run_sha256": run_contract_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("generation provenance mismatch: draft manifest differs")
    calls = manifest.get("calls")
    if not isinstance(calls, list):
        raise RuntimeError("generation provenance mismatch: draft manifest calls missing")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for call in calls:
        _require_exact_keys(call, _MANIFEST_CALL_KEYS, "manifest call")
        key = (call.get("label"), call.get("draft_index"))
        if key in indexed:
            raise RuntimeError("generation provenance mismatch: duplicate manifest identity")
        indexed[key] = call
    if set(indexed) != expected_identities:
        raise RuntimeError("generation provenance mismatch: manifest call set differs")
    parsed_count = sum(call["parse_status"] == "parsed" for call in calls)
    rejected_count = sum(call["parse_status"] == "rejected" for call in calls)
    summary_expected = {
        "manifest_version": 2,
        "status": "complete",
        "expected_calls": len(expected_identities),
        "completed_calls": len(calls),
        "parsed_count": parsed_count,
        "rejected_count": rejected_count,
        "generation_run_path": "generation_run.json",
    }
    if any(manifest.get(key) != value for key, value in summary_expected.items()):
        raise RuntimeError("generation provenance mismatch: manifest summary differs")
    expected_delivered_sha256 = contract["generated_script_sha256"]
    if _MANIFEST_MIGRATION_KEYS <= set(manifest):
        expected_delivered_sha256 = manifest.get("delivered_script_sha256")
        if not _is_sha256(expected_delivered_sha256):
            raise RuntimeError("generation provenance mismatch: delivered script SHA differs")
        _validate_delivery_record(
            output_dir,
            expected_delivered_sha256,
            contract["generated_script_sha256"],
            run_contract_sha256,
        )
    if manifest.get("delivered_script_sha256") != expected_delivered_sha256:
        raise RuntimeError("generation provenance mismatch: delivered script SHA differs")
    labels = manifest.get("labels")
    if not isinstance(labels, list):
        raise RuntimeError("generation provenance mismatch: manifest labels missing")
    contract_labels = contract["labels"]
    if len(labels) != len(contract_labels):
        raise RuntimeError("generation provenance mismatch: manifest label set differs")
    for label_summary, contract_label in zip(labels, contract_labels):
        _require_exact_keys(label_summary, _MANIFEST_LABEL_KEYS, "manifest label")
        label_calls = [
            call for call in calls if call["label"] == contract_label["label"]
        ]
        expected_label_summary = {
            "label": contract_label["label"],
            "kind": contract_label["kind"],
            "requested": contract["drafts_per_label"],
            "parsed": sum(call["parse_status"] == "parsed" for call in label_calls),
            "rejected": sum(
                call["parse_status"] == "rejected" for call in label_calls
            ),
        }
        if any(
            label_summary.get(key) != value
            for key, value in expected_label_summary.items()
        ) or not isinstance(label_summary["elapsed_seconds"], (int, float)):
            raise RuntimeError(
                "generation provenance mismatch: manifest label summary differs"
            )
    if not all(
        isinstance(manifest.get(key), str) for key in ("started_at", "finished_at")
    ) or not isinstance(manifest.get("elapsed_seconds"), (int, float)):
        raise RuntimeError("generation provenance mismatch: manifest timing metadata differs")
    if _MANIFEST_MIGRATION_KEYS <= set(manifest):
        _validate_manifest_migration_audit(output_dir, manifest, calls, contract)
    return indexed


def _resume_single_draft(
    *,
    output_dir: Path,
    spec: BatchSpec,
    prompt: str,
    prompt_sha256: str,
    draft_index: int,
    seed: int,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
    known_labels: set[str],
    run_contract_sha256: str,
    contract_input_sha256: dict[str, str],
    contract_model_manifest_sha256: str,
    contract_generated_script_sha256: str,
    manifest_call: dict[str, Any] | None,
) -> tuple[dict[str, Any], Path, Path] | None:
    provenance_path = (
        output_dir / "provenance" / f"{_draft_identity_stem(spec.label, draft_index)}.json"
    )
    if not provenance_path.is_file():
        matches = _identity_artifacts(output_dir, spec.label, draft_index)
        if matches:
            raise RuntimeError(
                f"generation provenance mismatch: artifacts exist without identity sidecar "
                f"for {spec.label} draft {draft_index}"
            )
        return None
    try:
        provenance_text = provenance_path.read_text(encoding="utf-8")
        provenance = json.loads(provenance_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"generation provenance mismatch: invalid identity sidecar {provenance_path}"
        ) from exc
    _require_exact_keys(provenance, _SIDECAR_KEYS, "identity sidecar")
    if provenance_text != _json_artifact_text(provenance):
        raise RuntimeError("generation provenance mismatch: non-canonical identity sidecar")
    expected_identity = {
        "provenance_version": 1,
        "label": spec.label,
        "kind": spec.kind,
        "draft_index": draft_index,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "generation_parameters": generation_parameters,
        "model_revisions": model_revisions,
        "input_sha256": contract_input_sha256,
        "model_manifest_sha256": contract_model_manifest_sha256,
        "generated_script_sha256": contract_generated_script_sha256,
        "run_contract_sha256": run_contract_sha256,
    }
    if any(provenance.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(
            f"generation provenance mismatch: identity sidecar differs for "
            f"{spec.label} draft {draft_index}"
        )
    raw_path = output_dir / str(provenance.get("raw_path", ""))
    parsed_path = output_dir / str(provenance.get("parsed_path", ""))
    identity_paths = {path.resolve() for path in _identity_artifacts(output_dir, spec.label, draft_index)}
    if identity_paths != {raw_path.resolve(), parsed_path.resolve()}:
        raise RuntimeError("generation provenance mismatch: alternate identity artifact branch")
    if not raw_path.is_file() or not parsed_path.is_file():
        raise RuntimeError("generation provenance mismatch: sidecar artifact path missing")
    if (
        sha256_file(raw_path) != provenance.get("raw_file_sha256")
        or sha256_file(parsed_path) != provenance.get("parsed_file_sha256")
    ):
        raise RuntimeError("generation provenance mismatch: sidecar artifact SHA differs")
    if manifest_call is not None:
        manifest_expected = {
            "raw_path": provenance.get("raw_path"),
            "parsed_path": provenance.get("parsed_path"),
            "provenance_path": provenance_path.relative_to(output_dir).as_posix(),
            "raw_file_sha256": provenance.get("raw_file_sha256"),
            "parsed_file_sha256": provenance.get("parsed_file_sha256"),
            "provenance_file_sha256": sha256_file(provenance_path),
            "prompt_sha256": prompt_sha256,
            "seed": seed,
            "raw_response_sha256": provenance.get("raw_response_sha256"),
            "parsed_artifact_sha256": provenance.get("parsed_artifact_sha256"),
        }
        if any(manifest_call.get(key) != value for key, value in manifest_expected.items()):
            raise RuntimeError("generation provenance mismatch: manifest call anchor differs")
    try:
        raw_text = raw_path.read_text(encoding="utf-8")
        raw_payload = json.loads(raw_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid immutable raw artifact: {raw_path}") from exc
    _require_exact_keys(raw_payload, _RAW_SINGLE_DRAFT_KEYS, "raw artifact")
    if raw_text != _json_artifact_text(raw_payload):
        raise RuntimeError(f"generation provenance mismatch: non-canonical raw artifact {raw_path}")
    expected_metadata = {
        "label": spec.label,
        "kind": spec.kind,
        "draft_index": draft_index,
        "seed": seed,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "generation_parameters": generation_parameters,
        "model_revisions": model_revisions,
    }
    if not isinstance(raw_payload, dict) or any(
        raw_payload.get(key) != value for key, value in expected_metadata.items()
    ):
        raise RuntimeError(f"immutable raw provenance mismatch: {raw_path}")
    raw_response = raw_payload.get("raw_response")
    response_sha256 = raw_payload.get("raw_response_sha256")
    parsed_sha256 = raw_payload.get("parsed_artifact_sha256")
    if (
        not isinstance(raw_response, str)
        or not isinstance(response_sha256, str)
        or sha256_text(raw_response) != response_sha256
        or not raw_path.stem.endswith(f"__response-{response_sha256}")
        or not isinstance(parsed_sha256, str)
    ):
        raise RuntimeError(f"immutable raw SHA mismatch: {raw_path}")
    try:
        parsed_text = parsed_path.read_text(encoding="utf-8")
        parsed_record = json.loads(parsed_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid parsed artifact: {parsed_path}") from exc
    if sha256_text(parsed_text) != parsed_sha256:
        raise RuntimeError(
            f"generation provenance mismatch: parsed artifact SHA differs: {parsed_path}"
        )
    _require_exact_keys(parsed_record, _PARSED_SINGLE_DRAFT_KEYS, "parsed artifact")
    expected_record = parse_single_draft_response(
        raw_response,
        label=spec.label,
        kind=spec.kind,
        draft_index=draft_index,
        seed=seed,
        prompt_sha256=prompt_sha256,
        generation_parameters=generation_parameters,
        model_revisions=model_revisions,
        known_labels=known_labels,
    )
    if parsed_record != expected_record or parsed_text != _json_artifact_text(expected_record):
        raise RuntimeError(f"parsed artifact content mismatch: {parsed_path}")
    expected_sidecar = {
        "provenance_version": 1,
        "label": spec.label,
        "kind": spec.kind,
        "draft_index": draft_index,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "raw_response_sha256": response_sha256,
        "parsed_artifact_sha256": parsed_sha256,
        "generation_parameters": dict(generation_parameters),
        "model_revisions": dict(model_revisions),
        "input_sha256": dict(contract_input_sha256),
        "model_manifest_sha256": contract_model_manifest_sha256,
        "generated_script_sha256": contract_generated_script_sha256,
        "run_contract_sha256": run_contract_sha256,
        "raw_path": raw_path.relative_to(output_dir).as_posix(),
        "raw_file_sha256": sha256_file(raw_path),
        "parsed_path": parsed_path.relative_to(output_dir).as_posix(),
        "parsed_file_sha256": sha256_file(parsed_path),
    }
    if provenance != expected_sidecar:
        raise RuntimeError("generation provenance mismatch: reconstructed sidecar differs")
    expected_call = {
        "label": spec.label,
        "kind": spec.kind,
        "draft_index": draft_index,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "raw_response_sha256": response_sha256,
        "parsed_artifact_sha256": parsed_sha256,
        "generation_parameters": dict(generation_parameters),
        "model_revisions": dict(model_revisions),
        "parse_status": expected_record["parse_status"],
        "rejection_reasons": list(expected_record["rejection_reasons"]),
        "raw_path": expected_sidecar["raw_path"],
        "parsed_path": expected_sidecar["parsed_path"],
        "provenance_path": provenance_path.relative_to(output_dir).as_posix(),
        "raw_file_sha256": expected_sidecar["raw_file_sha256"],
        "parsed_file_sha256": expected_sidecar["parsed_file_sha256"],
        "provenance_file_sha256": sha256_file(provenance_path),
    }
    if manifest_call is not None:
        if any(manifest_call.get(key) != value for key, value in expected_call.items()):
            raise RuntimeError(
                "generation provenance mismatch: reconstructed manifest call differs"
            )
        if not isinstance(manifest_call["resumed"], bool) or not isinstance(
            manifest_call["elapsed_seconds"], (int, float)
        ):
            raise RuntimeError("generation provenance mismatch: invalid manifest call metadata")
    return parsed_record, raw_path, parsed_path


def run_single_draft_pack(
    *,
    entries: list[dict[str, Any]],
    client: Any,
    output_dir: Path,
    labels: list[str],
    drafts_per_label: int,
    base_seed: int,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
    input_hashes: dict[str, str],
    model_manifest_sha256: str,
    script_sha256: str,
) -> dict[str, Any]:
    """Generate one JSON object per call and checkpoint a draft-only manifest."""

    if drafts_per_label <= 0:
        raise ValueError("drafts_per_label must be positive")
    output_dir = Path(output_dir)
    allowed_root = SINGLE_DRAFT_OUTPUT_DIR.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output != allowed_root and allowed_root not in resolved_output.parents:
        raise ValueError(f"output path must stay within {SINGLE_DRAFT_OUTPUT_DIR}")
    specs = build_single_draft_specs(entries, labels=labels)
    known_labels = {entry["label"] for entry in entries}
    started_at = _utc_now()
    started_clock = time.perf_counter()
    calls: list[dict[str, Any]] = []
    label_summaries: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _generation_contract(
        specs=specs,
        drafts_per_label=drafts_per_label,
        base_seed=base_seed,
        generation_parameters=generation_parameters,
        model_revisions=model_revisions,
        input_hashes=input_hashes,
        model_manifest_sha256=model_manifest_sha256,
        script_sha256=script_sha256,
    )
    contract_path = output_dir / "generation_run.json"
    if contract_path.exists():
        _load_generation_contract(contract_path, contract)
    else:
        existing = [
            path
            for directory in ("raw", "parsed", "provenance")
            for path in (output_dir / directory).glob("*.json")
        ]
        if existing:
            raise RuntimeError(
                "generation provenance mismatch: artifacts exist without immutable run contract"
            )
        _write_json_immutable(contract_path, contract)
    run_contract_sha256 = sha256_file(contract_path)
    expected_identities = {
        (spec.label, draft_index)
        for spec in specs
        for draft_index in range(drafts_per_label)
    }
    resume_manifest_calls = _load_resume_manifest(
        output_dir, contract, run_contract_sha256, expected_identities
    )

    for spec in specs:
        label_started = time.perf_counter()
        label_calls: list[dict[str, Any]] = []
        for draft_index in range(drafts_per_label):
            seed = derive_draft_seed(base_seed, spec.label, draft_index)
            prompt = (
                spec.prompt
                + f"\n本次独立草稿索引为 {draft_index}；只影响随机采样，不要输出索引。"
            )
            prompt_sha256 = sha256_text(prompt)
            call_started = time.perf_counter()
            resumed_payload = _resume_single_draft(
                output_dir=output_dir,
                spec=spec,
                prompt=prompt,
                prompt_sha256=prompt_sha256,
                draft_index=draft_index,
                seed=seed,
                generation_parameters=generation_parameters,
                model_revisions=model_revisions,
                known_labels=known_labels,
                run_contract_sha256=run_contract_sha256,
                contract_input_sha256=contract["input_sha256"],
                contract_model_manifest_sha256=contract["model_manifest_sha256"],
                contract_generated_script_sha256=contract["generated_script_sha256"],
                manifest_call=resume_manifest_calls.get((spec.label, draft_index)),
            )
            resumed = resumed_payload is not None
            if resumed_payload is not None:
                record, raw_path, parsed_path = resumed_payload
                response_sha256 = record["raw_response_sha256"]
                parsed_sha256 = sha256_text(parsed_path.read_text(encoding="utf-8"))
            else:
                raw_response = client.generate(
                    prompt,
                    seed=seed,
                    generation_parameters=generation_parameters,
                )
                response_sha256 = sha256_text(raw_response)
                record = parse_single_draft_response(
                    raw_response,
                    label=spec.label,
                    kind=spec.kind,
                    draft_index=draft_index,
                    seed=seed,
                    prompt_sha256=prompt_sha256,
                    generation_parameters=generation_parameters,
                    model_revisions=model_revisions,
                    known_labels=known_labels,
                )
                parsed_text = _json_artifact_text(record)
                parsed_sha256 = sha256_text(parsed_text)
                stem = _single_draft_stem(
                    spec.label,
                    prompt_sha256,
                    seed,
                    response_sha256,
                )
                raw_path = output_dir / "raw" / f"{stem}.json"
                parsed_path = output_dir / "parsed" / f"{stem}.json"
                raw_payload = {
                    "label": spec.label,
                    "kind": spec.kind,
                    "draft_index": draft_index,
                    "seed": seed,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "raw_response": raw_response,
                    "raw_response_sha256": response_sha256,
                    "generation_parameters": dict(generation_parameters),
                    "model_revisions": dict(model_revisions),
                    "parsed_artifact_sha256": parsed_sha256,
                }
                _write_json_immutable(raw_path, raw_payload)
                _write_json_immutable(parsed_path, record)
                provenance_path = (
                    output_dir
                    / "provenance"
                    / f"{_draft_identity_stem(spec.label, draft_index)}.json"
                )
                _write_json_immutable(
                    provenance_path,
                    {
                        "provenance_version": 1,
                        "label": spec.label,
                        "kind": spec.kind,
                        "draft_index": draft_index,
                        "seed": seed,
                        "prompt_sha256": prompt_sha256,
                        "raw_response_sha256": response_sha256,
                        "parsed_artifact_sha256": parsed_sha256,
                        "generation_parameters": dict(generation_parameters),
                        "model_revisions": dict(model_revisions),
                        "input_sha256": dict(input_hashes),
                        "model_manifest_sha256": model_manifest_sha256,
                        "generated_script_sha256": script_sha256,
                        "run_contract_sha256": run_contract_sha256,
                        "raw_path": raw_path.relative_to(output_dir).as_posix(),
                        "raw_file_sha256": sha256_file(raw_path),
                        "parsed_path": parsed_path.relative_to(output_dir).as_posix(),
                        "parsed_file_sha256": sha256_file(parsed_path),
                    },
                )
            provenance_path = (
                output_dir
                / "provenance"
                / f"{_draft_identity_stem(spec.label, draft_index)}.json"
            )
            call = {
                "label": spec.label,
                "kind": spec.kind,
                "draft_index": draft_index,
                "seed": seed,
                "prompt_sha256": prompt_sha256,
                "raw_response_sha256": response_sha256,
                "parsed_artifact_sha256": parsed_sha256,
                "generation_parameters": dict(generation_parameters),
                "model_revisions": dict(model_revisions),
                "parse_status": record["parse_status"],
                "rejection_reasons": list(record["rejection_reasons"]),
                "resumed": resumed,
                "raw_path": raw_path.relative_to(output_dir).as_posix(),
                "parsed_path": parsed_path.relative_to(output_dir).as_posix(),
                "provenance_path": provenance_path.relative_to(output_dir).as_posix(),
                "raw_file_sha256": sha256_file(raw_path),
                "parsed_file_sha256": sha256_file(parsed_path),
                "provenance_file_sha256": sha256_file(provenance_path),
                "elapsed_seconds": round(time.perf_counter() - call_started, 3),
            }
            calls.append(call)
            label_calls.append(call)

        label_summaries.append(
            {
                "label": spec.label,
                "kind": spec.kind,
                "requested": drafts_per_label,
                "parsed": sum(item["parse_status"] == "parsed" for item in label_calls),
                "rejected": sum(item["parse_status"] == "rejected" for item in label_calls),
                "elapsed_seconds": round(time.perf_counter() - label_started, 3),
            }
        )

    manifest = {
        "manifest_version": 2,
        "mode": "single-draft",
        "status": "complete",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started_clock, 3),
        "expected_calls": len(specs) * drafts_per_label,
        "completed_calls": len(calls),
        "parsed_count": sum(item["parse_status"] == "parsed" for item in calls),
        "rejected_count": sum(item["parse_status"] == "rejected" for item in calls),
        "drafts_per_label": drafts_per_label,
        "base_seed": base_seed,
        "labels": label_summaries,
        "calls": calls,
        "input_sha256": dict(input_hashes),
        "script_sha256": script_sha256,
        "generated_script_sha256": script_sha256,
        "delivered_script_sha256": script_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "model_revisions": dict(model_revisions),
        "generation_parameters": dict(generation_parameters),
        "generation_run_path": contract_path.relative_to(output_dir).as_posix(),
        "generation_run_sha256": run_contract_sha256,
    }
    manifest_path = output_dir / "draft_manifest.json"
    if not (calls and all(item["resumed"] for item in calls) and manifest_path.is_file()):
        _write_json_atomic(manifest_path, manifest)
        _write_completion_record(output_dir, run_contract_sha256, expected_identities)
    return manifest


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path_has_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(
        getattr(metadata, "st_file_attributes", 0) & 0x400
    )


def _validate_migration_path(
    output_dir: Path,
    relative: Any,
    expected_parts: tuple[str, ...],
) -> Path:
    expected_text = PurePosixPath(*expected_parts).as_posix()
    if not isinstance(relative, str) or relative != expected_text:
        raise RuntimeError(f"migration path boundary mismatch: {relative!r}")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"migration path escape rejected: {relative!r}")
    output_resolved = output_dir.resolve()
    candidate = output_dir.joinpath(*expected_parts)
    expected_parent = output_dir.joinpath(*expected_parts[:-1])
    if (
        candidate.resolve(strict=False) != expected_parent.resolve(strict=False)
        / expected_parts[-1]
        or output_resolved not in candidate.resolve(strict=False).parents
    ):
        raise RuntimeError(f"migration path resolution escape rejected: {relative!r}")
    current = output_dir
    for part in expected_parts:
        current = current / part
        if _path_has_reparse_point(current):
            raise RuntimeError(f"migration path reparse point rejected: {relative!r}")
    return candidate


def _validate_migration_pairs(
    output_dir: Path,
    pairs: Any,
    *,
    formal_raw: set[str],
    formal_parsed: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(pairs, list):
        raise RuntimeError("migration path plan pairs must be a list")
    validated: list[dict[str, Any]] = []
    identities: set[str] = set()
    for pair in pairs:
        _require_exact_keys(pair, _MIGRATION_PAIR_KEYS, "migration pair")
        raw_source = PurePosixPath(pair["original_raw_path"])
        parsed_source = PurePosixPath(pair["original_parsed_path"])
        if (
            len(raw_source.parts) != 2
            or len(parsed_source.parts) != 2
            or raw_source.parts[0] != "raw"
            or parsed_source.parts[0] != "parsed"
            or raw_source.name != parsed_source.name
            or not raw_source.name.endswith(".json")
        ):
            raise RuntimeError("migration path pair identity mismatch")
        name = raw_source.name
        if name in identities:
            raise RuntimeError("migration path duplicate pair identity")
        identities.add(name)
        if pair["original_raw_path"] in formal_raw or pair["original_parsed_path"] in formal_parsed:
            raise RuntimeError("migration path references a formal artifact identity")
        expected_paths = {
            "original_raw_path": ("raw", name),
            "migrated_raw_path": ("superseded", "raw", name),
            "original_parsed_path": ("parsed", name),
            "migrated_parsed_path": ("superseded", "parsed", name),
        }
        for key, parts in expected_paths.items():
            _validate_migration_path(output_dir, pair[key], parts)
        if not _is_sha256(pair["raw_sha256"]) or not _is_sha256(
            pair["parsed_sha256"]
        ):
            raise RuntimeError("migration path pair SHA schema mismatch")
        validated.append(pair)
    return validated


def _validate_migration_state(
    state_path: Path,
    state: Any,
    *,
    plan_sha256: str,
    snapshot_path: str,
    pair_names: set[str],
    manifest_sha256: str,
) -> list[str]:
    _require_exact_keys(state, _MIGRATION_STATE_KEYS, "migration state")
    try:
        state_text = state_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("generation provenance mismatch: invalid migration state") from exc
    if state_text != _json_artifact_text(state):
        raise RuntimeError("generation provenance mismatch: non-canonical migration state")
    completed = state["completed_pairs"]
    if (
        state["state_version"] != 1
        or state["status"] not in {"moving", "complete"}
        or state["plan_sha256"] != plan_sha256
        or state["previous_manifest_snapshot_path"] != snapshot_path
        or not isinstance(completed, list)
        or not all(isinstance(name, str) for name in completed)
        or len(completed) != len(set(completed))
        or not set(completed) <= pair_names
    ):
        raise RuntimeError("generation provenance mismatch: migration state differs")
    if state["status"] == "moving":
        if state["final_manifest_sha256"] is not None:
            raise RuntimeError("generation provenance mismatch: moving state has final SHA")
    elif set(completed) != pair_names or state["final_manifest_sha256"] != manifest_sha256:
        raise RuntimeError("generation provenance mismatch: complete migration state differs")
    return completed


def _validate_manifest_migration_audit(
    output_dir: Path,
    manifest: dict[str, Any],
    calls: list[dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    superseded = manifest["superseded_audit"]
    migration = manifest["migration"]
    parser_audit = manifest["parser_migration_audit"]
    _require_exact_keys(superseded, _SUPERSEDED_AUDIT_KEYS, "superseded audit")
    _require_exact_keys(migration, _MIGRATION_RECORD_KEYS, "migration record")
    _require_exact_keys(
        parser_audit, _PARSER_MIGRATION_AUDIT_KEYS, "parser migration audit"
    )
    plan_path = _validate_migration_path(
        output_dir, "migration/plan.json", ("migration", "plan.json")
    )
    state_path = _validate_migration_path(
        output_dir, "migration/state.json", ("migration", "state.json")
    )
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
        plan = json.loads(plan_text)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("generation provenance mismatch: invalid migration audit") from exc
    _require_exact_keys(plan, _MIGRATION_PLAN_KEYS, "migration plan")
    if (
        plan_text != _json_artifact_text(plan)
        or plan["plan_version"] != 1
        or not _is_sha256(plan["previous_manifest_sha256"])
        or not _is_sha256(plan["delivered_script_sha256"])
    ):
        raise RuntimeError("generation provenance mismatch: migration plan differs")
    formal_raw = {str(call["raw_path"]) for call in calls}
    formal_parsed = {str(call["parsed_path"]) for call in calls}
    pairs = _validate_migration_pairs(
        output_dir,
        plan["pairs"],
        formal_raw=formal_raw,
        formal_parsed=formal_parsed,
    )
    if superseded["pairs"] != pairs:
        raise RuntimeError("generation provenance mismatch: superseded audit pairs differ")
    expected_reason = "superseded prompt architecture; excluded from formal calls"
    if (
        superseded["directory"] != "superseded"
        or superseded["raw_count"] != len(pairs)
        or superseded["parsed_count"] != len(pairs)
        or superseded["reason"] != expected_reason
    ):
        raise RuntimeError("generation provenance mismatch: superseded audit differs")
    for pair in pairs:
        raw_target = output_dir / pair["migrated_raw_path"]
        parsed_target = output_dir / pair["migrated_parsed_path"]
        if (
            not raw_target.is_file()
            or not parsed_target.is_file()
            or sha256_file(raw_target) != pair["raw_sha256"]
            or sha256_file(parsed_target) != pair["parsed_sha256"]
            or (output_dir / pair["original_raw_path"]).exists()
            or (output_dir / pair["original_parsed_path"]).exists()
        ):
            raise RuntimeError("generation provenance mismatch: superseded files differ")
    if (
        len(list((output_dir / "superseded" / "raw").glob("*.json"))) != len(pairs)
        or len(list((output_dir / "superseded" / "parsed").glob("*.json")))
        != len(pairs)
    ):
        raise RuntimeError("generation provenance mismatch: superseded inventory differs")
    snapshot_relative = plan["previous_manifest_snapshot_path"]
    snapshot_path = _validate_migration_path(
        output_dir,
        snapshot_relative,
        (
            "manifest_history",
            f"manifest-{plan['previous_manifest_sha256']}.json",
        ),
    )
    if (
        not snapshot_path.is_file()
        or sha256_file(snapshot_path) != plan["previous_manifest_sha256"]
        or migration["previous_manifest_sha256"] != plan["previous_manifest_sha256"]
        or migration["previous_manifest_snapshot_path"] != snapshot_relative
        or migration["formal_raw_count"] != len(calls)
        or migration["formal_parsed_count"] != len(calls)
        or not isinstance(migration["migrated_at"], str)
    ):
        raise RuntimeError("generation provenance mismatch: migration record differs")
    pair_names = {PurePosixPath(pair["original_raw_path"]).name for pair in pairs}
    _validate_migration_state(
        state_path,
        state,
        plan_sha256=sha256_file(plan_path),
        snapshot_path=snapshot_relative,
        pair_names=pair_names,
        manifest_sha256=sha256_file(output_dir / "draft_manifest.json"),
    )
    if sha256_file(
        PROJECT_ROOT / "tools" / "generate_spoken_annotation_candidates.py"
    ) != manifest["delivered_script_sha256"]:
        raise RuntimeError("generation provenance mismatch: delivery contract differs")
    known_labels = {str(item["label"]) for item in contract["labels"]}
    delivered_parsed = 0
    changed = 0
    for call in calls:
        raw = json.loads((output_dir / call["raw_path"]).read_text(encoding="utf-8"))
        stored = json.loads(
            (output_dir / call["parsed_path"]).read_text(encoding="utf-8")
        )
        delivered = parse_single_draft_response(
            raw["raw_response"],
            label=call["label"],
            kind=call["kind"],
            draft_index=call["draft_index"],
            seed=call["seed"],
            prompt_sha256=call["prompt_sha256"],
            generation_parameters=call["generation_parameters"],
            model_revisions=call["model_revisions"],
            known_labels=known_labels,
        )
        delivered_parsed += delivered["parse_status"] == "parsed"
        changed += delivered != stored
    expected_parser_audit = {
        "formal_artifact_count": len(calls),
        "changed_record_count": changed,
        "delivered_parser_parsed": delivered_parsed,
        "delivered_parser_rejected": len(calls) - delivered_parsed,
        "formal_artifacts_rewritten": False,
    }
    if parser_audit != expected_parser_audit:
        raise RuntimeError("generation provenance mismatch: parser migration audit differs")


def migrate_single_draft_artifacts(
    output_dir: Path,
    *,
    delivered_script_sha256: str,
    progress_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Add immutable provenance sidecars and isolate unreferenced legacy artifacts."""

    output_dir = Path(output_dir)
    if not _is_sha256(delivered_script_sha256):
        raise ValueError("delivered_script_sha256 must be a lowercase SHA-256")
    allowed_root = SINGLE_DRAFT_OUTPUT_DIR.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output != allowed_root and allowed_root not in resolved_output.parents:
        raise ValueError(f"output path must stay within {SINGLE_DRAFT_OUTPUT_DIR}")
    manifest_path = output_dir / "draft_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    calls = manifest.get("calls")
    if not isinstance(calls, list) or not calls:
        raise RuntimeError("migration requires a complete draft manifest")
    formal_raw = {str(call["raw_path"]) for call in calls}
    formal_parsed = {str(call["parsed_path"]) for call in calls}
    plan_path = output_dir / "migration" / "plan.json"
    state_path = output_dir / "migration" / "state.json"
    existing_audit = manifest.get("superseded_audit")
    reuse_existing_audit = isinstance(existing_audit, dict) and isinstance(
        existing_audit.get("pairs"), list
    )
    if plan_path.exists():
        _validate_migration_path(
            output_dir, "migration/plan.json", ("migration", "plan.json")
        )
        plan_text = plan_path.read_text(encoding="utf-8")
        plan = json.loads(plan_text)
        _require_exact_keys(plan, _MIGRATION_PLAN_KEYS, "migration plan")
        if plan_text != _json_artifact_text(plan):
            raise RuntimeError("generation provenance mismatch: non-canonical migration plan")
        if (
            plan["plan_version"] != 1
            or not _is_sha256(plan["previous_manifest_sha256"])
            or not _is_sha256(plan["delivered_script_sha256"])
        ):
            raise RuntimeError("migration plan schema mismatch")
    else:
        previous_manifest_sha256, previous_manifest_bytes = (
            _recover_previous_manifest_bytes(manifest, manifest_bytes)
        )
        snapshot_path = (
            output_dir
            / "manifest_history"
            / f"manifest-{previous_manifest_sha256}.json"
        )
        _write_bytes_immutable(snapshot_path, previous_manifest_bytes)
        all_raw = {
            path.relative_to(output_dir).as_posix(): path
            for path in (output_dir / "raw").glob("*.json")
        }
        all_parsed = {
            path.relative_to(output_dir).as_posix(): path
            for path in (output_dir / "parsed").glob("*.json")
        }
        extra_raw = {key: value for key, value in all_raw.items() if key not in formal_raw}
        extra_parsed = {
            key: value for key, value in all_parsed.items() if key not in formal_parsed
        }
        raw_by_name = {path.name: path for path in extra_raw.values()}
        parsed_by_name = {path.name: path for path in extra_parsed.values()}
        if set(raw_by_name) != set(parsed_by_name):
            raise RuntimeError("superseded raw/parsed artifact names do not pair")
        if reuse_existing_audit:
            pairs = list(existing_audit["pairs"])
        else:
            pairs = [
                {
                    "original_raw_path": raw_by_name[name]
                    .relative_to(output_dir)
                    .as_posix(),
                    "migrated_raw_path": (
                        Path("superseded") / "raw" / name
                    ).as_posix(),
                    "raw_sha256": sha256_file(raw_by_name[name]),
                    "original_parsed_path": parsed_by_name[name]
                    .relative_to(output_dir)
                    .as_posix(),
                    "migrated_parsed_path": (
                        Path("superseded") / "parsed" / name
                    ).as_posix(),
                    "parsed_sha256": sha256_file(parsed_by_name[name]),
                }
                for name in sorted(raw_by_name)
            ]
        plan = {
            "plan_version": 1,
            "previous_manifest_sha256": previous_manifest_sha256,
            "previous_manifest_snapshot_path": snapshot_path.relative_to(
                output_dir
            ).as_posix(),
            "delivered_script_sha256": delivered_script_sha256,
            "pairs": pairs,
        }
        _write_json_immutable(plan_path, plan)

    previous_manifest_sha256 = str(plan["previous_manifest_sha256"])
    snapshot_relative = f"manifest_history/manifest-{previous_manifest_sha256}.json"
    snapshot_path = _validate_migration_path(
        output_dir,
        plan["previous_manifest_snapshot_path"],
        ("manifest_history", f"manifest-{previous_manifest_sha256}.json"),
    )
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != previous_manifest_sha256:
        raise RuntimeError("previous manifest snapshot SHA mismatch")
    pairs = _validate_migration_pairs(
        output_dir,
        plan["pairs"],
        formal_raw=formal_raw,
        formal_parsed=formal_parsed,
    )
    if reuse_existing_audit and pairs != existing_audit["pairs"]:
        raise RuntimeError("generation provenance mismatch: migration plan/audit differs")
    pair_names = {PurePosixPath(pair["original_raw_path"]).name for pair in pairs}
    plan_sha256 = sha256_file(plan_path)
    if state_path.exists():
        _validate_migration_path(
            output_dir, "migration/state.json", ("migration", "state.json")
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed_pairs = _validate_migration_state(
            state_path,
            state,
            plan_sha256=plan_sha256,
            snapshot_path=snapshot_relative,
            pair_names=pair_names,
            manifest_sha256=sha256_file(manifest_path),
        )
    else:
        completed_pairs = []
        state = {
            "state_version": 1,
            "status": "moving",
            "plan_sha256": plan_sha256,
            "previous_manifest_snapshot_path": snapshot_relative,
            "completed_pairs": completed_pairs,
            "final_manifest_sha256": None,
        }
        _write_json_atomic(state_path, state)
    for pair in pairs:
        pair_name = PurePosixPath(pair["original_raw_path"]).name
        for source_key, target_key, sha_key, source_parts, target_parts in (
            (
                "original_raw_path",
                "migrated_raw_path",
                "raw_sha256",
                ("raw", pair_name),
                ("superseded", "raw", pair_name),
            ),
            (
                "original_parsed_path",
                "migrated_parsed_path",
                "parsed_sha256",
                ("parsed", pair_name),
                ("superseded", "parsed", pair_name),
            ),
        ):
            source = _validate_migration_path(
                output_dir, pair[source_key], source_parts
            )
            target = _validate_migration_path(
                output_dir, pair[target_key], target_parts
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = _validate_migration_path(
                output_dir, pair[source_key], source_parts
            )
            target = _validate_migration_path(
                output_dir, pair[target_key], target_parts
            )
            if source.exists() and not target.exists():
                if sha256_file(source) != pair[sha_key]:
                    raise RuntimeError("superseded migration source SHA mismatch")
                source.rename(target)
            elif source.exists() and target.exists():
                raise RuntimeError("superseded migration source and target both exist")
            elif not target.exists():
                raise RuntimeError("superseded migration source and target both missing")
            target = _validate_migration_path(
                output_dir, pair[target_key], target_parts
            )
            if sha256_file(target) != pair[sha_key]:
                raise RuntimeError("superseded migration target SHA mismatch")
        if pair_name not in completed_pairs:
            completed_pairs.append(pair_name)
            state["completed_pairs"] = list(completed_pairs)
            _write_json_atomic(state_path, state)
            if progress_hook is not None:
                progress_hook(pair_name)

    generated_script_sha256 = str(
        manifest.get("generated_script_sha256", manifest["script_sha256"])
    )
    label_contracts: list[dict[str, Any]] = []
    for summary in manifest["labels"]:
        label = summary["label"]
        call = next(item for item in calls if item["label"] == label)
        raw = json.loads((output_dir / call["raw_path"]).read_text(encoding="utf-8"))
        suffix = (
            f"\n本次独立草稿索引为 {call['draft_index']}；"
            "只影响随机采样，不要输出索引。"
        )
        prompt = str(raw["prompt"])
        if not prompt.endswith(suffix):
            raise RuntimeError("cannot recover immutable base prompt from formal raw")
        label_contracts.append(
            {
                "label": label,
                "kind": summary["kind"],
                "base_prompt_sha256": sha256_text(prompt[: -len(suffix)]),
            }
        )
    contract = {
        "contract_version": 1,
        "mode": "single-draft",
        "labels": label_contracts,
        "drafts_per_label": manifest["drafts_per_label"],
        "base_seed": manifest["base_seed"],
        "generation_parameters": manifest["generation_parameters"],
        "model_revisions": manifest["model_revisions"],
        "input_sha256": manifest["input_sha256"],
        "model_manifest_sha256": manifest["model_manifest_sha256"],
        "generated_script_sha256": generated_script_sha256,
    }
    contract_path = output_dir / "generation_run.json"
    if contract_path.exists():
        _load_generation_contract(contract_path, contract)
    else:
        _write_json_immutable(contract_path, contract)
    run_contract_sha256 = sha256_file(contract_path)

    migrated_calls: list[dict[str, Any]] = []
    known_labels = {str(call["label"]) for call in calls}
    delivered_parser_parsed = 0
    delivered_parser_rejected = 0
    changed_record_count = 0
    for original_call in calls:
        call = dict(original_call)
        raw_path = output_dir / call["raw_path"]
        parsed_path = output_dir / call["parsed_path"]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        stored_record = json.loads(parsed_path.read_text(encoding="utf-8"))
        delivered_record = parse_single_draft_response(
            raw["raw_response"],
            label=call["label"],
            kind=call["kind"],
            draft_index=call["draft_index"],
            seed=call["seed"],
            prompt_sha256=call["prompt_sha256"],
            generation_parameters=call["generation_parameters"],
            model_revisions=call["model_revisions"],
            known_labels=known_labels,
        )
        if delivered_record["parse_status"] == "parsed":
            delivered_parser_parsed += 1
        else:
            delivered_parser_rejected += 1
        if delivered_record != stored_record:
            changed_record_count += 1
        sidecar_path = (
            output_dir
            / "provenance"
            / f"{_draft_identity_stem(call['label'], call['draft_index'])}.json"
        )
        sidecar = {
            "provenance_version": 1,
            "label": call["label"],
            "kind": call["kind"],
            "draft_index": call["draft_index"],
            "seed": call["seed"],
            "prompt_sha256": call["prompt_sha256"],
            "raw_response_sha256": call["raw_response_sha256"],
            "parsed_artifact_sha256": call["parsed_artifact_sha256"],
            "generation_parameters": manifest["generation_parameters"],
            "model_revisions": manifest["model_revisions"],
            "input_sha256": manifest["input_sha256"],
            "model_manifest_sha256": manifest["model_manifest_sha256"],
            "generated_script_sha256": generated_script_sha256,
            "run_contract_sha256": run_contract_sha256,
            "raw_path": call["raw_path"],
            "raw_file_sha256": sha256_file(raw_path),
            "parsed_path": call["parsed_path"],
            "parsed_file_sha256": sha256_file(parsed_path),
        }
        if raw.get("parsed_artifact_sha256") != call["parsed_artifact_sha256"]:
            raise RuntimeError("formal raw/manifest parsed SHA mismatch during migration")
        if sidecar_path.exists():
            existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if existing != sidecar:
                raise RuntimeError("existing provenance sidecar mismatch during migration")
        else:
            _write_json_immutable(sidecar_path, sidecar)
        call.update(
            {
                "raw_file_sha256": sidecar["raw_file_sha256"],
                "parsed_file_sha256": sidecar["parsed_file_sha256"],
                "provenance_path": sidecar_path.relative_to(output_dir).as_posix(),
                "provenance_file_sha256": sha256_file(sidecar_path),
            }
        )
        migrated_calls.append(call)

    if reuse_existing_audit:
        superseded_audit = dict(existing_audit)
        migration_record = dict(manifest.get("migration", {}))
        migration_record["previous_manifest_snapshot_path"] = plan[
            "previous_manifest_snapshot_path"
        ]
    else:
        superseded_audit = {
            "directory": "superseded",
            "raw_count": len(pairs),
            "parsed_count": len(pairs),
            "pairs": pairs,
            "reason": "superseded prompt architecture; excluded from formal calls",
        }
        migration_record = {
            "migrated_at": _utc_now(),
            "previous_manifest_sha256": previous_manifest_sha256,
            "previous_manifest_snapshot_path": plan[
                "previous_manifest_snapshot_path"
            ],
            "formal_raw_count": len(formal_raw),
            "formal_parsed_count": len(formal_parsed),
        }
    migrated = dict(manifest)
    migrated.update(
        {
            "manifest_version": 2,
            "calls": migrated_calls,
            "script_sha256": generated_script_sha256,
            "generated_script_sha256": generated_script_sha256,
            "delivered_script_sha256": delivered_script_sha256,
            "generation_run_path": contract_path.relative_to(output_dir).as_posix(),
            "generation_run_sha256": run_contract_sha256,
            "parser_migration_audit": {
                "formal_artifact_count": len(migrated_calls),
                "changed_record_count": changed_record_count,
                "delivered_parser_parsed": delivered_parser_parsed,
                "delivered_parser_rejected": delivered_parser_rejected,
                "formal_artifacts_rewritten": False,
            },
            "superseded_audit": superseded_audit,
            "migration": migration_record,
        }
    )
    _write_delivery_record(
        output_dir,
        delivered_script_sha256,
        generated_script_sha256,
        run_contract_sha256,
    )
    _write_json_atomic(manifest_path, migrated)
    expected_identities = {
        (str(call["label"]), int(call["draft_index"])) for call in migrated_calls
    }
    _write_completion_record(output_dir, run_contract_sha256, expected_identities)
    state.update(
        {
            "status": "complete",
            "completed_pairs": [Path(pair["original_raw_path"]).name for pair in pairs],
            "final_manifest_sha256": sha256_file(manifest_path),
        }
    )
    _write_json_atomic(state_path, state)
    return migrated


def _is_matching_raw(
    payload: dict[str, Any],
    *,
    spec: BatchSpec,
    count: int,
    seed: int,
    prompt_hash: str,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
) -> bool:
    expected = {
        "batch_label": spec.label,
        "kind": spec.kind,
        "safety_class": spec.safety_class,
        "candidates_requested": count,
        "seed": seed,
        "prompt_sha256": prompt_hash,
        "generation_parameters": generation_parameters,
        "model_revisions": model_revisions,
    }
    return all(payload.get(key) == value for key, value in expected.items()) and isinstance(
        payload.get("raw_response"), str
    )


def run_batch(
    spec: BatchSpec,
    *,
    client: Any,
    output_dir: Path,
    count: int,
    seed: int,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
    known_labels: set[str],
) -> dict[str, Any]:
    """Generate or resume one independently prompted label batch."""

    output_dir = Path(output_dir)
    allowed_root = DEFAULT_OUTPUT_DIR.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output != allowed_root and allowed_root not in resolved_output.parents:
        raise ValueError(f"output path must stay within {DEFAULT_OUTPUT_DIR}")
    raw_path = output_dir / "raw" / f"{_batch_stem(spec.label)}.json"
    candidates_path = output_dir / "candidates" / f"{_batch_stem(spec.label)}.json"
    prompt_hash = sha256_text(spec.prompt)
    resumed = False
    raw_payload: dict[str, Any] | None = None
    if raw_path.is_file():
        try:
            loaded = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and _is_matching_raw(
            loaded,
            spec=spec,
            count=count,
            seed=seed,
            prompt_hash=prompt_hash,
            generation_parameters=generation_parameters,
            model_revisions=model_revisions,
        ):
            raw_payload = loaded
            resumed = True

    if raw_payload is None:
        raw_response = client.generate(
            spec.prompt,
            seed=seed,
            generation_parameters=generation_parameters,
        )
        raw_payload = {
            "batch_label": spec.label,
            "kind": spec.kind,
            "safety_class": spec.safety_class,
            "candidates_requested": count,
            "seed": seed,
            "prompt_sha256": prompt_hash,
            "raw_response_sha256": sha256_text(raw_response),
            "generation_parameters": dict(generation_parameters),
            "model_revisions": dict(model_revisions),
            "raw_response": raw_response,
        }
        _write_json_atomic(raw_path, raw_payload)

    records: list[dict[str, Any]] | None = None
    if resumed and candidates_path.is_file():
        try:
            candidate_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            candidate_payload = None
        if (
            isinstance(candidate_payload, dict)
            and candidate_payload.get("prompt_sha256") == prompt_hash
            and candidate_payload.get("raw_response_sha256")
            == raw_payload["raw_response_sha256"]
            and isinstance(candidate_payload.get("candidates"), list)
        ):
            records = candidate_payload["candidates"]

    if records is None:
        records = parse_batch_response(
            raw_payload["raw_response"],
            batch_label=spec.label,
            safety_class=spec.safety_class,
            count=count,
            seed=seed,
            prompt_sha256=prompt_hash,
            generation_parameters=generation_parameters,
            model_revisions=model_revisions,
            known_labels=known_labels,
        )
        _write_json_atomic(
            candidates_path,
            {
                "batch_label": spec.label,
                "prompt_sha256": prompt_hash,
                "raw_response_sha256": raw_payload["raw_response_sha256"],
                "candidates": records,
            },
        )

    parsed_count = sum(record["parse_status"] == "parsed" for record in records)
    return {
        "batch_label": spec.label,
        "kind": spec.kind,
        "safety_class": spec.safety_class,
        "raw_count": count,
        "parsed_count": parsed_count,
        "rejected_count": count - parsed_count,
        "resumed": resumed,
        "raw_path": raw_path.relative_to(output_dir).as_posix(),
        "candidates_path": candidates_path.relative_to(output_dir).as_posix(),
        "records": records,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deduplicate_across_batches(batch_results: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for result in batch_results:
        for record in result["records"]:
            if record["parse_status"] != "parsed":
                continue
            normalized = normalize_text(record["text"])
            if normalized in seen:
                record["parse_status"] = "rejected"
                record["rejection_reasons"].append("normalized_duplicate_across_batches")
            elif normalized:
                seen[normalized] = record["candidate_id"]


def run_pilot(
    *,
    entries: list[dict[str, Any]],
    client: Any,
    output_dir: Path,
    labels: list[str] | None,
    count: int,
    seed: int,
    generation_parameters: dict[str, Any],
    model_revisions: dict[str, str],
    input_hashes: dict[str, str],
    model_manifest_sha256: str,
    script_sha256: str,
) -> dict[str, Any]:
    """Run selected independent batches and write auditable pilot artifacts."""

    specs = build_batch_specs(entries, labels=labels, count=count)
    known_labels = {entry["label"] for entry in entries}
    started_at = _utc_now()
    started_clock = time.perf_counter()
    batch_results: list[dict[str, Any]] = []
    output_dir = Path(output_dir)
    for spec in specs:
        batch_start = time.perf_counter()
        result = run_batch(
            spec,
            client=client,
            output_dir=output_dir,
            count=count,
            seed=seed,
            generation_parameters=generation_parameters,
            model_revisions=model_revisions,
            known_labels=known_labels,
        )
        result["elapsed_seconds"] = round(time.perf_counter() - batch_start, 3)
        batch_results.append(result)
        checkpoint = {
            "manifest_version": 1,
            "status": "in_progress",
            "started_at": started_at,
            "updated_at": _utc_now(),
            "batch_count": len(batch_results),
            "expected_batch_count": len(specs),
            "raw_count": count * len(batch_results),
            "parsed_count": sum(item["parsed_count"] for item in batch_results),
            "rejected_count": sum(item["rejected_count"] for item in batch_results),
            "requirement_corrections": REQUIREMENT_CORRECTIONS,
            "model_revisions": model_revisions,
            "model_manifest_sha256": model_manifest_sha256,
            "input_sha256": input_hashes,
            "script_sha256": script_sha256,
            "generation_parameters": generation_parameters,
            "seed": seed,
            "batches": [
                {key: value for key, value in item.items() if key != "records"}
                for item in batch_results
            ],
        }
        _write_json_atomic(output_dir / "pilot_manifest.json", checkpoint)

    # Work on detached record dictionaries so per-batch mechanical parse artifacts
    # remain independently reproducible when a different subset is resumed.
    detached_results = json.loads(json.dumps(batch_results, ensure_ascii=False))
    _deduplicate_across_batches(detached_results)
    all_records = [record for result in detached_results for record in result["records"]]
    parsed = [record for record in all_records if record["parse_status"] == "parsed"]
    rejected = [record for record in all_records if record["parse_status"] == "rejected"]
    _write_json_atomic(output_dir / "parsed_candidates.json", parsed)
    _write_json_atomic(output_dir / "rejected_candidates.json", rejected)

    for original, detached in zip(batch_results, detached_results):
        original["parsed_count"] = sum(
            record["parse_status"] == "parsed" for record in detached["records"]
        )
        original["rejected_count"] = count - original["parsed_count"]
    manifest = {
        "manifest_version": 1,
        "status": "complete",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started_clock, 3),
        "batch_count": len(batch_results),
        "expected_batch_count": len(specs),
        "raw_count": count * len(batch_results),
        "parsed_count": len(parsed),
        "rejected_count": len(rejected),
        "requirement_corrections": REQUIREMENT_CORRECTIONS,
        "model_revisions": model_revisions,
        "model_manifest_sha256": model_manifest_sha256,
        "input_sha256": input_hashes,
        "script_sha256": script_sha256,
        "generation_parameters": generation_parameters,
        "seed": seed,
        "batches": [
            {key: value for key, value in item.items() if key != "records"}
            for item in batch_results
        ],
    }
    _write_json_atomic(output_dir / "pilot_manifest.json", manifest)
    return manifest


def write_single_draft_report(
    manifest: dict[str, Any],
    *,
    path: Path = SINGLE_DRAFT_REPORT_PATH,
    test_summary: str = "专项单元测试在模型调用前独立执行。",
) -> None:
    """Write the Task 2D draft-generation report; never certify samples."""

    concerns: list[str] = []
    if manifest["completed_calls"] != manifest["expected_calls"]:
        concerns.append(
            f"仅完成 {manifest['completed_calls']} / {manifest['expected_calls']} 次模型调用。"
        )
    if manifest["rejected_count"]:
        concerns.append(
            f"{manifest['rejected_count']} 条严格解析失败，已逐条隔离为 rejected；"
            "未自动修复、重生成或签发。"
        )
    superseded = manifest.get("superseded_audit")
    if isinstance(superseded, dict) and superseded.get("raw_count"):
        concerns.append(
            f"保留 {superseded['raw_count']} 条旧 prompt raw 与 "
            f"{superseded.get('parsed_count', 0)} 条旧解析证据作为 superseded audit；"
            "它们未混入本报告正式 parsed/rejected 计数。"
        )
    delivered_script_sha256 = manifest.get("delivered_script_sha256")
    if (
        isinstance(delivered_script_sha256, str)
        and delivered_script_sha256 != manifest["script_sha256"]
    ):
        concerns.append(
            "生成后仅报告/测试代码继续变更；manifest 中 generator script SHA 仍精确指向"
            "执行 300 次调用时的脚本，另记录最终交付脚本 SHA。"
        )
    if not concerns:
        concerns.append("无。")
    status = "DONE" if concerns == ["无。"] else "DONE_WITH_CONCERNS"
    rows = [
        "| label | kind | requested | parsed | rejected | seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in manifest["labels"]:
        rows.append(
            f"| `{item['label']}` | {item['kind']} | {item['requested']} | "
            f"{item['parsed']} | {item['rejected']} | {item['elapsed_seconds']:.3f} |"
        )
    reason_counts: dict[str, int] = {}
    for call in manifest.get("calls", []):
        for reason in call.get("rejection_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        reason_section = "## 正式拒绝原因\n\n" + "\n".join(
            f"- `{reason}`：{count}"
            for reason, count in sorted(
                reason_counts.items(), key=lambda item: (-item[1], item[0])
            )
        )
    else:
        reason_section = "## 正式拒绝原因\n\n- 无。"
    if isinstance(superseded, dict) and superseded.get("raw_count"):
        superseded_section = f"""## Superseded audit

- 旧 raw：{superseded['raw_count']}
- 旧 parsed/rejected artifacts：{superseded.get('parsed_count', 0)}
- 原因：{superseded.get('reason', 'superseded prompt architecture')}
- 边界：仅保留不可变审计证据，不计入正式 300 次调用的 parsed/rejected。
"""
    else:
        superseded_section = ""
    migration = manifest.get("migration")
    parser_audit = manifest.get("parser_migration_audit")
    if isinstance(migration, dict) and isinstance(parser_audit, dict):
        migration_section = f"""## Migration audit

- previous manifest SHA-256：`{migration.get('previous_manifest_sha256', '')}`
- formal raw / parsed：{migration.get('formal_raw_count', 0)} / {migration.get('formal_parsed_count', 0)}
- delivered parser parsed / rejected：{parser_audit.get('delivered_parser_parsed', 0)} / {parser_audit.get('delivered_parser_rejected', 0)}
- changed record：{parser_audit.get('changed_record_count', 0)}
- formal artifacts rewritten：{str(parser_audit.get('formal_artifacts_rewritten', False)).lower()}
"""
    else:
        migration_section = ""
    report = f"""# Task 2D 单条 Qwen 不可变草稿报告

## 状态

- 状态：{status}
- 完成调用：{manifest['completed_calls']} / {manifest['expected_calls']}
- parsed / rejected：{manifest['parsed_count']} / {manifest['rejected_count']}
- 总耗时：{manifest['elapsed_seconds']:.3f} 秒
- 产物边界：仅生成内容寻址、追加式不可变草稿；不生成 safety，不签发训练样本。

## RED / GREEN 与测试

- 严格按 RED→GREEN 新增 single-draft 行为测试，模型调用不替代 fake client/tokenizer/model 测试。
- {test_summary}

## 离线与 provenance

- 完全离线，本地模型在加载前校验 manifest；只允许 BF16/CUDA，无 CPU、dtype 或在线 fallback。
- HF revision：`{manifest['model_revisions']['hf']}`
- ModelScope revision：`{manifest['model_revisions']['modelscope']}`
- catalog SHA-256：`{manifest['input_sha256']['catalog']}`
- guidelines SHA-256：`{manifest['input_sha256']['guidelines']}`
- model manifest SHA-256：`{manifest['model_manifest_sha256']}`
- generator script SHA-256：`{manifest['script_sha256']}`
- delivered script SHA-256：`{delivered_script_sha256 or manifest['script_sha256']}`

## 每标签结果

{chr(10).join(rows)}

{reason_section}

{superseded_section}

{migration_section}

## Concerns

{chr(10).join(f'- {item}' for item in concerns)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)


def write_pilot_report(manifest: dict[str, Any], path: Path = REPORT_PATH) -> None:
    concerns: list[str] = []
    if manifest["requirement_corrections"]:
        concerns.append(
            "Brief 中不存在于 catalog 的 `object:original:stove` 已按上级纠正为真实标签 "
            "`object:vla:vla_011`（炉灶），manifest 保留了纠正记录。"
        )
    if manifest["rejected_count"]:
        concerns.append(
            f"{manifest['rejected_count']} 条候选未通过机械结构/span/label/数量/去重检查；"
            "它们仅保留在 rejected log，未被自动修复或签发。"
        )
    if not concerns:
        concerns.append("无。")
    rows = [
        "| batch | kind | parsed | rejected | resumed | seconds |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for batch in manifest["batches"]:
        rows.append(
            f"| `{batch['batch_label']}` | {batch['kind']} | {batch['parsed_count']} | "
            f"{batch['rejected_count']} | {str(batch['resumed']).lower()} | "
            f"{batch['elapsed_seconds']:.3f} |"
        )
    report = f"""# Task 2C Qwen 离线候选试点报告

## 状态

- 状态：{'DONE_WITH_CONCERNS' if concerns != ['无。'] else 'DONE'}
- 批次：{manifest['batch_count']} / {manifest['expected_batch_count']}
- raw / parsed / rejected：{manifest['raw_count']} / {manifest['parsed_count']} / {manifest['rejected_count']}
- 总耗时：{manifest['elapsed_seconds']:.3f} 秒

## RED / GREEN

- RED：逐项观察到模块、object/scene/safety prompt、严格解析、离线加载、fake Qwen client、断点续跑、manifest/CLI 等行为在实现前按预期失败。
- GREEN：`tests/test_generate_spoken_annotation_candidates.py` 专项单元测试通过后才进入实际模型试点；实际模型生成不替代 fake tokenizer/model/client 单测。

## 离线加载

- 加载前调用 `download_annotation_generator.verify_directory` 校验可信 13 文件基线。
- tokenizer/model 均仅从本地加载；`local_files_only=True`、`trust_remote_code=False`，model 另设 `use_safetensors=True`、BF16、CUDA；无联网或 CPU/dtype 降级。
- HF revision：`{manifest['model_revisions']['hf']}`
- ModelScope revision：`{manifest['model_revisions']['modelscope']}`

## 每批结果

{chr(10).join(rows)}

## Concerns

{chr(10).join(f'- {item}' for item in concerns)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate strictly offline spoken-annotation candidates with local Qwen."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="catalog labels and/or safety:<class>; default is the 21-batch pilot",
    )
    parser.add_argument("--candidates-per-label", type=int)
    parser.add_argument(
        "--single-draft",
        action="store_true",
        help="generate one JSON object per call into the immutable Task 2D draft area",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    model_loader: Callable[..., tuple[Any, Any, dict[str, Any]]] = load_local_qwen,
) -> int:
    args = _build_parser().parse_args(argv)
    count = args.candidates_per_label
    if count is None:
        count = 30 if args.single_draft else 12
    if count <= 0:
        raise ValueError("--candidates-per-label must be positive")
    catalog_payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = catalog_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("catalog entries must be an array")
    if args.single_draft:
        labels = list(SINGLE_DRAFT_LABELS) if args.labels is None else args.labels
        specs = build_single_draft_specs(entries, labels=labels)
        output_dir = SINGLE_DRAFT_OUTPUT_DIR
    else:
        labels = args.labels
        specs = build_batch_specs(entries, labels=labels, count=count)
        output_dir = DEFAULT_OUTPUT_DIR
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "single-draft" if args.single_draft else "pilot-batch",
                    "batch_count": len(specs),
                    "label_count": len(specs),
                    "candidate_count": len(specs) * count,
                    "labels": [spec.label for spec in specs],
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    tokenizer, model, model_manifest = model_loader(DEFAULT_MODEL_DIR)
    import torch

    client = QwenLocalClient(tokenizer, model, torch)
    model_revisions = {
        "hf": str(model_manifest["revision"]),
        "modelscope": str(model_manifest["transport_revision"]),
    }
    generation_parameters = {
        "max_new_tokens": 512 if args.single_draft else 4096,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    }
    shared_hashes = {
        "catalog": sha256_file(CATALOG_PATH),
        "guidelines": sha256_file(GUIDELINES_PATH),
    }
    model_manifest_sha256 = sha256_file(DEFAULT_MODEL_DIR / "manifest.json")
    script_sha256 = sha256_file(Path(__file__))
    if args.single_draft:
        manifest = run_single_draft_pack(
            entries=entries,
            client=client,
            output_dir=SINGLE_DRAFT_OUTPUT_DIR,
            labels=labels,
            drafts_per_label=count,
            base_seed=314159,
            generation_parameters=generation_parameters,
            model_revisions=model_revisions,
            input_hashes=shared_hashes,
            model_manifest_sha256=model_manifest_sha256,
            script_sha256=script_sha256,
        )
        write_single_draft_report(manifest)
    else:
        manifest = run_pilot(
            entries=entries,
            client=client,
            output_dir=DEFAULT_OUTPUT_DIR,
            labels=labels,
            count=count,
            seed=314159,
            generation_parameters=generation_parameters,
            model_revisions=model_revisions,
            input_hashes=shared_hashes,
            model_manifest_sha256=model_manifest_sha256,
            script_sha256=script_sha256,
        )
        write_pilot_report(manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_object_prompt(entry: dict[str, Any], count: int) -> str:
    """Build a label-isolated object prompt from reviewed catalog context."""

    context = {
        "label": entry["label"],
        "object_name": entry["object_name_zh"],
        "task": entry["task"],
        "catalog_actions": entry["detailed_actions"],
    }
    return f"""你是独立候选撰写员。只为下面这一个对象标签生成 {count} 条自然口语中文命令。
标签上下文：{json.dumps(context, ensure_ascii=False, sort_keys=True)}

先按常识判断对象的真实角色，包括固定设施、工具、容器、电器、耗材等；若 catalog_actions 与常识冲突，以物体真实可操作性为准。每条命令必须直接提及该对象，并包含机器人可真实执行的操作证据。不要复述详细动作序列本身，不要生成其他标签的命令，不要让固定设施被搬动或归位。

span 是硬性机械约束：object_mentions 和 operation_evidence 的每个值，都必须从同一项 text 中逐字复制一个连续片段，不能概括、扩写或换同义词。例如 text="把窗户打开" 时应写 "object_mentions":["窗户"],"operation_evidence":["打开"]。

仅返回严格 JSON 数组，不要 Markdown 或解释，数组长度必须为 {count}。每项字段必须包含：
{{"text":"...","label_id":"{entry['label']}","expected_executable":true,"object_mentions":["text 中的直接原文 span"],"operation_evidence":["text 中的直接原文 span"],"semantic_rationale":"简短语义理由"}}
所有 text 必须彼此自然多样，不得只增删礼貌词。"""


def build_scene_prompt(entry: dict[str, Any], count: int) -> str:
    """Build a label-isolated scene prompt from its canonical goal and names."""

    context = {
        "label": entry["label"],
        "canonical_goal": entry["command"],
        "object_names": [item["object_name_zh"] for item in entry["objects"]],
    }
    return f"""你是独立候选撰写员。只为下面这一个多设备场景标签生成 {count} 条自然口语中文命令。
场景上下文：{json.dumps(context, ensure_ascii=False, sort_keys=True)}

每条命令要表达 canonical_goal 对应的整体生活目标，不机械枚举物体。不得要求搬动或归位固定设施，不得把垃圾恢复原状，不得把同义对象当成两个对象。命令应足以指向整体目标，但不要复述详细动作序列。

span 是硬性机械约束：object_mentions 和 operation_evidence 的每个值，都必须从同一项 text 中逐字复制一个连续片段，不能概括、扩写或换同义词。例如 text="把窗户打开" 时应写 "object_mentions":["窗户"],"operation_evidence":["打开"]。

仅返回严格 JSON 数组，不要 Markdown 或解释，数组长度必须为 {count}。每项字段必须包含：
{{"text":"...","label_id":"{entry['label']}","expected_executable":true,"object_mentions":["text 中的直接原文 span"],"operation_evidence":["text 中的直接原文 span"],"semantic_rationale":"简短语义理由"}}
所有 text 必须彼此自然多样，不得只增删礼貌词。"""


if __name__ == "__main__":
    raise SystemExit(main())
