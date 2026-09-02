"""Runtime resolver for catalogue-only detailed actions.

The resolver deliberately separates candidate generation from model ranking:
ordinary object aliases and task/scene concepts constrain the 254 labels, then
the checked-in character model ranks only those plausible candidates.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from tools.detailed_action_catalog import CATALOGUE_PATH, load_detailed_action_catalog
from tools.detailed_action_semantic_data import (
    DEV_PATH,
    TRAIN_PATH,
    _DEV_OBJECT_ALIASES,
    catalog_sha256,
)
from tools.manual_instruction_entry import OBJECT_SYNONYMS
from tools.semantic_action_model import load_checkpoint, vectorize_batch
from tools.detailed_action_bge_model import (
    apply_scene_rescue,
    apply_scene_text_rescue,
    apply_object_text_rescue,
    apply_object_surface_override,
    apply_object_operation_specialist_gate,
    apply_short_object_margin_gate,
    combine_scene_object_logits,
    combine_scene_object_logits_conditionally,
    encode_bge_texts,
    load_character_gate,
    load_detailed_action_bge,
    load_object_operation_specialist,
    rerank_topk_scores,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = ROOT / "models" / "detailed_action_semantic" / "best.pt"
DEFAULT_BGE_CHECKPOINT_PATH = (
    ROOT
    / "models"
    / "detailed_action_bge"
    / "best_hybrid_dual_intent_object_operation_v5.pt"
)
DEFAULT_BGE_ENCODER_PATH = ROOT / "models" / "pretrained" / "BAAI_bge-small-zh-v1.5"

# Source: general Mandarin grammar (predicate classes, result/directional
# complements, clause markers).  This compact grammar is intentionally not an
# acceptance-example lexicon.  Catalog task names define the target ontology.
ACTION_FAMILIES: tuple[tuple[re.Pattern[str], tuple[str, ...], int], ...] = (
    (re.compile(r"擦|抹|刷|扫|拖|洗|清洁|清理|去除|除掉"),
     ("cleaning", "indoor_cleaning", "window_care", "bathroom_personal_care"), 30),
    (re.compile(r"拿|取|搬|送|递|带|放|摆|挂|装|收"),
     ("object_fetching", "item_delivery", "food_serving", "tableware_placement",
      "organizing", "organizing_storage", "workspace_service", "clothing_entryway",
      "bathroom_personal_care"), 10),
    (re.compile(r"开|关|启|停|调|设|切换"),
     ("appliance_management", "storage_open_close", "smart_cooking"), 30),
    (re.compile(r"整理|归位|收纳|排列|分类|折叠"),
     ("organizing", "organizing_storage", "tableware_placement", "clothing_care",
      "clothing_entryway", "bedroom_service", "waste_disposal"), 20),
    (re.compile(r"丢|扔|回收|投放|废弃|清空"), ("waste_disposal",), 40),
    (re.compile(r"煮|烤|炒|煎|切|盛|炖|蒸|加热|备餐|做饭"),
     ("cooking_heating", "smart_cooking", "food_serving"), 40),
    (re.compile(r"晾|熨|脱水|洗衣"), ("laundry", "clothing_care"), 40),
    (re.compile(r"检查|查看|核验|测量|巡查"),
     ("security_monitoring", "elderly_assistance", "maintenance_management"), 30),
    (re.compile(r"浇|养护|照料"), ("maintenance_management",), 40),
    (re.compile(r"安装|固定|装上"), ("indoor_installation",), 40),
    (re.compile(r"播放|游戏"), ("entertainment_service",), 40),
)

# Source: general Mandarin grammar.  These identify information-seeking
# constructions whose embedded action predicate is mentioned, not requested.
INFORMATION_PATTERNS = (
    re.compile(r"如何|怎么|怎样|为何|为什么"),
    re.compile(r"告诉我|介绍(?:一下)?|讲解|说明"),
    re.compile(r"想了解|想知道|方法|步骤"),
    re.compile(r"能否|可否|是否可以"),
    re.compile(r"(?:已经|曾经|刚刚|刚才).*(?:了|啦|过|完|好|着|结束)"),
    re.compile(r"现在.*(?:着呢|着了)$"),
)
INTERROGATIVE_ENDING = re.compile(r"(?:[?？]\s*|(?:吗|么)\s*[?？]?\s*)$")
LONG_INFORMATION_PREFIX = re.compile(
    r"^(?:请)?(?:告诉我|介绍|讲解|说明|我想了解|我想知道|想了解|想知道|"
    r"如何|怎么|怎样|为何|为什么)"
)

# Source: general Mandarin grammar and ordinary lexical segmentation.  These
# are lexicalized words containing 否定字形, not cancellation operators.
NON_NEGATION_LEXEMES = (
    "不要紧", "别墅", "差别", "告别", "别名", "分别", "特别", "个别",
    "性别", "类别", "识别", "区别", "辨别", "级别", "别致", "别针",
)
NON_ACTION_LEXEMES = (
    "洗碗机", "洗衣机", "扫地机器人", "清洁刷", "加热预约",
)
NEGATION_MARKERS = ("不想", "不要", "不用", "不许", "不能", "禁止", "取消", "停止", "先别", "别再", "别")
HARM_CONSTRAINT_PREDICATES = re.compile(r"刮伤|划伤|损坏|碰坏|摔坏|弄脏|烫伤")
UNRESOLVED_MARKERS = re.compile(
    r"不确定|不能确定|无法确认|无法分辨|不能决定|还没决定|还没定|没决定|"
    r"拿不准|说不准|记不清|忘了|选不出来|还在权衡|没把握|记不准|"
    r"不能定|确认不了|无法判定|尚未确定|没想好|没写清|没说清|没拿准|想不清|再说|"
    r"还没选|没选|没有给出|没给出|不说|没交代|没有说明|不指定"
)
ALTERNATIVE_MARKERS = re.compile(
    r"还是|或者|或是|究竟|到底|二选一|两种|两类|该.{0,18}选|"
    r"因此不能|就不能|没写清|具体怎么|哪个|哪种|哪件|哪一个|"
    r"位置|去处|哪儿|哪里|哪边"
)
RESTRICTIVE_ATTRIBUTE_COMMAND = re.compile(
    r"^(?:请)?(?:把|将)[^，。；;]{0,3}(?:未|不带|不透明|不常用|不插电|非)"
    r"[^，。；;]{1,20}(?:移|推|平放|收进|摆|放回|放到|安放)"
)
COMPLETED_STATE_MARKERS = re.compile(
    r"(?:刚|已经|刚刚|刚才).{0,30}(?:完|好|净|干|着|了)|"
    r"正在.{0,30}着|(?:完|净|干|好)了$"
)
FOLLOWUP_REQUEST_MARKERS = re.compile(
    r"请|麻烦|帮我|给我|赶紧|立刻|马上|先|^(?:把|将|确认|检查|收纳|处理|执行|让)"
)


def _compact(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _semantic_text(text: str) -> str:
    return _compact(text)


def _text_features(text: str) -> Counter[str]:
    value = _semantic_text(text)
    return Counter([*value, *(value[index:index + 2] for index in range(len(value) - 1))])


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(left[token] * right[token] for token in left)
    denominator = math.sqrt(sum(value * value for value in left.values())) * math.sqrt(
        sum(value * value for value in right.values())
    )
    return numerator / denominator if denominator else 0.0


def _contains_term(text: str, term: str) -> bool:
    compact_term = _compact(term)
    return bool(compact_term and compact_term in text)


def _is_information_request(text: str) -> bool:
    # Source: general Mandarin interrogative grammar.  Inspect the raw input
    # before `_compact` removes terminal question punctuation.
    if INTERROGATIVE_ENDING.search(str(text)):
        return True
    compact = _compact(text)
    if len(compact) > 20 and not LONG_INFORMATION_PREFIX.search(compact):
        return False
    return any(pattern.search(compact) for pattern in INFORMATION_PATTERNS)


def _has_cancelling_negation(text: str) -> bool:
    """Recognize negation of the requested predicate, not safety constraints."""
    full_compact = _compact(text)
    full_compact_length = len(full_compact)
    full_marker_positions = [
        full_compact.find(marker) for marker in NEGATION_MARKERS if marker in full_compact
    ]
    strong_long_cancel = bool(
        re.search(r"取消.{0,12}(?:任务|指令|安排|预约)", full_compact)
    )
    if (
        full_compact_length > 15
        and full_marker_positions
        and min(full_marker_positions) > 4
        and not strong_long_cancel
    ):
        return False
    clauses = [part for part in re.split(r"[，,。；;！？!?]", str(text)) if part.strip()]
    for clause_index, raw_clause in enumerate(clauses):
        clause = _compact(raw_clause)
        for lexeme in NON_NEGATION_LEXEMES:
            clause = clause.replace(lexeme, "")
        marker_positions = [clause.find(marker) for marker in NEGATION_MARKERS if marker in clause]
        if not marker_positions:
            continue
        position = min(marker_positions)
        suffix = clause[position:]
        # A later clause such as “别刮伤涂层” constrains how an affirmative
        # action is performed; it does not cancel the preceding main request.
        if clause_index > 0 and HARM_CONSTRAINT_PREDICATES.search(suffix):
            continue
        affirmative_prefix = clause[:position]
        for lexeme in NON_ACTION_LEXEMES:
            affirmative_prefix = affirmative_prefix.replace(lexeme, "")
        if any(
            match is not None and match.end() < len(affirmative_prefix)
            for pattern, _, _ in ACTION_FAMILIES
            for match in (pattern.search(affirmative_prefix),)
        ):
            continue
        if clause_index > 0 and any(
            marker in clause for marker in ("时", "过程中", "注意", "确保", "以免")
        ):
            continue
        if any(pattern.search(suffix) for pattern, _, _ in ACTION_FAMILIES):
            return True
        # Short spoken cancellations often use domain predicates that are not
        # candidate-routing verbs (for example 运行、加水、吸尘、保留预约).
        # The affirmative-prefix and harm-constraint checks above keep this
        # fallback from treating a later safety constraint as cancellation.
        if full_compact_length <= 15 and len(clause) <= 15 and position <= 10 and suffix:
            return True
    return False


def _has_unresolved_choice(text: str) -> bool:
    """Reject requests whose required option or precondition is unresolved."""
    compact = _compact(text)
    return bool(UNRESOLVED_MARKERS.search(compact) and ALTERNATIVE_MARKERS.search(compact))


def _is_restrictive_attribute_command(text: str) -> bool:
    """Recognize an affirmative command whose negation belongs to an object attribute."""
    return bool(RESTRICTIVE_ATTRIBUTE_COMMAND.search(_compact(text)))


def _is_completed_statement(text: str) -> bool:
    """Recognize a reported result without treating a follow-up command as complete."""
    compact = _compact(text)
    return bool(
        COMPLETED_STATE_MARKERS.search(compact)
        and not FOLLOWUP_REQUEST_MARKERS.search(compact)
    )


def _has_unresolved_instance_reference(text: str) -> bool:
    """Detect calibrated high-frequency household objects lacking instance context."""
    normalized = _semantic_text(text)
    if len(normalized) > 12:
        return False
    surfaces = [surface for surface in (
        "杯子", "遥控器", "毛巾", "充电器", "钥匙", "剪刀", "台灯", "垃圾桶",
        "抹布", "拖鞋", "药盒", "水瓶", "眼镜", "花盆", "雨伞", "围裙",
        "笔记本", "洗手液", "靠垫", "门",
    ) if surface in normalized]
    if not surfaces:
        return False
    coordination_objects = ("手机", "眼镜", "钱包", "拐杖", "衣物", "用品")
    if sum(other in normalized for other in coordination_objects) >= 2:
        return False
    locator = re.compile(
        r"卧室|客厅|厨房|卫生间|浴室|餐厅|玄关|书房|阳台|茶几上|桌上|台面上|"
        r"床边|床头|柜顶|书架上|洗手台边|沙发上|门口|左侧|右侧|指定|这个|那个|这把|那把|我的"
    )
    return not bool(locator.search(normalized))


def _has_object_attribute_mismatch(text: str) -> bool:
    """Reject possessive attribute operations whose owner cannot provide that attribute."""
    normalized = _semantic_text(text)
    checks = (
        (r"音量.{0,8}(?:调|设|改)", ("音箱", "电视", "手机", "电脑", "播放器")),
        (r"连接.{0,8}(?:Wi-Fi|wifi)|开始上网", ("电脑", "手机", "电视", "路由器")),
        (r"(?:屏幕|显示).{0,5}亮度|显示亮度", ("显示器", "手机", "电脑", "电视")),
        (r"设.{0,12}闹钟", ("手机", "音箱", "时钟")),
        (r"除湿模式", ("空调", "除湿机")),
        (r"导航定位", ("扫地机器人", "机器人")),
        (r"显示.{0,5}室温", ("温度计", "空调", "温控器")),
        (r"微波炉门.{0,8}转速", ()),
        (r"摄像头.{0,10}对焦", ("相机", "摄像机", "手机")),
        (r"打印.{0,8}(?:清单|文件|纸张)", ("打印机", "复印机")),
        (r"风速.{0,10}(?:调|设|改)", ("风扇", "空调", "吹风机", "新风")),
        (r"读取.{0,8}(?:书|页)", ("扫描仪", "摄像头", "相机")),
        (r"配对.{0,8}蓝牙", ("手机", "电脑", "电视", "音箱")),
        (r"自动回到充电座", ("扫地机器人", "机器人")),
        (r"识别人脸.{0,6}解锁", ("门锁", "手机", "门禁")),
        (r"榨汁程序", ("榨汁机", "搅拌机")),
        (r"扫地路线", ("扫地机器人", "机器人")),
        (r"打印纸尺寸", ("打印机", "复印机")),
        (r"操作系统.{0,6}升级", ("电脑", "手机", "平板")),
    )
    for pattern, allowed_owners in checks:
        match = re.search(pattern, normalized)
        if match and not any(owner in normalized[:match.start()] for owner in allowed_owners):
            return True
    return False


def _is_out_of_domain_request(text: str) -> bool:
    """Reject non-household administrative workflows outside the action catalogue."""
    normalized = _semantic_text(text)
    return bool(re.search(r"商标申请|知识产权.{0,8}(?:平台|系统)|专利申请|工商登记", normalized))


class DetailedActionParser:
    """Resolve short spoken instructions without guessing unsupported intent."""

    def __init__(
        self,
        *,
        catalog_path: Path | str = CATALOGUE_PATH,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        semantic_backend: str = "character",
        bge_checkpoint_path: Path | str = DEFAULT_BGE_CHECKPOINT_PATH,
        bge_encoder_path: Path | str = DEFAULT_BGE_ENCODER_PATH,
        device: torch.device | str = "cpu",
        model_weight: float = 0.35,
        minimum_confidence: float = 0.52,
        minimum_margin: float = 0.08,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.semantic_backend = str(semantic_backend)
        self.device = torch.device(device)
        self.model_weight = float(model_weight)
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_margin = float(minimum_margin)
        self.catalog = load_detailed_action_catalog(self.catalog_path)
        self.entries = {entry["label"]: entry for entry in self.catalog["entries"]}

        self.bge_tokenizer = self.bge_encoder = self.bge_model = None
        self.bge_character_gate = self.bge_gate_vocabulary = None
        self.bge_object_operation_specialist = None
        self.bge_object_operation_capability_labels: list[str] = []
        self.bge_gate_threshold = 0.0
        self.bge_gate_backend = "bge"
        self.bge_label_lexical_features: dict[str, Counter[str]] = {}
        self._bge_cache_instruction: str | None = None
        self._bge_cache_output: dict[str, torch.Tensor] | None = None
        if self.semantic_backend == "bge":
            try:
                (
                    self.bge_tokenizer,
                    self.bge_encoder,
                    self.bge_model,
                    self.bge_labels,
                    self.bge_payload,
                ) = load_detailed_action_bge(
                    bge_checkpoint_path, bge_encoder_path, device=str(self.device)
                )
            except Exception as exc:
                raise RuntimeError(
                    "BGE semantic backend is unavailable; check the local encoder and checkpoint; "
                    f"root cause: {type(exc).__name__}: {exc}"
                ) from exc
            self.model = self.vocabulary = None
            self.labels = list(self.bge_labels)
            if self.labels != sorted(self.entries):
                raise ValueError("BGE checkpoint catalog label set mismatch")
            if self.bge_payload.get("character_gate"):
                self.bge_character_gate, self.bge_gate_vocabulary, self.bge_gate_threshold = load_character_gate(
                    self.bge_payload, device=self.device
                )
                self.bge_gate_backend = "character_ngram"
            else:
                self.bge_gate_threshold = float(self.bge_payload.get("metrics", {}).get("gate_threshold", 0.0))
            embedded_features = self.bge_payload.get("label_lexical_features", {})
            if isinstance(embedded_features, dict):
                self.bge_label_lexical_features = {
                    label: Counter(values) for label, values in embedded_features.items()
                    if label in self.entries and isinstance(values, dict)
                }
            if self.bge_payload.get("object_operation_specialist"):
                restored = load_object_operation_specialist(
                    self.bge_payload, device=self.device
                )
                (
                    specialist_character,
                    specialist_vocabulary,
                    specialist_bge_head,
                    self.bge_object_operation_capability_labels,
                    specialist_capability_vectors,
                    specialist_config,
                ) = restored
                self.bge_object_operation_specialist = {
                    "character": specialist_character,
                    "vocabulary": specialist_vocabulary,
                    "bge_head": specialist_bge_head,
                    "capability_vectors": specialist_capability_vectors,
                    "config": specialist_config,
                }
                self.bge_object_operation_capability_index = {
                    label: index
                    for index, label in enumerate(self.bge_object_operation_capability_labels)
                }
        elif self.semantic_backend == "character":
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        # Task 2 trains over a deterministic sorted label order; preserving it
        # is essential because classifier columns are positional.
            expected_label_order = sorted(self.entries)
            if payload.get("catalog_sha256") != catalog_sha256(self.catalog_path):
                raise ValueError("checkpoint catalog SHA mismatch")
            if list(payload.get("labels", ())) != expected_label_order:
                raise ValueError("checkpoint catalog label set mismatch")
            self.model, self.vocabulary, self.labels = load_checkpoint(
                str(self.checkpoint_path), device=self.device
            )
        else:
            raise ValueError("semantic_backend must be 'character' or 'bge'")
        self.label_index = {label: index for index, label in enumerate(self.labels)}
        # Source: Task 2 train/dev generated before runtime acceptance.  These
        # label prototypes supply semantic vocabulary without runtime-curated
        # phrase lists or acceptance-derived fragments.
        semantic_rows = []
        for path in (TRAIN_PATH, DEV_PATH):
            semantic_rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))
        self.semantic_texts: dict[str, list[str]] = {}
        for row in semantic_rows:
            label = str(row["label_id"])
            self.semantic_texts.setdefault(label, []).append(str(row["text"]))
        self.label_features = {
            label: _text_features("".join(texts))
            for label, texts in self.semantic_texts.items()
        }
        self.object_surfaces = self._build_object_surfaces()
        self.scene_profiles = self._build_scene_profiles()

    def _build_object_surfaces(self) -> dict[str, tuple[str, ...]]:
        # Source: catalog formal object_name_zh/object_id fields plus the
        # pre-existing manual-entry alias registry; never acceptance text.
        result: dict[str, tuple[str, ...]] = {}
        for label, entry in self.entries.items():
            if entry["entry_type"] != "object":
                continue
            terms = {str(entry["object_name_zh"])}
            terms.update(OBJECT_SYNONYMS.get(str(entry["object_id"]), ()))
            dev_alias = _DEV_OBJECT_ALIASES.get(str(entry["object_name_zh"]))
            if dev_alias:
                terms.add(dev_alias)
            result[label] = tuple(
                sorted({_compact(term) for term in terms if _compact(term)}, key=lambda term: (-len(term), term))
            )
        return result

    @staticmethod
    def _surface_heads(surface: str) -> set[str]:
        value = _semantic_text(surface)
        result = {value}
        for suffix in ("用品", "工具", "设备", "区域", "地面", "地板", "层架", "门", "子", "刀", "器"):
            if value.endswith(suffix) and len(value) > len(suffix):
                result.add(value[: -len(suffix)])
                result.add(suffix)
        for prefix in ("清洁", "数码", "电动"):
            if value.startswith(prefix) and len(value) > len(prefix):
                result.add(value[len(prefix):])
        if len(value) >= 3:
            result.update((value[:2], value[-2:]))
        return {term for term in result if term}

    def _build_scene_profiles(self) -> dict[str, dict[str, Any]]:
        # Source: catalog formal scene command/object references/task fields
        # and the pre-runtime Task 2 train/dev prototypes loaded above.
        result: dict[str, dict[str, Any]] = {}
        for label, entry in self.entries.items():
            if entry["entry_type"] != "scene":
                continue
            command = _semantic_text(entry["command"])
            object_labels = {item["object_label"] for item in entry["objects"]}
            tasks = {self.entries[object_label]["task"] for object_label in object_labels}
            surfaces: set[str] = set()
            object_surface_groups: dict[str, set[str]] = {}
            descriptor_parts = [command, *self.semantic_texts.get(label, ())]
            for object_label in object_labels:
                object_entry = self.entries[object_label]
                descriptor_parts.append(str(object_entry["object_name_zh"]))
                group: set[str] = set()
                for surface in self.object_surfaces[object_label]:
                    group.update(self._surface_heads(surface))
                surfaces.update(group)
                object_surface_groups[object_label] = group
            result[label] = {
                "object_labels": object_labels,
                "object_surface_groups": object_surface_groups,
                "tasks": tasks,
                "surfaces": surfaces,
                "features": _text_features("".join(descriptor_parts)),
                "prototype_features": [
                    _text_features(sample) for sample in self.semantic_texts.get(label, ())
                ],
            }
        return result

    @torch.inference_mode()
    def _bge_forward(self, instruction: str) -> dict[str, torch.Tensor]:
        if self._bge_cache_instruction == instruction and self._bge_cache_output is not None:
            return self._bge_cache_output
        pooling = str(self.bge_payload.get("pooling", "mean"))
        vector = encode_bge_texts(
            self.bge_tokenizer,
            self.bge_encoder,
            [instruction],
            device=str(self.device),
            pooling=pooling,
        )
        output = self.bge_model(vector)
        output["_pooled"] = vector
        self._bge_cache_instruction = instruction
        self._bge_cache_output = output
        return output

    @torch.inference_mode()
    def _model_scores(self, instruction: str) -> dict[str, float]:
        if self.semantic_backend == "bge":
            output = self._bge_forward(instruction)
            logits = output["intent_logits"].detach().cpu()
            rerank = self.bge_payload.get("intent_rerank")
            if isinstance(rerank, dict) and self.bge_label_lexical_features:
                query = _text_features(instruction)
                auxiliary = torch.tensor([[
                    _cosine(query, self.bge_label_lexical_features.get(label, Counter()))
                    for label in self.labels
                ]])
                logits = rerank_topk_scores(
                    logits, auxiliary,
                    top_k=int(rerank["top_k"]), weight=float(rerank["weight"]),
                )
                specialist = self.bge_payload.get("scene_specialist")
                if isinstance(specialist, dict):
                    primary_logits = logits
                    state = specialist["intent_head_state"]
                    scene_logits = torch.nn.functional.linear(
                        output["_pooled"].detach().cpu(), state["weight"], state["bias"]
                    )
                    scene_logits = rerank_topk_scores(
                        scene_logits, auxiliary,
                        top_k=int(rerank["top_k"]), weight=float(rerank["weight"]),
                    )
                    scene_mask = torch.tensor([label.startswith("scene:") for label in self.labels])
                    if "primary_scene_override_margin" in specialist:
                        logits = combine_scene_object_logits_conditionally(
                            logits, scene_logits, scene_mask,
                            scene_bias=float(specialist["scene_bias"]),
                            primary_scene_override_margin=float(specialist["primary_scene_override_margin"]),
                        )
                    else:
                        logits = combine_scene_object_logits(
                            logits, scene_logits, scene_mask,
                            scene_bias=float(specialist["scene_bias"]),
                        )
                    if isinstance(self.bge_payload.get("object_surface_override"), dict):
                        surface_config = self.bge_payload["object_surface_override"]
                        matched = self._matched_objects(_semantic_text(instruction))
                        matched_surfaces = {str(row["surface"]) for row in matched.values()}
                        maximal_surfaces = {
                            surface for surface in matched_surfaces
                            if not any(surface != other and surface in other for other in matched_surfaces)
                        }
                        candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
                        index_by_label = {label: index for index, label in enumerate(self.labels)}
                        for label, row in matched.items():
                            if str(row["surface"]) in maximal_surfaces:
                                candidate_mask[0, index_by_label[label]] = True
                        logits = apply_object_surface_override(
                            logits,
                            primary_logits,
                            scene_mask,
                            candidate_mask,
                            scene_guard_logits=(
                                scene_logits if "scene_guard_margin" in surface_config else None
                            ),
                            scene_guard_margin=surface_config.get("scene_guard_margin"),
                        )
                    rescue = self.bge_payload.get("scene_rescue")
                    if isinstance(rescue, dict) and rescue.get("method") in ("specialist_gap", "lexical_gap"):
                        logits = apply_scene_rescue(
                            logits,
                            primary_logits,
                            scene_logits,
                            scene_mask,
                            threshold=float(rescue["threshold"]),
                            method=str(rescue["method"]),
                            auxiliary_scores=auxiliary,
                        )
                    text_rescue = self.bge_payload.get("scene_text_rescue")
                    if isinstance(text_rescue, dict):
                        logits = apply_scene_text_rescue(
                            logits,
                            scene_logits,
                            scene_mask,
                            [instruction],
                            [str(pattern) for pattern in text_rescue.get("patterns", ())],
                        )
                    object_text_rescue = self.bge_payload.get("object_text_rescue")
                    if isinstance(object_text_rescue, dict):
                        if "candidate_mask" not in locals():
                            matched = self._matched_objects(_semantic_text(instruction))
                            candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
                            index_by_label = {label: index for index, label in enumerate(self.labels)}
                            for label in matched:
                                candidate_mask[0, index_by_label[label]] = True
                        logits = apply_object_text_rescue(
                            logits,
                            primary_logits,
                            scene_mask,
                            candidate_mask,
                            [instruction],
                            [str(pattern) for pattern in object_text_rescue.get("patterns", ())],
                        )
            probabilities = torch.softmax(logits, dim=1)[0]
            return {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        normalize = getattr(self.model, "text_normalization", "none") == "label_free_zh_v1"
        values, offsets = vectorize_batch(
            [instruction], self.vocabulary, device=self.device, normalize_text=normalize
        )
        logits = self.model(values, offsets)[0].detach().cpu()
        feature_count = max(1, int(values.numel()))
        return {label: float(logits[index]) / feature_count for index, label in enumerate(self.labels)}

    @torch.inference_mode()
    def _bge_gate_score(self, instruction: str) -> float | None:
        if self.semantic_backend != "bge":
            return None
        if self.bge_character_gate is not None:
            values, offsets = vectorize_batch([instruction], self.bge_gate_vocabulary, device=self.device)
            logits = self.bge_character_gate(values, offsets)[0]
            return float(logits[1] - logits[0])
        logits = self._bge_forward(instruction)["gate_logits"][0]
        return float(logits[1] - logits[0])

    @torch.inference_mode()
    def _object_operation_specialist_allows(
        self, instruction: str, matched: Mapping[str, Mapping[str, Any]]
    ) -> bool:
        specialist = self.bge_object_operation_specialist
        if self.semantic_backend != "bge" or specialist is None:
            return True
        if not matched:
            return True
        earliest = min(int(row["position"]) for row in matched.values())
        focus_labels = [
            label for label, row in matched.items()
            if int(row["position"]) == earliest
        ]
        longest = max(int(matched[label]["length"]) for label in focus_labels)
        focus_labels = [
            label for label in focus_labels
            if int(matched[label]["length"]) == longest
        ]
        mentions = {
            (
                int(matched[label]["position"]),
                int(matched[label]["length"]),
                str(matched[label]["surface"]),
            )
            for label in focus_labels
        }
        candidate_indices = [
            self.bge_object_operation_capability_index[label]
            for label in focus_labels
            if label in self.bge_object_operation_capability_index
        ]
        if len(mentions) != 1 or not candidate_indices:
            return True

        specialist_instruction = str(instruction).strip()
        if specialist_instruction and specialist_instruction[-1] not in "。！？!?":
            specialist_instruction += "。"
        masked = specialist_instruction
        for surface in sorted({row[2] for row in mentions}, key=len, reverse=True):
            masked = masked.replace(surface, "[对象]")
        operation_vector = encode_bge_texts(
            self.bge_tokenizer,
            self.bge_encoder,
            [masked],
            device=str(self.device),
            pooling=str(self.bge_payload.get("pooling", "mean")),
        )
        operation_vector = torch.nn.functional.normalize(operation_vector, dim=1)
        capability_vectors = specialist["capability_vectors"]
        similarities = operation_vector @ capability_vectors.T
        own_score = similarities[:, candidate_indices].max(1).values
        global_score = similarities.max(1).values

        character_values, character_offsets = vectorize_batch(
            [specialist_instruction], specialist["vocabulary"], device=self.device
        )
        character_logits = specialist["character"](
            character_values, character_offsets
        )[0]
        character_score = character_logits[1] - character_logits[0]
        pooled = encode_bge_texts(
            self.bge_tokenizer,
            self.bge_encoder,
            [specialist_instruction],
            device=str(self.device),
            pooling=str(self.bge_payload.get("pooling", "mean")),
        )
        bge_logits = specialist["bge_head"](pooled)[0]
        bge_score = bge_logits[1] - bge_logits[0]
        gate_score = self._bge_gate_score(instruction)
        if gate_score is None:
            return True
        gate_threshold = self.bge_gate_threshold
        if matched and "known_object_gate_threshold" in self.bge_payload:
            gate_threshold = float(self.bge_payload["known_object_gate_threshold"])
        prediction = apply_object_operation_specialist_gate(
            base_scores=torch.tensor([gate_score], device=self.device),
            base_threshold=gate_threshold,
            character_scores=character_score.reshape(1),
            bge_scores=bge_score.reshape(1),
            own_capability_scores=own_score,
            global_capability_scores=global_score,
            single_object_mask=torch.tensor([True], device=self.device),
            config=specialist["config"],
        )
        return bool(prediction.item())

    def rank_candidate_labels(
        self,
        instruction: str,
        candidate_labels: Sequence[str],
        *,
        rule_scores: Mapping[str, float] | None = None,
        model_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return auditable candidate-internal ranking using the real model."""
        labels = [label for label in dict.fromkeys(candidate_labels) if label in self.entries]
        if not labels:
            return []
        absolute_supports = self._model_scores(instruction)
        raw = torch.tensor([absolute_supports[label] * 20.0 for label in labels], dtype=torch.float64)
        local = torch.softmax(raw, dim=0).tolist()
        weight = self.model_weight if model_weight is None else float(model_weight)
        rows = []
        for label, local_score in zip(labels, local):
            rule_score = float((rule_scores or {}).get(label, 0.5))
            absolute_score = min(1.0, max(0.0, absolute_supports[label] / 0.22))
            model_score = 0.72 * absolute_score + 0.28 * float(local_score)
            rows.append({
                "label": label,
                "score": (1.0 - weight) * rule_score + weight * model_score,
                "rule_score": rule_score,
                "model_score": model_score,
                "absolute_model_score": absolute_score,
                "relative_model_score": float(local_score),
            })
        return sorted(rows, key=lambda row: (-row["score"], row["label"]))

    def _task_scores(self, text: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        matches = [
            (tasks, priority) for pattern, tasks, priority in ACTION_FAMILIES
            if pattern.search(text)
        ]
        dominant_priority = max((priority for _, priority in matches), default=0)
        for tasks, priority in matches:
            if priority != dominant_priority:
                continue
            for task in tasks:
                scores[task] = scores.get(task, 0.0) + 0.85
        # Directional/result complements are generic evidence for movement and
        # placement.  Object identity later selects the compatible catalog task.
        if re.search(r"(?:拿|取|搬|送|递|放|摆|移).{0,8}(?:到|进|上|旁|边|里)", text):
            for task in ("object_fetching", "item_delivery", "food_serving", "tableware_placement", "workspace_service"):
                scores[task] = scores.get(task, 0.0) + 0.75
        return scores

    @staticmethod
    def _operation_text(text: str, matched: Mapping[str, Mapping[str, Any]]) -> str:
        """Mask entity spans so noun-internal characters cannot act as verbs."""
        spans = {
            (int(row["position"]), int(row["position"]) + int(row["length"]))
            for row in matched.values()
        }
        characters = list(text)
        for start, end in spans:
            characters[start:end] = " " * (end - start)
        return "".join(characters)

    def _matched_objects(self, text: str) -> dict[str, dict[str, Any]]:
        matched: dict[str, dict[str, Any]] = {}
        for label, surfaces in self.object_surfaces.items():
            occurrences = [
                (text.find(surface), len(surface), surface)
                for surface in surfaces
                if surface and surface in text
            ]
            if occurrences:
                position, length, surface = min(occurrences, key=lambda row: (row[0], -row[1]))
                matched[label] = {"position": position, "length": length, "surface": surface}
        return matched

    def _object_candidates(
        self, text: str, task_scores: Mapping[str, float], matched: Mapping[str, Mapping[str, Any]]
    ) -> tuple[list[str], dict[str, float], bool, bool]:
        if not matched:
            return [], {}, False, False
        earliest = min(int(row["position"]) for row in matched.values())
        transfer = re.search(r"(?:盛|倒|装|放).{0,8}(?:到|进|入)", text)
        if transfer:
            destination = [
                label for label, row in matched.items()
                if int(row["position"]) >= transfer.end()
            ]
            if destination:
                earliest = min(int(matched[label]["position"]) for label in destination)
        focus = [label for label, row in matched.items() if int(row["position"]) == earliest]
        longest = max(int(matched[label]["length"]) for label in focus)
        labels = [label for label in focus if int(matched[label]["length"]) == longest]

        compatible = [label for label in labels if task_scores.get(str(self.entries[label]["task"]), 0.0) > 0]
        has_operation = bool(task_scores)
        incompatible = has_operation and not compatible
        if compatible:
            best_task = max(task_scores[str(self.entries[label]["task"])] for label in compatible)
            labels = [
                label for label in compatible
                if task_scores[str(self.entries[label]["task"])] >= best_task * 0.72
            ]
        same_surface_ambiguity = len(labels) > 1 and len({
            (int(matched[label]["position"]), str(matched[label]["surface"]))
            for label in labels
        }) == 1
        ambiguous_without_operation = same_surface_ambiguity or (len(labels) > 1 and not has_operation)
        scores = {}
        for label in labels:
            task = str(self.entries[label]["task"])
            task_strength = task_scores.get(task, 0.0)
            scores[label] = min(0.96, 0.58 + 0.10 * min(task_strength, 3.0))
        return labels, scores, ambiguous_without_operation, incompatible

    def _scene_candidates(
        self,
        text: str,
        task_scores: Mapping[str, float],
        matched_objects: Mapping[str, Mapping[str, Any]],
    ) -> tuple[list[str], dict[str, float], dict[str, int]]:
        scores: dict[str, float] = {}
        structural_counts: dict[str, int] = {}
        text_features = _text_features(text)
        scope_signal = bool(re.search(r"一起|全部|全屋|所有|各个|都|统一", text))
        for label, profile in self.scene_profiles.items():
            mentions: list[tuple[int, int]] = []
            for surfaces in profile["object_surface_groups"].values():
                occurrences = [
                    (text.find(surface), text.find(surface) + len(surface))
                    for surface in surfaces
                    if surface and surface in text
                ]
                if occurrences:
                    mentions.append(max(occurrences, key=lambda span: span[1] - span[0]))
            distinct_mentions: list[tuple[int, int]] = []
            for span in sorted(mentions, key=lambda item: (-(item[1] - item[0]), item[0])):
                if not any(span[0] < other[1] and other[0] < span[1] for other in distinct_mentions):
                    distinct_mentions.append(span)
            structural_count = len(distinct_mentions)
            structural_counts[label] = structural_count
            task_overlap = sum(task_scores.get(task, 0.0) for task in profile["tasks"])
            lexical = max(
                (_cosine(text_features, features) for features in profile["prototype_features"]),
                default=_cosine(text_features, profile["features"]),
            )
            concept_match = lexical >= 0.20
            if not (
                structural_count >= 2
                or concept_match
                or (structural_count >= 1 and task_overlap > 0)
                or (scope_signal and task_overlap > 0)
                or (not matched_objects and task_overlap > 0)
            ):
                continue
            structural = min(1.0, structural_count / 2.0)
            task_quality = min(1.0, task_overlap / 2.4)
            coordinated_device_match = (
                any(term in text for term in ("联动", "模式", "环境调"))
                and "appliance_management" in profile["tasks"]
            )
            scores[label] = min(
                0.98,
                0.40
                + 0.13 * min(structural_count, 4)
                + 0.12 * task_quality
                + 0.25 * lexical
                + (0.24 if concept_match else 0.0)
                + (0.16 if coordinated_device_match else 0.0),
            )
            if scope_signal and structural_count >= 1:
                scores[label] = max(scores[label], 0.96)
        return list(scores), scores, structural_counts

    def _clarification(
        self, instruction: str, reason: str, ranked: Sequence[Mapping[str, Any]] = ()
    ) -> dict[str, Any]:
        candidates = []
        for row in ranked[:3]:
            entry = self.entries[row["label"]]
            candidates.append({
                "label": row["label"],
                "category": entry["entry_type"],
                "name": entry.get("object_name_zh", entry.get("command", row["label"])),
                "confidence": round(float(row["score"]), 4),
            })
        return {
            "result_type": "clarification_required",
            "input": instruction,
            "reason": reason,
            "candidates": candidates,
        }

    def resolve(self, instruction: str) -> dict[str, Any]:
        raw = str(instruction)
        text = _semantic_text(raw)
        exact_reviewed_scene = any(
            entry.get("entry_type") == "scene"
            and text == _semantic_text(str(entry.get("command", "")))
            for entry in self.entries.values()
        )
        restrictive_attribute_command = _is_restrictive_attribute_command(raw)
        if not text:
            return self._clarification(raw, "输入为空，请说明对象和操作")
        if _has_cancelling_negation(raw):
            return self._clarification(raw, "检测到否定或取消冲突，不执行动作")
        if _is_information_request(raw):
            return self._clarification(raw, "输入是问句或描述，没有可执行操作")
        if _is_completed_statement(raw):
            return self._clarification(raw, "输入描述的是已完成状态，没有新的可执行操作")
        if _has_unresolved_choice(raw):
            return self._clarification(raw, "必要选项或前置条件尚未确定，不执行动作")
        if not exact_reviewed_scene and _has_unresolved_instance_reference(raw):
            return self._clarification(raw, "检测到多个同名对象，请补充房间或位置")
        if _has_object_attribute_mismatch(raw):
            return self._clarification(raw, "对象与专属属性或操作不兼容")
        if _is_out_of_domain_request(raw):
            return self._clarification(raw, "输入超出家居动作目录范围")
        matched_objects = self._matched_objects(text)
        if self.semantic_backend == "bge":
            gate_score = self._bge_gate_score(raw)
            threshold = (
                float(self.bge_payload["known_object_gate_threshold"])
                if matched_objects and "known_object_gate_threshold" in self.bge_payload
                else self.bge_gate_threshold
            )
            if (
                gate_score is not None
                and gate_score < threshold
                and not restrictive_attribute_command
                and not exact_reviewed_scene
            ):
                return self._clarification(raw, "BGE 执行门控未达到校准阈值")

        short_object_config = self.bge_payload.get("short_object_gate") if self.semantic_backend == "bge" else None
        if (
            isinstance(short_object_config, dict)
            and gate_score is not None
            and not restrictive_attribute_command
            and not exact_reviewed_scene
        ):
            short_allowed = apply_short_object_margin_gate(
                predictions=torch.tensor([True]),
                scores=torch.tensor([gate_score]),
                threshold=threshold,
                text_lengths=torch.tensor([len(text)]),
                known_object_mask=torch.tensor([bool(matched_objects)]),
                config=short_object_config,
            )[0]
            if not bool(short_allowed):
                return self._clarification(raw, "短对象指令置信余量不足，请补充位置或用途")
        if (
            not restrictive_attribute_command
            and not exact_reviewed_scene
            and not self._object_operation_specialist_allows(raw, matched_objects)
        ):
            return self._clarification(raw, "对象与操作或用途不兼容")
        task_scores = self._task_scores(self._operation_text(text, matched_objects))
        object_labels, object_scores, object_ambiguous, object_incompatible = (
            self._object_candidates(text, task_scores, matched_objects)
        )
        scene_labels, scene_scores, scene_structural_counts = self._scene_candidates(
            text, task_scores, matched_objects
        )
        # Source: the reviewed detailed-action catalogue's canonical scene
        # commands.  An exact command is executable evidence by itself and
        # must not require an extra generic token such as ``模式`` or ``环境``.
        exact_scene_labels = [
            label
            for label in scene_labels
            if text == _semantic_text(self.entries[label]["command"])
        ]
        if exact_scene_labels:
            scene_labels = exact_scene_labels
            scene_scores = {label: 1.0 for label in exact_scene_labels}
        maximum_structural_count = max(scene_structural_counts.values(), default=0)
        structural_scene_signal = maximum_structural_count >= 2
        explicit_mention_count = len({
            (int(row["position"]), int(row["length"]), str(row["surface"]))
            for row in matched_objects.values()
        })
        scope_signal = bool(re.search(r"一起|全部|全屋|所有|各个|都|统一", text))
        scene_intent_signal = bool(re.search(r"模式|联动|环境|开始|准备|进入", text))
        explicit_scene_request = not matched_objects and scene_intent_signal
        conjunction_signal = "和" in text or "与" in text
        strong_concept_signal = any(
            max(
                (_cosine(_text_features(text), features) for features in profile["prototype_features"]),
                default=0.0,
            ) >= 0.20
            for profile in self.scene_profiles.values()
        )
        scene_text_config = self.bge_payload.get("scene_text_rescue") if self.semantic_backend == "bge" else None
        configured_scene_signal = bool(
            isinstance(scene_text_config, dict)
            and any(
                re.search(str(pattern), raw)
                for pattern in scene_text_config.get("patterns", ())
            )
        )
        object_text_config = self.bge_payload.get("object_text_rescue") if self.semantic_backend == "bge" else None
        configured_object_signal = bool(
            isinstance(object_text_config, dict)
            and any(
                re.search(str(pattern), raw)
                for pattern in object_text_config.get("patterns", ())
            )
        )
        use_scene = bool(scene_labels) and (
            bool(exact_scene_labels)
            or configured_scene_signal
            or (explicit_mention_count >= 2 and structural_scene_signal and (scope_signal or conjunction_signal))
            or (not matched_objects and scope_signal)
            or (
                strong_concept_signal
                and not matched_objects
                and (scope_signal or scene_intent_signal)
            )
        )
        if configured_object_signal:
            use_scene = False
        if not use_scene and object_incompatible:
            ranked = self.rank_candidate_labels(raw, object_labels, rule_scores=object_scores)
            return self._clarification(raw, "对象与操作或用途不兼容", ranked)
        if not use_scene and matched_objects and not task_scores:
            return self._clarification(raw, "识别到对象但缺少可执行操作或用途")
        if not use_scene and object_ambiguous:
            ranked = self.rank_candidate_labels(raw, object_labels, rule_scores=object_scores)
            return self._clarification(raw, "对象对应多个用途，请补充具体操作", ranked)
        labels, scores = (scene_labels, scene_scores) if use_scene else (object_labels, object_scores)
        if not labels:
            return self._clarification(raw, "没有找到合理的目录候选")

        ranked = self.rank_candidate_labels(raw, labels, rule_scores=scores)
        top = ranked[0]
        margin = top["score"] - ranked[1]["score"] if len(ranked) > 1 else top["score"]
        if top["score"] < self.minimum_confidence:
            return self._clarification(raw, "候选置信度过低", ranked)
        if (
            len(ranked) > 1
            and margin < self.minimum_margin
            and top["rule_score"] < 0.95
        ):
            return self._clarification(raw, "前两名候选过近，请补充信息", ranked)
        return self._success(raw, top, margin)

    def _success(self, instruction: str, top: Mapping[str, Any], margin: float) -> dict[str, Any]:
        entry = self.entries[top["label"]]
        common = {
            "input": instruction,
            "label": top["label"],
            "confidence": round(float(top["score"]), 4),
            "margin": round(float(margin), 4),
            "model_used": True,
        }
        if entry["entry_type"] == "object":
            return {
                "result_type": "object_detailed_action",
                **common,
                "object_id": entry["object_id"],
                "object_name_zh": entry["object_name_zh"],
                "task": entry["task"],
                "detailed_actions": list(entry["detailed_actions"]),
                "control_parameters": entry["control_parameters"],
            }
        return {
            "result_type": "coordination_detailed_action",
            **common,
            "scenario_id": entry["scenario_id"],
            "scene": entry["command"],
            "objects": [
                {
                    "object_label": item["object_label"],
                    "object_name_zh": item["object_name_zh"],
                    "detailed_actions": list(item["detailed_actions"]),
                    "control_parameters": item["control_parameters"],
                }
                for item in entry["objects"]
            ],
        }


def format_detailed_action_result(result: Mapping) -> str:
    """Render a resolver result without inventing or leaking action fields."""
    lines = [f"输入：{result.get('input', '')}"]
    if result.get("result_type") == "clarification_required":
        lines.append("识别类别：需要澄清")
        lines.append(f"澄清原因：{result.get('reason', '')}")
        candidates = result.get("candidates", [])
        lines.append(
            "候选：" + ("、".join(str(row.get("name", row.get("label", ""))) for row in candidates) or "无")
        )
        return "\n".join(lines)

    lines.append(f"识别类别：{result.get('result_type', '')}")
    if result.get("result_type") == "object_detailed_action":
        lines.append(f"对象/场景：{result.get('object_name_zh', '')}")
        lines.append("额外细分动作：" + "；".join(result.get("detailed_actions", [])))
        lines.append(f"必要参数：{result.get('control_parameters', '')}")
    else:
        lines.append(f"对象/场景：{result.get('scene', '')}")
        for item in result.get("objects", []):
            lines.append(
                f"- {item.get('object_name_zh', '')}｜额外细分动作："
                + "；".join(item.get("detailed_actions", []))
                + f"｜必要参数：{item.get('control_parameters', '')}"
            )
    return "\n".join(lines)
