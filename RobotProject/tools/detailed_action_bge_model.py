"""Small, testable heads used above the local BGE text encoder."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional


def mean_pool_embeddings(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool valid token states without allowing padding to affect scores."""
    weights = attention_mask.to(token_embeddings.dtype).unsqueeze(-1)
    return (token_embeddings * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def cls_pool_embeddings(token_embeddings: torch.Tensor) -> torch.Tensor:
    """Return the first-token representation used by BGE sentence embeddings."""
    if token_embeddings.ndim != 3 or token_embeddings.shape[1] == 0:
        raise ValueError("token_embeddings must have shape (batch, sequence, hidden)")
    return token_embeddings[:, 0]


def load_local_bge_encoder(path: str | Path, device: str = "cpu"):
    """Load the pinned local encoder with networking disabled."""
    location = Path(path)
    if not location.is_dir():
        raise FileNotFoundError(f"local BGE encoder is missing: {location}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    encoder = AutoModel.from_pretrained(location, local_files_only=True)
    return tokenizer, encoder.to(device).eval()


def dual_head_loss(
    output: dict[str, torch.Tensor], intent_targets: torch.Tensor, gate_targets: torch.Tensor
) -> torch.Tensor:
    """Equal-weight intent and executable-gate cross-entropy objective."""
    return functional.cross_entropy(output["intent_logits"], intent_targets) + functional.cross_entropy(
        output["gate_logits"], gate_targets
    )


def supervised_contrastive_loss(
    representations: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull same-label training views together while separating other labels."""
    if representations.ndim != 2 or labels.ndim != 1 or representations.shape[0] != labels.shape[0]:
        raise ValueError("representations and labels must be aligned batches")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    vectors = functional.normalize(representations, dim=1)
    similarity = vectors @ vectors.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    similarity = similarity.masked_fill(identity, float("-inf"))
    positive = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~identity
    anchors = positive.any(1)
    if not anchors.any():
        return representations.sum() * 0.0
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    return -(log_probability.masked_fill(~positive, 0.0).sum(1)[anchors] / positive.sum(1)[anchors]).mean()


def rerank_topk_scores(
    logits: torch.Tensor,
    auxiliary_scores: torch.Tensor,
    *,
    top_k: int,
    weight: float,
) -> torch.Tensor:
    """Fuse auxiliary evidence only within the model's original top-k labels."""
    if logits.shape != auxiliary_scores.shape or logits.ndim != 2:
        raise ValueError("logits and auxiliary_scores must share a 2-D shape")
    if not 1 <= top_k <= logits.shape[1]:
        raise ValueError("top_k is outside the label dimension")
    candidates = logits.topk(top_k, dim=1).indices
    fused = logits + float(weight) * auxiliary_scores
    restricted = torch.full_like(fused, float("-inf"))
    restricted.scatter_(1, candidates, fused.gather(1, candidates))
    return restricted


def rerank_finite_scores(
    logits: torch.Tensor,
    auxiliary_scores: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Fuse auxiliary scores without reviving candidates already masked out."""
    if logits.shape != auxiliary_scores.shape or logits.ndim != 2:
        raise ValueError("logits and auxiliary_scores must share a 2-D shape")
    fused = logits + float(weight) * auxiliary_scores
    return fused.masked_fill(~torch.isfinite(logits), float("-inf"))


def load_character_gate(payload: dict, device: str | torch.device = "cpu"):
    """Restore an optional character n-gram execution gate embedded in BGE checkpoint."""
    config = payload.get("character_gate")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain character_gate")
    from tools.semantic_action_model import CharacterNgramRelationClassifier

    vocabulary = dict(config["vocabulary"])
    model = CharacterNgramRelationClassifier(
        len(vocabulary), 2, int(config["embedding_dim"])
    ).to(device)
    model.load_state_dict(config["state_dict"])
    model.eval()
    return model, vocabulary, float(config["threshold"])


def combine_scene_object_logits(
    primary_logits: torch.Tensor,
    scene_logits: torch.Tensor,
    scene_mask: torch.Tensor,
    *,
    scene_bias: float,
) -> torch.Tensor:
    """Use a scene-specialized head only for scene columns."""
    if primary_logits.shape != scene_logits.shape or scene_mask.shape != (primary_logits.shape[1],):
        raise ValueError("scene/object logit shapes do not match")
    combined = primary_logits.clone()
    combined[:, scene_mask] = scene_logits[:, scene_mask] + float(scene_bias)
    return combined


def combine_scene_object_logits_conditionally(
    primary_logits: torch.Tensor,
    scene_logits: torch.Tensor,
    scene_mask: torch.Tensor,
    *,
    scene_bias: float,
    primary_scene_override_margin: float,
) -> torch.Tensor:
    """Keep strong primary-scene rows while retaining specialist protection elsewhere."""
    combined = combine_scene_object_logits(
        primary_logits, scene_logits, scene_mask, scene_bias=scene_bias
    )
    if not scene_mask.any() or scene_mask.all():
        raise ValueError("conditional scene fusion requires scene and object columns")
    primary_scene = primary_logits[:, scene_mask].max(1).values
    primary_object = primary_logits[:, ~scene_mask].max(1).values
    override = primary_scene - primary_object >= float(primary_scene_override_margin)
    combined[override] = primary_logits[override]
    return combined


def route_scene_object_logits(
    primary_logits: torch.Tensor,
    scene_logits: torch.Tensor,
    scene_mask: torch.Tensor,
    *,
    scene_type_scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Route each row exclusively through the scene or object label group."""
    if primary_logits.ndim != 2 or primary_logits.shape != scene_logits.shape:
        raise ValueError("primary and scene logits must share one 2-D shape")
    if scene_mask.shape != (primary_logits.shape[1],):
        raise ValueError("scene_mask must match the label dimension")
    if scene_type_scores.shape != (primary_logits.shape[0],):
        raise ValueError("scene_type_scores must match the batch dimension")
    if not scene_mask.any() or scene_mask.all():
        raise ValueError("scene routing requires both scene and object labels")

    routed = primary_logits.clone()
    routed[:, scene_mask] = scene_logits[:, scene_mask]
    scene_rows = scene_type_scores >= float(threshold)
    routed = routed.masked_fill(
        scene_rows.unsqueeze(1) & ~scene_mask.unsqueeze(0), float("-inf")
    )
    routed = routed.masked_fill(
        ~scene_rows.unsqueeze(1) & scene_mask.unsqueeze(0), float("-inf")
    )
    return routed


def apply_scene_rescue(
    final_logits: torch.Tensor,
    primary_logits: torch.Tensor,
    scene_logits: torch.Tensor,
    scene_mask: torch.Tensor,
    *,
    threshold: float,
    method: str = "specialist_gap",
    auxiliary_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace object predictions only when the scene specialist wins decisively."""
    if final_logits.ndim != 2 or final_logits.shape != primary_logits.shape:
        raise ValueError("final and primary logits must share one 2-D shape")
    if scene_logits.shape != final_logits.shape:
        raise ValueError("scene logits must match final logits")
    if scene_mask.shape != (final_logits.shape[1],):
        raise ValueError("scene_mask must match the label dimension")
    if not scene_mask.any() or scene_mask.all():
        raise ValueError("scene rescue requires both scene and object labels")

    result = final_logits.clone()
    current_is_object = ~scene_mask[final_logits.argmax(1)]
    best_scene_values, best_scene_local = scene_logits[:, scene_mask].max(1)
    if method == "specialist_gap":
        rescue_scores = best_scene_values - primary_logits[:, ~scene_mask].max(1).values
    elif method == "lexical_gap":
        if auxiliary_scores is None or auxiliary_scores.shape != final_logits.shape:
            raise ValueError("lexical scene rescue requires aligned auxiliary_scores")
        rescue_scores = (
            auxiliary_scores[:, scene_mask].max(1).values
            - auxiliary_scores[:, ~scene_mask].max(1).values
        )
    else:
        raise ValueError("unsupported scene rescue method")
    rescue = current_is_object & (
        rescue_scores >= float(threshold)
    )
    scene_columns = torch.where(scene_mask)[0]
    proposed = scene_columns[best_scene_local]
    if rescue.any():
        result[rescue] = float("-inf")
        result[rescue, proposed[rescue]] = best_scene_values[rescue]
    return result


def apply_object_surface_override(
    final_logits: torch.Tensor,
    primary_logits: torch.Tensor,
    scene_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    scene_guard_logits: torch.Tensor | None = None,
    scene_guard_margin: float | None = None,
) -> torch.Tensor:
    """Constrain primary-object rows to object labels named in the input text."""
    if final_logits.ndim != 2 or final_logits.shape != primary_logits.shape:
        raise ValueError("final and primary logits must share one 2-D shape")
    if candidate_mask.shape != final_logits.shape or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be a boolean matrix aligned with logits")
    if scene_mask.shape != (final_logits.shape[1],):
        raise ValueError("scene_mask must match the label dimension")
    primary_is_object = ~scene_mask[primary_logits.argmax(1)]
    eligible = primary_is_object & candidate_mask.any(1)
    if scene_guard_logits is not None or scene_guard_margin is not None:
        if scene_guard_logits is None or scene_guard_margin is None:
            raise ValueError("scene guard logits and margin must be provided together")
        if scene_guard_logits.shape != final_logits.shape:
            raise ValueError("scene_guard_logits must align with final logits")
        scene_strength = scene_guard_logits[:, scene_mask].max(1).values
        object_strength = primary_logits[:, ~scene_mask].max(1).values
        eligible &= (scene_strength - object_strength) < float(scene_guard_margin)
    constrained = primary_logits.masked_fill(~candidate_mask, float("-inf"))
    proposed = constrained.argmax(1)
    result = final_logits.clone()
    if eligible.any():
        result[eligible] = float("-inf")
        result[eligible, proposed[eligible]] = primary_logits[eligible, proposed[eligible]]
    return result


def apply_scene_text_rescue(
    final_logits: torch.Tensor,
    scene_logits: torch.Tensor,
    scene_mask: torch.Tensor,
    texts: list[str],
    patterns: list[str],
) -> torch.Tensor:
    """Route matching object-side predictions through the scene specialist."""
    if final_logits.ndim != 2 or scene_logits.shape != final_logits.shape:
        raise ValueError("final and scene logits must share one 2-D shape")
    if scene_mask.shape != (final_logits.shape[1],):
        raise ValueError("scene_mask must match the label dimension")
    if len(texts) != final_logits.shape[0]:
        raise ValueError("texts must align with the batch dimension")
    compiled = [re.compile(pattern) for pattern in patterns]
    signal = torch.tensor(
        [any(pattern.search(str(text)) for pattern in compiled) for text in texts],
        dtype=torch.bool,
        device=final_logits.device,
    )
    predicted_object = ~scene_mask.to(final_logits.device)[final_logits.argmax(1)]
    eligible = signal & predicted_object
    result = final_logits.clone()
    if eligible.any():
        device_scene_mask = scene_mask.to(final_logits.device)
        rescued = scene_logits.to(final_logits.device)[eligible].masked_fill(
            ~device_scene_mask,
            float("-inf"),
        )
        result[eligible] = rescued
    return result


def apply_object_text_rescue(
    final_logits: torch.Tensor,
    primary_logits: torch.Tensor,
    scene_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    texts: list[str],
    patterns: list[str],
) -> torch.Tensor:
    """Route matching scene-side predictions to explicit object candidates."""
    if final_logits.ndim != 2 or primary_logits.shape != final_logits.shape:
        raise ValueError("final and primary logits must share one 2-D shape")
    if candidate_mask.shape != final_logits.shape or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be a boolean matrix aligned with logits")
    if scene_mask.shape != (final_logits.shape[1],) or len(texts) != final_logits.shape[0]:
        raise ValueError("scene mask and texts must align with logits")
    compiled = [re.compile(pattern) for pattern in patterns]
    signal = torch.tensor(
        [any(pattern.search(str(text)) for pattern in compiled) for text in texts],
        dtype=torch.bool,
        device=final_logits.device,
    )
    predicted_scene = scene_mask.to(final_logits.device)[final_logits.argmax(1)]
    eligible = signal & predicted_scene & candidate_mask.any(1).to(final_logits.device)
    constrained = primary_logits.to(final_logits.device).masked_fill(
        ~candidate_mask.to(final_logits.device),
        float("-inf"),
    )
    proposed = constrained.argmax(1)
    result = final_logits.clone()
    if eligible.any():
        result[eligible] = float("-inf")
        result[eligible, proposed[eligible]] = primary_logits.to(final_logits.device)[eligible, proposed[eligible]]
    return result


def apply_object_operation_specialist_gate(
    *,
    base_scores: torch.Tensor,
    base_threshold: float | torch.Tensor,
    character_scores: torch.Tensor,
    bge_scores: torch.Tensor,
    own_capability_scores: torch.Tensor,
    global_capability_scores: torch.Tensor,
    single_object_mask: torch.Tensor,
    known_object_mask: torch.Tensor | None = None,
    config: dict,
) -> torch.Tensor:
    """Apply the calibrated object-operation veto on top of the general gate."""
    tensors = (
        character_scores,
        bge_scores,
        own_capability_scores,
        global_capability_scores,
        single_object_mask,
    )
    if any(value.shape != base_scores.shape for value in tensors):
        raise ValueError("object-operation specialist inputs must share one shape")
    if known_object_mask is None:
        known_object_mask = own_capability_scores.ge(0)
    if known_object_mask.shape != base_scores.shape:
        raise ValueError("known_object_mask must share the specialist input shape")
    character_std = max(float(config["character_std"]), 1e-6)
    bge_std = max(float(config["bge_std"]), 1e-6)
    fused = (
        (character_scores - float(config["character_mean"])) / character_std
        + float(config["bge_weight"])
        * (bge_scores - float(config["bge_mean"])) / bge_std
    )
    thresholds = torch.as_tensor(
        base_threshold, dtype=base_scores.dtype, device=base_scores.device
    )
    if thresholds.ndim == 0:
        thresholds = thresholds.expand_as(base_scores)
    if thresholds.shape != base_scores.shape:
        raise ValueError("base_threshold must be scalar or align with scores")
    base_predictions = base_scores >= thresholds
    fusion_eligible = base_predictions & (
        base_scores <= thresholds + float(config["general_margin"])
    )
    fusion_veto = fusion_eligible & (fused < float(config["fusion_threshold"]))
    capability_gap = global_capability_scores - own_capability_scores
    relation_veto = (
        single_object_mask.bool()
        & known_object_mask.bool()
        & (capability_gap >= float(config["minimum_capability_gap"]))
        & (own_capability_scores <= float(config["maximum_own_capability"]))
    )
    unknown_veto = (
        single_object_mask.bool()
        & ~known_object_mask.bool()
        & base_predictions
        & (
            base_scores
            <= thresholds + float(config.get("unknown_object_margin", 0.0))
        )
    )
    return base_predictions & ~(fusion_veto | relation_veto | unknown_veto)


def apply_short_object_margin_gate(
    *,
    predictions: torch.Tensor,
    scores: torch.Tensor,
    threshold: float | torch.Tensor,
    text_lengths: torch.Tensor,
    known_object_mask: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Reject only short known-object commands sitting near the gate boundary."""
    if any(value.shape != predictions.shape for value in (scores, text_lengths, known_object_mask)):
        raise ValueError("short-object gate inputs must share one shape")
    thresholds = torch.as_tensor(threshold, dtype=scores.dtype, device=scores.device)
    if thresholds.ndim == 0:
        thresholds = thresholds.expand_as(scores)
    if thresholds.shape != scores.shape:
        raise ValueError("threshold must be scalar or align with scores")
    borderline = scores < thresholds + float(config["additional_margin"])
    short = text_lengths <= int(config["maximum_chars"])
    return predictions.bool() & ~(short & known_object_mask.bool() & borderline)


def load_object_operation_specialist(
    payload: dict, device: str | torch.device = "cpu"
):
    """Restore the optional object-operation compatibility specialist."""
    specialist = payload.get("object_operation_specialist")
    if not isinstance(specialist, dict):
        raise ValueError("checkpoint does not contain object_operation_specialist")
    from tools.semantic_action_model import CharacterNgramRelationClassifier

    character_config = specialist["character"]
    vocabulary = dict(character_config["vocabulary"])
    character = CharacterNgramRelationClassifier(
        len(vocabulary), 2, int(character_config["embedding_dim"])
    ).to(device)
    character.load_state_dict(character_config["state_dict"])
    character.eval()
    capability_vectors = specialist["capability_vectors"].to(device)
    bge_head = nn.Linear(int(capability_vectors.shape[1]), 2).to(device)
    bge_head.load_state_dict(specialist["bge_head_state"])
    bge_head.eval()
    return (
        character,
        vocabulary,
        bge_head,
        list(specialist["capability_labels"]),
        capability_vectors,
        dict(specialist["config"]),
    )


@torch.no_grad()
def encode_bge_texts(
    tokenizer,
    encoder,
    texts: list[str],
    device: str = "cpu",
    pooling: str = "mean",
) -> torch.Tensor:
    """Encode a small batch through a local BGE model.

    ``mean`` remains available for backwards compatibility; ``cls`` follows
    the official BGE sentence-embedding pooling convention.
    """
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = encoder(**encoded).last_hidden_state
    if pooling == "mean":
        return mean_pool_embeddings(hidden, encoded["attention_mask"])
    if pooling == "cls":
        return cls_pool_embeddings(hidden)
    raise ValueError(f"unsupported pooling mode: {pooling}")


class DetailedActionBgeClassifier(nn.Module):
    """Classify a pooled BGE embedding and gate executable requests."""

    def __init__(self, hidden_size: int, label_count: int) -> None:
        super().__init__()
        self.intent_head = nn.Linear(hidden_size, label_count)
        self.gate_head = nn.Linear(hidden_size, 2)

    def forward(self, pooled_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "intent_logits": self.intent_head(pooled_embeddings),
            "gate_logits": self.gate_head(pooled_embeddings),
        }


def load_detailed_action_bge(
    checkpoint: str | Path,
    encoder_dir: str | Path,
    device: str = "cpu",
):
    """Load a locally pinned BGE encoder and its dual heads for inference."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = list(payload.get("labels", ()))
    if not labels:
        raise ValueError("BGE checkpoint does not contain ordered labels")
    tokenizer, encoder = load_local_bge_encoder(encoder_dir, device=device)
    model = DetailedActionBgeClassifier(encoder.config.hidden_size, len(labels)).to(device)
    state = payload.get("state_dict")
    if state is None:
        raise ValueError("BGE checkpoint does not contain classifier state_dict")
    model.load_state_dict(state)
    model.eval()
    return tokenizer, encoder, model, labels, payload
