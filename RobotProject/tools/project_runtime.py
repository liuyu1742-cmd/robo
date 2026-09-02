"""Checkpoint-safe runtime generation for manual project instructions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from tools.manual_instruction_entry import ManualInstruction
from tools.project_conditions import MANUAL_SOURCE, OPERATION_TO_VERB
from tools.project_encode import (
    ACTION_VOCAB,
    ARG_VOCAB,
    OBJECT_VOCAB,
    TARGET_VOCAB,
    TASK_VOCAB,
    encode_object,
    encode_target,
    encode_task,
)
from tools.project_generation import joint_beam_search
from train_project_transformer_v2 import (
    ProjectConditionedTransformer,
    SOURCE_VOCAB,
    VERB_VOCAB,
)


@dataclass(frozen=True)
class RuntimeGeneration:
    actions: list[str]
    score: float
    source: str
    verb: str
    terminated: bool
    truncated: bool
    invalid: bool = False


def load_project_v2_model(
    checkpoint_path: Path, device: str
) -> tuple[ProjectConditionedTransformer, dict]:
    """Load the v2 model only if every persisted vocabulary is current."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Retrain the model first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_vocabularies = {
        "source_vocab": SOURCE_VOCAB,
        "verb_vocab": VERB_VOCAB,
        "task_vocab": TASK_VOCAB,
        "object_vocab": OBJECT_VOCAB,
        "action_vocab": ACTION_VOCAB,
        "argument_vocab": ARG_VOCAB,
        "target_vocab": TARGET_VOCAB,
    }
    for key, expected in expected_vocabularies.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key} does not match current vocab. Retrain the model."
            )
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing v2 model configuration or weights. Retrain the model.")
    model = ProjectConditionedTransformer(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def generate_for_instruction(
    model: ProjectConditionedTransformer,
    resolved: ManualInstruction,
    *,
    device: str,
    beam_width: int = 3,
    max_actions: int = 12,
) -> RuntimeGeneration:
    """Generate from parser-derived conditions without looking up reference actions."""
    try:
        verb = OPERATION_TO_VERB[resolved.operation]
    except KeyError as exc:
        raise ValueError(
            f"No model verb mapping for operation: {resolved.operation}"
        ) from exc

    encoded = {
        "verb_id": VERB_VOCAB[verb],
        "task_id": encode_task(resolved.task),
        "object_id": encode_object(resolved.object),
        "target_id": encode_target(resolved.target),
    }
    torch_device = torch.device(device)
    if MANUAL_SOURCE not in SOURCE_VOCAB:
        raise ValueError(f"Current source vocab does not contain {MANUAL_SOURCE}.")
    plan = joint_beam_search(
        model,
        source_id=SOURCE_VOCAB[MANUAL_SOURCE],
        **encoded,
        device=torch_device,
        beam_width=beam_width,
        max_actions=max_actions,
    )
    return RuntimeGeneration(
        actions=plan.actions,
        score=plan.score,
        source=MANUAL_SOURCE,
        verb=verb,
        terminated=plan.terminated,
        truncated=plan.truncated,
        invalid=plan.invalid,
    )
