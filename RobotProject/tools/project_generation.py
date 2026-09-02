"""Autoregressive action-plan generation for the project v2 Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from tools.project_encode import (
    ACTION_VOCAB,
    ARG_NONE_ID,
    ARG_PAD_ID,
    BOS_ID,
    EOS_ID,
    ID_TO_ACTION,
    ID_TO_ARG,
    PAD_ID,
    encode_object,
    encode_target,
    encode_task,
)


@dataclass(frozen=True)
class GeneratedPlan:
    """One model-generated action sequence and its normalized score."""

    actions: list[str]
    score: float
    terminated: bool
    truncated: bool
    invalid: bool = False


def _normalized_score(score: float, action_count: int) -> float:
    return score / max(action_count, 1)


@torch.no_grad()
def joint_beam_search(
    model: torch.nn.Module,
    *,
    source_id: int,
    verb_id: int,
    task_id: int,
    object_id: int,
    target_id: int,
    device: torch.device,
    beam_width: int = 3,
    max_actions: int = 12,
) -> GeneratedPlan:
    """Generate action/argument pairs without consulting a reference plan."""
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if max_actions < 1:
        raise ValueError("max_actions must be at least 1")

    conditions = tuple(
        torch.tensor([value], dtype=torch.long, device=device)
        for value in (source_id, verb_id, task_id, object_id, target_id)
    )
    beams: list[tuple[list[int], list[int], float, bool]] = [
        ([BOS_ID], [ARG_NONE_ID], 0.0, False)
    ]

    for _ in range(max_actions):
        candidates: list[tuple[list[int], list[int], float, bool]] = []
        for action_ids, argument_ids, score, finished in beams:
            if finished:
                candidates.append((action_ids, argument_ids, score, True))
                continue

            actions = torch.tensor([action_ids], dtype=torch.long, device=device)
            arguments = torch.tensor([argument_ids], dtype=torch.long, device=device)
            action_logits, argument_logits = model(*conditions, actions, arguments)
            action_log_probs = functional.log_softmax(action_logits[0, -1], dim=-1)
            argument_log_probs = functional.log_softmax(
                argument_logits[0, -1], dim=-1
            )
            action_log_probs[PAD_ID] = float("-inf")
            action_log_probs[BOS_ID] = float("-inf")
            argument_log_probs[ARG_PAD_ID] = float("-inf")
            argument_log_probs[ARG_NONE_ID] = float("-inf")

            action_candidates = torch.topk(
                action_log_probs, min(beam_width, action_log_probs.numel())
            ).indices.tolist()
            argument_candidates = torch.topk(
                argument_log_probs, min(beam_width, argument_log_probs.numel())
            ).indices.tolist()
            for action_id in action_candidates:
                action_score = score + action_log_probs[action_id].item()
                if action_id == EOS_ID:
                    candidates.append(
                        (
                            action_ids + [EOS_ID],
                            argument_ids + [ARG_NONE_ID],
                            action_score,
                            True,
                        )
                    )
                    continue
                for argument_id in argument_candidates:
                    candidates.append(
                        (
                            action_ids + [action_id],
                            argument_ids + [argument_id],
                            action_score + argument_log_probs[argument_id].item(),
                            False,
                        )
                    )

        if not candidates:
            raise ValueError("model generated no valid token candidates")
        beams = sorted(
            candidates,
            key=lambda beam: _normalized_score(beam[2], len(beam[0]) - 1),
            reverse=True,
        )[:beam_width]
        if all(beam[3] for beam in beams):
            break

    best_actions, best_arguments, score, terminated = max(
        beams,
        key=lambda beam: _normalized_score(beam[2], len(beam[0]) - 1),
    )
    decoded = [
        f"{ID_TO_ACTION[action_id]}({ID_TO_ARG[argument_id]})"
        for action_id, argument_id in zip(best_actions[1:], best_arguments[1:])
        if action_id != EOS_ID
    ]
    return GeneratedPlan(
        actions=decoded,
        score=_normalized_score(score, len(decoded)),
        terminated=terminated,
        truncated=not terminated,
        invalid=not decoded,
    )


def autoregressive_metrics(
    model: torch.nn.Module,
    raw_items: list[dict],
    device: torch.device,
    beam_width: int = 3,
    max_actions: int = 12,
) -> dict[str, float | int]:
    """Score strict exact matches from generated, never teacher-forced, plans."""
    from train_project_transformer_v2 import SOURCE_VOCAB, VERB_VOCAB

    total = len(raw_items)
    exact_matches = 0
    truncated_count = 0
    invalid_count = 0
    for item in raw_items:
        plan = joint_beam_search(
            model,
            source_id=SOURCE_VOCAB[item["source"]],
            verb_id=VERB_VOCAB[item["verb"]],
            task_id=encode_task(item["task"]),
            object_id=encode_object(item["object"]),
            target_id=encode_target(item["target"]),
            device=device,
            beam_width=beam_width,
            max_actions=max_actions,
        )
        truncated_count += int(plan.truncated)
        invalid_count += int(plan.invalid)
        exact_matches += int(
            plan.terminated
            and not plan.truncated
            and not plan.invalid
            and plan.actions == item["actions"]
        )
    return {
        "total": total,
        "sequence_accuracy": exact_matches / total if total else 0.0,
        "truncated_count": truncated_count,
        "invalid_count": invalid_count,
    }
