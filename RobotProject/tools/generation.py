import torch
import torch.nn.functional as F

from tools.encode import (
    ACTION_VOCAB,
    ARG_VOCAB,
    ARG_NONE_ID,
    ARG_PAD_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    OBJECT_VOCAB,
    TARGET_VOCAB,
)

OBJECT_ID_TO_ARG_ID = {
    object_id: ARG_VOCAB[name] for name, object_id in OBJECT_VOCAB.items()
}
TARGET_ID_TO_ARG_ID = {
    target_id: ARG_VOCAB.get(name, ARG_NONE_ID)
    for name, target_id in TARGET_VOCAB.items()
}
MOVE_ID = ACTION_VOCAB["move"]


def _normalized_score(score, token_count, length_penalty):
    return score / (((5.0 + token_count) / 6.0) ** length_penalty)


@torch.no_grad()
def beam_search(model, task_id, object_id, target_id, beam_width=3,
                max_actions=10, length_penalty=0.6, device="cpu"):
    """Generate [(action_id, argument_id), ...] without BOS/EOS."""
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    task = torch.tensor([task_id], dtype=torch.long, device=device)
    obj = torch.tensor([object_id], dtype=torch.long, device=device)
    target = torch.tensor([target_id], dtype=torch.long, device=device)
    # verb ids, argument ids, cumulative joint log probability, finished
    beams = [([BOS_ID], [ARG_NONE_ID], 0.0, False)]

    for _ in range(max_actions + 1):
        candidates = []
        for verbs, arguments, score, finished in beams:
            if finished:
                candidates.append((verbs, arguments, score, True))
                continue

            verb_input = torch.tensor([verbs], dtype=torch.long, device=device)
            arg_input = torch.tensor([arguments], dtype=torch.long, device=device)
            verb_logits, arg_logits = model(
                task, obj, target, verb_input, arg_input
            )
            verb_log_probs = F.log_softmax(verb_logits[0, -1], dim=-1)
            arg_log_probs = F.log_softmax(arg_logits[0, -1], dim=-1)
            verb_log_probs[PAD_ID] = float("-inf")
            verb_log_probs[BOS_ID] = float("-inf")
            arg_log_probs[ARG_PAD_ID] = float("-inf")

            verb_scores, verb_ids = torch.topk(verb_log_probs, k=beam_width)
            for verb_score, verb_id in zip(verb_scores.tolist(), verb_ids.tolist()):
                if verb_id == EOS_ID:
                    candidates.append(
                        (verbs + [EOS_ID], arguments + [ARG_NONE_ID],
                         score + verb_score, True)
                    )
                    continue

                # Robot action schemas provide hard grounding constraints:
                # move binds to the goal; manipulation/perception binds to object.
                arg_id = (
                    TARGET_ID_TO_ARG_ID[target_id]
                    if verb_id == MOVE_ID
                    else OBJECT_ID_TO_ARG_ID[object_id]
                )
                candidates.append(
                    (verbs + [verb_id], arguments + [arg_id],
                     score + verb_score, False)
                )

        beams = sorted(
            candidates,
            key=lambda beam: _normalized_score(
                beam[2], len(beam[0]) - 1, length_penalty
            ),
            reverse=True,
        )[:beam_width]
        if all(beam[3] for beam in beams):
            break

    best = max(
        beams,
        key=lambda beam: _normalized_score(
            beam[2], len(beam[0]) - 1, length_penalty
        ),
    )
    return [
        (verb, argument)
        for verb, argument in zip(best[0][1:], best[1][1:])
        if verb != EOS_ID
    ]
