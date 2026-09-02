from pathlib import Path

import torch

from tools.instruction_action_model import ActionTransformer
from tools.encode import (
    ACTION_VOCAB,
    ARG_VOCAB,
    OBJECT_VOCAB,
    TARGET_VOCAB,
    TASK_VOCAB,
)


def load_model(checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train_transformer.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for key, current_vocab in (
        ("task_vocab", TASK_VOCAB),
        ("object_vocab", OBJECT_VOCAB),
        ("action_vocab", ACTION_VOCAB),
        ("argument_vocab", ARG_VOCAB),
        ("target_vocab", TARGET_VOCAB),
    ):
        saved_vocab = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if saved_vocab is not None and saved_vocab != current_vocab:
            raise ValueError(
                f"Checkpoint {key} does not match the current encoder. Retrain the model."
            )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model = ActionTransformer(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
