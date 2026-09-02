import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.manual_instruction_entry import ManualInstruction
from tools.project_encode import (
    ACTION_VOCAB,
    ARG_VOCAB,
    OBJECT_VOCAB,
    TARGET_VOCAB,
    TASK_VOCAB,
)


def minimal_config():
    from train_project_transformer_v2 import SOURCE_VOCAB, VERB_VOCAB

    return {
        "source_size": len(SOURCE_VOCAB),
        "verb_size": len(VERB_VOCAB),
        "task_size": len(TASK_VOCAB),
        "obj_size": len(OBJECT_VOCAB),
        "target_size": len(TARGET_VOCAB),
        "vocab_size": len(ACTION_VOCAB),
        "arg_vocab_size": len(ARG_VOCAB),
        "d_model": 96,
        "visual_dim": 0,
    }


class ProjectRuntimeTest(unittest.TestCase):
    def test_fold_operation_uses_checkpoint_fold_verb(self):
        from tools.project_conditions import OPERATION_TO_VERB
        from train_project_transformer_v2 import VERB_VOCAB

        self.assertEqual(OPERATION_TO_VERB["fold"], "fold")
        self.assertIn("fold", VERB_VOCAB)

    def test_every_runtime_operation_verb_is_available_to_the_transformer(self):
        from tools.project_conditions import OPERATION_TO_VERB
        from train_project_transformer_v2 import VERB_VOCAB

        self.assertTrue(set(OPERATION_TO_VERB.values()).issubset(VERB_VOCAB))

    def test_loader_rejects_stale_object_vocab_before_loading_weights(self):
        from tools.project_runtime import load_project_v2_model

        checkpoint = {
            "model_config": minimal_config(),
            "model_state_dict": {},
            "source_vocab": __import__("train_project_transformer_v2").SOURCE_VOCAB,
            "verb_vocab": __import__("train_project_transformer_v2").VERB_VOCAB,
            "task_vocab": TASK_VOCAB,
            "object_vocab": {"stale": 0},
            "action_vocab": ACTION_VOCAB,
            "argument_vocab": ARG_VOCAB,
            "target_vocab": TARGET_VOCAB,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "object_vocab.*Retrain"):
                load_project_v2_model(path, "cpu")

    def test_generation_uses_manual_source_with_operation_mapped_verb(self):
        import tools.project_runtime as runtime
        from tools.project_generation import GeneratedPlan
        from train_project_transformer_v2 import SOURCE_VOCAB

        resolved = ManualInstruction(
            normalized_instruction="帮我照顾花",
            task="maintenance_management",
            object="flower_pot",
            target="flower_pot",
            operation="care_plant",
            used_fallback=False,
            note="test",
        )
        calls = []

        def fake_search(*args, **kwargs):
            del args
            calls.append(kwargs)
            return GeneratedPlan(
                actions=["inspect(flower_pot)"],
                score=float(kwargs["source_id"]),
                terminated=True,
                truncated=False,
            )

        with patch.object(runtime, "joint_beam_search", fake_search):
            result = runtime.generate_for_instruction(object(), resolved, device="cpu")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["source_id"], SOURCE_VOCAB["manual_instruction"])
        self.assertEqual(result.verb, "repair")
        self.assertEqual(result.source, "manual_instruction")
        self.assertEqual(result.actions, ["inspect(flower_pot)"])


if __name__ == "__main__":
    unittest.main()
