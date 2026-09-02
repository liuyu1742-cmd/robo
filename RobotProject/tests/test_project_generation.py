import unittest

import torch
import torch.nn as nn

from tools.project_encode import ACTION_VOCAB, ARG_NONE_ID, ARG_VOCAB


class ScriptedModel(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def forward(
        self,
        source,
        verb,
        task,
        obj,
        target,
        actions,
        arguments,
        visual=None,
        visual_present=None,
    ):
        del source, verb, task, obj, target, arguments, visual, visual_present
        batch_size, length = actions.shape
        action_logits = torch.full(
            (batch_size, length, len(ACTION_VOCAB)), -100.0
        )
        argument_logits = torch.full(
            (batch_size, length, len(ARG_VOCAB)), -100.0
        )
        action, argument = self.steps[min(length - 1, len(self.steps) - 1)]
        action_logits[:, -1, ACTION_VOCAB[action]] = 10.0
        argument_logits[:, -1, ARG_VOCAB[argument]] = 10.0
        return action_logits, argument_logits


class ProjectGenerationTest(unittest.TestCase):
    def test_joint_beam_search_decodes_model_action_and_argument(self):
        from tools.project_generation import joint_beam_search

        result = joint_beam_search(
            ScriptedModel([("locate", "cup"), ("<eos>", "<none>")]),
            source_id=0,
            verb_id=0,
            task_id=0,
            object_id=0,
            target_id=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(result.actions, ["locate(cup)"])
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)

    def test_joint_beam_search_marks_missing_eos_as_truncated(self):
        from tools.project_generation import joint_beam_search

        result = joint_beam_search(
            ScriptedModel([("locate", "cup")] * 4),
            source_id=0,
            verb_id=0,
            task_id=0,
            object_id=0,
            target_id=0,
            device=torch.device("cpu"),
            max_actions=2,
        )

        self.assertEqual(result.actions, ["locate(cup)", "locate(cup)"])
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)

    def test_joint_beam_search_marks_empty_plan_as_invalid(self):
        from tools.project_generation import joint_beam_search

        result = joint_beam_search(
            ScriptedModel([("<eos>", "<none>")]),
            source_id=0,
            verb_id=0,
            task_id=0,
            object_id=0,
            target_id=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(result.actions, [])
        self.assertTrue(result.terminated)
        self.assertTrue(result.invalid)

    def test_autoregressive_metrics_require_exact_complete_plan(self):
        from tools.project_generation import autoregressive_metrics

        raw_items = [
            {
                "source": "epic_kitchens_100",
                "verb": "wash",
                "task": "cleaning",
                "object": "cup",
                "target": "sink",
                "actions": ["locate(cup)"],
            }
        ]
        metrics = autoregressive_metrics(
            ScriptedModel([("locate", "cup"), ("<eos>", "<none>")]),
            raw_items,
            torch.device("cpu"),
        )

        self.assertEqual(metrics["sequence_accuracy"], 1.0)
        self.assertEqual(metrics["truncated_count"], 0)

    def test_autoregressive_metrics_counts_empty_plan_without_crashing(self):
        from tools.project_generation import autoregressive_metrics

        raw_items = [
            {
                "source": "epic_kitchens_100",
                "verb": "wash",
                "task": "cleaning",
                "object": "cup",
                "target": "sink",
                "actions": ["locate(cup)"],
            }
        ]
        metrics = autoregressive_metrics(
            ScriptedModel([("<eos>", "<none>")]), raw_items, torch.device("cpu")
        )

        self.assertEqual(metrics["sequence_accuracy"], 0.0)
        self.assertEqual(metrics["invalid_count"], 1)

    def test_evaluator_counts_truncated_generation_as_failure(self):
        from evaluate_project_model_v2 import summarize_generated
        from tools.project_generation import GeneratedPlan

        metrics = summarize_generated(
            [
                {
                    "id": "case-1",
                    "source": "epic_kitchens_100",
                    "verb": "wash",
                    "task": "cleaning",
                    "object": "cup",
                    "target": "none",
                    "actions": ["locate(cup)"],
                }
            ],
            [
                GeneratedPlan(
                    actions=["locate(cup)"],
                    score=-2.0,
                    terminated=False,
                    truncated=True,
                )
            ],
        )

        self.assertEqual(metrics["sequence_accuracy"], 0.0)
        self.assertEqual(metrics["truncated_count"], 1)


if __name__ == "__main__":
    unittest.main()
