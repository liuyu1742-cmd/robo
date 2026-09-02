import unittest

import torch

from train_detailed_action_bge import (
    build_label_order,
    build_targets,
    calibrate_gate_threshold,
    build_training_text,
    build_catalog_anchor_text,
    load_training_splits,
    load_frozen_head_checkpoint,
    merge_dual_head_states,
    output_to_cpu,
    pool_training_hidden,
    shuffled_batch_indices,
)
from tools.detailed_action_bge_model import DetailedActionBgeClassifier


class DetailedActionBgeTrainingTest(unittest.TestCase):
    def test_build_targets_marks_rejections_for_gate_only(self):
        labels = ["object:a"]
        intent, gate = build_targets([
            {"label_id": "object:a", "expected_executable": True},
            {"label_id": None, "expected_executable": False},
        ], labels)
        self.assertEqual(intent.tolist(), [0, -100])
        self.assertEqual(gate.tolist(), [1, 0])

    def test_label_order_excludes_rejections_and_is_stable(self):
        order = build_label_order([
            {"label_id": "object:b", "expected_executable": True},
            {"label_id": None, "expected_executable": False},
            {"label_id": "object:a", "expected_executable": True},
        ])
        self.assertEqual(order, ["object:a", "object:b"])

    def test_training_loader_uses_train_and_dev_only(self):
        train, dev = load_training_splits("datasets/detailed_action_spoken_v2")
        self.assertEqual(len(train), 3080)
        self.assertEqual(len(dev), 942)
        self.assertTrue(all(item["annotation_source"].endswith("_train") for item in train))
        self.assertTrue(all(item["annotation_source"].endswith("_dev") for item in dev))

    def test_gate_threshold_calibration_uses_dev_scores(self):
        result = calibrate_gate_threshold(
            torch.tensor([-0.8, -0.2, 0.4, 0.9]), torch.tensor([0, 0, 1, 1])
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertGreaterEqual(result["threshold"], -0.2)

    def test_evidence_training_text_uses_only_annotation_evidence(self):
        record = {
            "text": "请把沾水的牙刷立放进漱口杯。",
            "object_mentions": ["牙刷", "漱口杯"],
            "operation_evidence": ["立放", "放进"],
        }
        self.assertEqual(build_training_text(record, "evidence"), "牙刷；漱口杯；立放；放进")

    def test_evidence_training_text_keeps_rejection_as_raw_text(self):
        record = {"text": "这件事先不要做。", "expected_executable": False}
        self.assertEqual(build_training_text(record, "evidence"), "这件事先不要做。")

    def test_merges_intent_and_gate_heads_from_their_own_best_epochs(self):
        intent_state = {
            "intent_head.weight": torch.tensor([[1.0]]),
            "intent_head.bias": torch.tensor([2.0]),
            "gate_head.weight": torch.tensor([[3.0]]),
            "gate_head.bias": torch.tensor([4.0]),
        }
        gate_state = {
            "intent_head.weight": torch.tensor([[5.0]]),
            "intent_head.bias": torch.tensor([6.0]),
            "gate_head.weight": torch.tensor([[7.0]]),
            "gate_head.bias": torch.tensor([8.0]),
        }
        merged = merge_dual_head_states(intent_state, gate_state)
        self.assertEqual(merged["intent_head.weight"].item(), 1.0)
        self.assertEqual(merged["gate_head.weight"].item(), 7.0)

    def test_moves_both_dual_head_outputs_to_cpu(self):
        output = output_to_cpu({"intent_logits": torch.ones(1), "gate_logits": torch.zeros(1)})
        self.assertEqual(set(output), {"intent_logits", "gate_logits"})
        self.assertEqual(output["intent_logits"].device.type, "cpu")

    def test_resume_loads_frozen_heads_only_when_label_order_matches(self):
        source = DetailedActionBgeClassifier(hidden_size=2, label_count=2)
        with torch.no_grad():
            source.intent_head.bias.fill_(3.0)
            source.gate_head.bias.fill_(5.0)
        resumed = DetailedActionBgeClassifier(hidden_size=2, label_count=2)
        load_frozen_head_checkpoint(resumed, {"labels": ["object:a", "object:b"], "state_dict": source.state_dict()}, ["object:a", "object:b"])
        self.assertTrue(torch.equal(resumed.intent_head.bias, source.intent_head.bias))
        self.assertTrue(torch.equal(resumed.gate_head.bias, source.gate_head.bias))

    def test_stage_two_pooling_honors_checkpoint_pooling_mode(self):
        hidden = torch.tensor([[[9.0, 8.0], [1.0, 2.0]]])
        mask = torch.tensor([[1, 1]])
        self.assertTrue(torch.equal(pool_training_hidden(hidden, mask, "cls"), torch.tensor([[9.0, 8.0]])))

    def test_catalog_anchor_text_preserves_the_label_definition(self):
        self.assertEqual(
            build_catalog_anchor_text({"entry_type": "object", "object_name_zh": "牙刷", "detailed_actions": ["立放到漱口杯"]}),
            "牙刷；立放到漱口杯",
        )

    def test_stage_two_batches_are_shuffled_and_complete(self):
        batches = shuffled_batch_indices(10, 4, seed=7)
        flattened = torch.cat(batches)
        self.assertEqual(sorted(flattened.tolist()), list(range(10)))
        self.assertNotEqual(flattened.tolist(), list(range(10)))


if __name__ == "__main__":
    unittest.main()
