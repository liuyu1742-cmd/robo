import unittest
import tempfile
from pathlib import Path

import torch

from tools.detailed_action_bge_model import (
    cls_pool_embeddings,
    DetailedActionBgeClassifier,
    dual_head_loss,
    encode_bge_texts,
    load_local_bge_encoder,
    mean_pool_embeddings,
    supervised_contrastive_loss,
    rerank_topk_scores,
    rerank_finite_scores,
    load_character_gate,
    combine_scene_object_logits,
    combine_scene_object_logits_conditionally,
    route_scene_object_logits,
    apply_scene_rescue,
    apply_object_surface_override,
    apply_scene_text_rescue,
    apply_object_text_rescue,
    apply_object_operation_specialist_gate,
    load_object_operation_specialist,
    apply_short_object_margin_gate,
)


class DetailedActionBgeModelTest(unittest.TestCase):
    def test_encode_bge_texts_returns_one_vector_per_text(self):
        root = Path(__file__).resolve().parents[1]
        tokenizer, encoder = load_local_bge_encoder(root / "models" / "pretrained" / "BAAI_bge-small-zh-v1.5")
        values = encode_bge_texts(tokenizer, encoder, ["开灯", "关窗"], device="cpu")
        self.assertEqual(tuple(values.shape), (2, 512))
        self.assertTrue(torch.isfinite(values).all())

    def test_dual_head_loss_is_finite_for_mixed_batch(self):
        output = {
            "intent_logits": torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
            "gate_logits": torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        }
        loss = dual_head_loss(output, torch.tensor([0, 1]), torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_real_local_encoder_loads_without_network(self):
        root = Path(__file__).resolve().parents[1]
        tokenizer, encoder = load_local_bge_encoder(
            root / "models" / "pretrained" / "BAAI_bge-small-zh-v1.5",
            device="cpu",
        )
        encoded = tokenizer(["离线测试"], return_tensors="pt")
        with torch.no_grad():
            states = encoder(**encoded).last_hidden_state
        self.assertEqual(states.shape[0], 1)
        self.assertEqual(states.shape[-1], 512)

    def test_loader_rejects_missing_local_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_local_bge_encoder(Path(directory) / "missing", device="cpu")

    def test_masked_mean_pooling_ignores_padding(self):
        states = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
        pooled = mean_pool_embeddings(states, torch.tensor([[1, 1, 0]]))
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 4.0]])))

    def test_cls_pooling_uses_first_token(self):
        states = torch.tensor([[[9.0, 8.0], [1.0, 2.0]]])
        self.assertTrue(torch.equal(cls_pool_embeddings(states), torch.tensor([[9.0, 8.0]])))

    def test_forward_returns_intent_and_gate_logits(self):
        model = DetailedActionBgeClassifier(hidden_size=8, label_count=3)
        output = model(torch.ones((2, 8)))
        self.assertEqual(tuple(output["intent_logits"].shape), (2, 3))
        self.assertEqual(tuple(output["gate_logits"].shape), (2, 2))

    def test_supervised_contrastive_loss_rewards_same_label_pairs(self):
        labels = torch.tensor([0, 0, 1, 1])
        aligned = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        crossed = torch.tensor([[1.0, 0.0], [0.1, 0.9], [0.0, 1.0], [0.9, 0.1]])
        self.assertLess(supervised_contrastive_loss(aligned, labels), supervised_contrastive_loss(crossed, labels))

    def test_rerank_changes_only_candidates_inside_top_k(self):
        logits = torch.tensor([[4.0, 3.0, 2.0]])
        lexical = torch.tensor([[0.0, 1.0, 100.0]])
        ranked = rerank_topk_scores(logits, lexical, top_k=2, weight=2.0)
        self.assertEqual(ranked.argmax(1).item(), 1)
        self.assertTrue(torch.isneginf(ranked[0, 2]))

    def test_finite_rerank_preserves_masked_candidates(self):
        logits = torch.tensor([[4.0, 3.0, float("-inf")]])
        auxiliary = torch.tensor([[0.0, 1.0, 100.0]])
        ranked = rerank_finite_scores(logits, auxiliary, weight=2.0)
        self.assertEqual(ranked.argmax(1).item(), 1)
        self.assertTrue(torch.isneginf(ranked[0, 2]))

    def test_load_character_gate_restores_embedded_gate(self):
        from tools.semantic_action_model import CharacterNgramRelationClassifier
        source = CharacterNgramRelationClassifier(3, 2, 4)
        payload = {"character_gate": {"vocabulary": {"<unk>": 0, "c:开": 1, "c:关": 2}, "embedding_dim": 4, "threshold": 0.25, "state_dict": source.state_dict()}}
        model, vocabulary, threshold = load_character_gate(payload)
        self.assertEqual(len(model.embedding.weight), 3)
        self.assertEqual(vocabulary["c:开"], 1)
        self.assertEqual(threshold, 0.25)

    def test_combine_scene_object_logits_uses_dedicated_scene_head(self):
        primary = torch.tensor([[1.0, 9.0]])
        scene = torch.tensor([[8.0, 2.0]])
        combined = combine_scene_object_logits(primary, scene, torch.tensor([True, False]), scene_bias=-1.0)
        self.assertTrue(torch.equal(combined, torch.tensor([[7.0, 9.0]])))

    def test_conditional_scene_fusion_keeps_strong_primary_scene_rows(self):
        primary = torch.tensor([[8.0, 2.0], [4.0, 5.0]])
        specialist = torch.tensor([[3.0, 2.0], [9.0, 5.0]])
        result = combine_scene_object_logits_conditionally(
            primary,
            specialist,
            torch.tensor([True, False]),
            scene_bias=-1.0,
            primary_scene_override_margin=2.0,
        )
        self.assertTrue(torch.equal(result, torch.tensor([[8.0, 2.0], [8.0, 5.0]])))

    def test_scene_type_router_masks_the_opposite_label_group(self):
        primary = torch.tensor([[2.0, 8.0], [2.0, 8.0]])
        scene = torch.tensor([[9.0, 1.0], [9.0, 1.0]])
        routed = route_scene_object_logits(
            primary,
            scene,
            torch.tensor([True, False]),
            scene_type_scores=torch.tensor([2.0, -2.0]),
            threshold=0.0,
        )
        self.assertEqual(routed.argmax(1).tolist(), [0, 1])
        self.assertTrue(torch.isneginf(routed[0, 1]))
        self.assertTrue(torch.isneginf(routed[1, 0]))

    def test_scene_rescue_only_replaces_high_confidence_object_rows(self):
        final = torch.tensor([[1.0, 8.0], [1.0, 8.0], [8.0, 1.0]])
        primary = torch.tensor([[2.0, 7.0], [2.0, 7.0], [7.0, 2.0]])
        scene = torch.tensor([[9.0, 0.0], [7.5, 0.0], [9.0, 0.0]])
        rescued = apply_scene_rescue(
            final,
            primary,
            scene,
            torch.tensor([True, False]),
            threshold=1.0,
        )
        self.assertEqual(rescued.argmax(1).tolist(), [0, 1, 0])

    def test_scene_rescue_can_use_lexical_gap(self):
        final = torch.tensor([[1.0, 8.0], [1.0, 8.0]])
        primary = torch.tensor([[2.0, 7.0], [2.0, 7.0]])
        scene = torch.tensor([[9.0, 0.0], [9.0, 0.0]])
        lexical = torch.tensor([[0.8, 0.2], [0.4, 0.3]])
        rescued = apply_scene_rescue(
            final, primary, scene, torch.tensor([True, False]),
            threshold=0.2, method="lexical_gap", auxiliary_scores=lexical,
        )
        self.assertEqual(rescued.argmax(1).tolist(), [0, 1])

    def test_object_surface_override_requires_primary_object_prediction(self):
        final = torch.tensor([[8.0, 1.0, 2.0], [8.0, 1.0, 2.0]])
        primary = torch.tensor([[1.0, 3.0, 7.0], [9.0, 3.0, 7.0]])
        candidates = torch.tensor([[False, True, False], [False, True, False]])
        result = apply_object_surface_override(
            final, primary, torch.tensor([True, False, False]), candidates
        )
        self.assertEqual(result.argmax(1).tolist(), [1, 0])

    def test_object_surface_override_preserves_strong_scene_specialist_rows(self):
        final = torch.tensor([[8.0, 1.0, 2.0], [8.0, 1.0, 2.0]])
        primary = torch.tensor([[1.0, 3.0, 7.0], [1.0, 3.0, 7.0]])
        specialist = torch.tensor([[10.0, 0.0, 0.0], [6.5, 0.0, 0.0]])
        candidates = torch.tensor([[False, True, False], [False, True, False]])
        result = apply_object_surface_override(
            final,
            primary,
            torch.tensor([True, False, False]),
            candidates,
            scene_guard_logits=specialist,
            scene_guard_margin=1.0,
        )
        self.assertEqual(result.argmax(1).tolist(), [0, 1])

    def test_scene_text_rescue_only_routes_matching_object_predictions(self):
        final = torch.tensor([[1.0, 8.0, 2.0], [1.0, 8.0, 2.0], [9.0, 1.0, 2.0]])
        specialist = torch.tensor([[7.0, 0.0, 1.0], [6.0, 0.0, 1.0], [8.0, 0.0, 1.0]])
        result = apply_scene_text_rescue(
            final,
            specialist,
            torch.tensor([True, False, False]),
            ["把厨房用具备齐", "把杯子洗净", "把厨房用具备齐"],
            [r"厨房用具.{0,4}备齐"],
        )
        self.assertEqual(result.argmax(1).tolist(), [0, 1, 0])

    def test_object_text_rescue_only_routes_matching_scene_predictions(self):
        final = torch.tensor([[8.0, 1.0, 2.0], [8.0, 1.0, 2.0], [1.0, 8.0, 2.0]])
        primary = torch.tensor([[9.0, 4.0, 7.0], [9.0, 6.0, 5.0], [9.0, 6.0, 5.0]])
        candidates = torch.tensor([[False, False, True], [False, True, False], [False, True, False]])
        result = apply_object_text_rescue(
            final,
            primary,
            torch.tensor([True, False, False]),
            candidates,
            ["床垫归位", "普通场景", "床垫归位"],
            [r"床垫.{0,4}归位"],
        )
        self.assertEqual(result.argmax(1).tolist(), [2, 0, 1])

    def test_object_operation_specialist_vetoes_only_supported_cases(self):
        predictions = apply_object_operation_specialist_gate(
            base_scores=torch.tensor([1.0, 3.0, 1.0, 1.0, 1.0, 3.0]),
            base_threshold=0.0,
            character_scores=torch.tensor([-2.0, -2.0, 2.0, 2.0, 2.0, 2.0]),
            bge_scores=torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
            own_capability_scores=torch.tensor([0.40, 0.50, 0.60, 0.40, -1.0, -1.0]),
            global_capability_scores=torch.tensor([0.55, 0.55, 0.65, 0.55, 0.60, 0.60]),
            single_object_mask=torch.tensor([True, True, True, False, True, True]),
            known_object_mask=torch.tensor([True, True, True, True, False, False]),
            config={
                "general_margin": 2.0,
                "character_mean": 0.0,
                "character_std": 1.0,
                "bge_mean": 0.0,
                "bge_std": 1.0,
                "bge_weight": 0.5,
                "fusion_threshold": -2.4,
                "minimum_capability_gap": 0.10,
                "maximum_own_capability": 0.45,
                "unknown_object_margin": 2.0,
            },
        )
        self.assertEqual(predictions.tolist(), [False, True, True, True, False, True])

        tensor_threshold_predictions = apply_object_operation_specialist_gate(
            base_scores=torch.tensor([1.0, 3.0]),
            base_threshold=torch.tensor([0.0, 2.0]),
            character_scores=torch.tensor([2.0, 2.0]),
            bge_scores=torch.tensor([1.0, 1.0]),
            own_capability_scores=torch.tensor([-1.0, -1.0]),
            global_capability_scores=torch.tensor([0.6, 0.6]),
            single_object_mask=torch.tensor([False, False]),
            known_object_mask=torch.tensor([False, False]),
            config={
                "general_margin": 2.0,
                "character_mean": 0.0,
                "character_std": 1.0,
                "bge_mean": 0.0,
                "bge_std": 1.0,
                "bge_weight": 0.5,
                "fusion_threshold": -2.4,
                "minimum_capability_gap": 0.10,
                "maximum_own_capability": 0.45,
                "unknown_object_margin": 0.0,
            },
        )
        self.assertEqual(tensor_threshold_predictions.tolist(), [True, True])

    def test_load_object_operation_specialist_restores_embedded_models(self):
        from tools.semantic_action_model import CharacterNgramRelationClassifier

        character = CharacterNgramRelationClassifier(2, 2, 4)
        bge_head = torch.nn.Linear(8, 2)
        payload = {
            "object_operation_specialist": {
                "config": {"general_margin": 1.0},
                "character": {
                    "vocabulary": {"<unk>": 0, "c:开": 1},
                    "embedding_dim": 4,
                    "state_dict": character.state_dict(),
                },
                "bge_head_state": bge_head.state_dict(),
                "capability_labels": ["object:a"],
                "capability_vectors": torch.ones((1, 8)),
            }
        }
        restored = load_object_operation_specialist(payload)
        self.assertEqual(restored[1]["c:开"], 1)
        self.assertEqual(restored[2].in_features, 8)
        self.assertEqual(restored[3], ["object:a"])
        self.assertEqual(tuple(restored[4].shape), (1, 8))

    def test_short_object_margin_gate_rejects_only_known_short_borderline_commands(self):
        result = apply_short_object_margin_gate(
            predictions=torch.tensor([True, True, True, True, False]),
            scores=torch.tensor([-8.2, -7.0, -8.2, -8.2, -8.2]),
            threshold=-9.0,
            text_lengths=torch.tensor([12, 12, 20, 12, 12]),
            known_object_mask=torch.tensor([True, True, True, False, True]),
            config={"maximum_chars": 15, "additional_margin": 1.0},
        )
        self.assertEqual(result.tolist(), [False, True, True, True, False])


if __name__ == "__main__":
    unittest.main()
