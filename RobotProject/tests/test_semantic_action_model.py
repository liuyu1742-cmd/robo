import unittest
import tempfile
from pathlib import Path

import torch

from tools.semantic_action_model import (
    CharacterNgramRelationClassifier,
    build_vocabulary,
    encode_text,
    load_checkpoint,
    make_checkpoint,
    predict_relations,
    vectorize_batch,
)


class SemanticActionModelTest(unittest.TestCase):
    def test_character_ngram_encoding_is_deterministic(self):
        vocab = build_vocabulary(["帮我关阅读灯", "请关闭电视"])
        self.assertEqual(encode_text("帮我关阅读灯", vocab), encode_text("帮我关阅读灯", vocab))
        self.assertTrue(encode_text("帮我关阅读灯", vocab))

    def test_vectorize_batch_preserves_example_boundaries(self):
        vocab = build_vocabulary(["阅读灯", "电视"])
        values, offsets = vectorize_batch(["阅读灯", "电视"], vocab)
        self.assertEqual(offsets.tolist(), [0, len(encode_text("阅读灯", vocab))])
        self.assertGreater(values.numel(), 0)

    def test_v1_checkpoint_fixture_loads_and_predicts_through_public_api(self):
        vocabulary = build_vocabulary(["打开台灯", "关闭电视"])
        labels = ["lamp:on", "television:off"]
        source_model = CharacterNgramRelationClassifier(
            len(vocabulary), len(labels), embedding_dim=8
        )
        checkpoint = make_checkpoint(
            source_model, vocabulary, labels, embedding_dim=8
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controlled_v1.pt"
            torch.save(checkpoint, path)
            model, loaded_vocabulary, loaded_labels = load_checkpoint(
                str(path), device="cpu"
            )
            predictions = predict_relations(
                model,
                loaded_vocabulary,
                loaded_labels,
                ["打开台灯"],
                device="cpu",
            )

        self.assertIsInstance(model, CharacterNgramRelationClassifier)
        self.assertEqual(loaded_vocabulary, vocabulary)
        self.assertEqual(loaded_labels, labels)
        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0]["relation_key"], labels)
        self.assertGreaterEqual(predictions[0]["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
