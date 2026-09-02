import json
import re
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import torch

import train_detailed_action_semantic_parser as training
from train_detailed_action_semantic_parser import parse_args
from tools.detailed_action_catalog import CATALOGUE_PATH, load_detailed_action_catalog
from tools.detailed_action_semantic_data import (
    build_detailed_action_examples,
    catalog_sha256,
    normalize_semantic_text,
    split_detailed_action_examples,
)
from tools.detailed_action_semantic_evaluation import evaluate_semantic_checkpoint
from tools.semantic_action_model import load_checkpoint
from tools.semantic_text_normalization import (
    normalize_for_semantic_model,
    normalization_resource_terms,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "datasets" / "detailed_action_semantic_train.json"
DEV_PATH = ROOT / "datasets" / "detailed_action_semantic_dev.json"
CHECKPOINT_PATH = ROOT / "models" / "detailed_action_semantic" / "best.pt"


class DetailedActionSemanticDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_detailed_action_catalog()
        cls.labels = {entry["label"] for entry in cls.catalog["entries"]}
        cls.examples = build_detailed_action_examples(cls.catalog)

    def test_examples_cover_all_labels_with_multiple_unique_phrasings(self):
        by_label = Counter(row["label_id"] for row in self.examples)

        self.assertEqual(set(by_label), self.labels)
        self.assertEqual(len(self.labels), 254)
        self.assertTrue(all(count >= 4 for count in by_label.values()))
        self.assertTrue(
            all(
                set(row) == {"text", "label_id", "source"}
                and isinstance(row["text"], str)
                and row["text"].strip()
                and isinstance(row["source"], str)
                and row["source"].strip()
                for row in self.examples
            )
        )

    def test_normalized_phrases_never_conflict_across_labels(self):
        labels_by_text = defaultdict(set)
        for row in self.examples:
            labels_by_text[normalize_semantic_text(row["text"])].add(row["label_id"])

        conflicts = {
            text: labels for text, labels in labels_by_text.items() if len(labels) > 1
        }
        self.assertEqual(conflicts, {})
        self.assertEqual(len(labels_by_text), len(self.examples))

    def test_label_free_normalizer_shares_concepts_without_answer_leakage(self):
        self.assertEqual(
            normalize_for_semantic_model("擦净煎锅"),
            normalize_for_semantic_model("清洁平底锅"),
        )
        self.assertEqual(
            normalize_for_semantic_model("把台灯布置好"),
            normalize_for_semantic_model("将阅读灯摆放妥当"),
        )
        resources = normalization_resource_terms()
        dev_texts = {
            normalize_semantic_text(row["text"])
            for row in split_detailed_action_examples(self.examples)[1]
        }
        self.assertFalse(
            [term for term in resources if any(label in term for label in self.labels)]
        )
        self.assertFalse(
            [term for term in resources if normalize_semantic_text(term) in dev_texts]
        )
        self.assertFalse([term for term in resources if term.startswith(("MDC-", "object:"))])

    def test_variants_are_not_politeness_wrappers_around_one_expression_skeleton(self):
        polite_prefix = re.compile(
            r"^(?:机器人|助手)?[，,:：]?(?:请按这个安排|请按要求|请|麻烦|帮我|现在)?[，,:：]?"
        )
        polite_suffix = re.compile(
            r"[，,]?(?:麻烦安排一下|现在开始吧|辛苦你了)$"
        )
        rows_by_label = defaultdict(list)
        for row in self.examples:
            rows_by_label[row["label_id"]].append(row)

        for label, rows in rows_by_label.items():
            stripped_skeletons = {
                normalize_semantic_text(
                    polite_suffix.sub("", polite_prefix.sub("", row["text"]))
                )
                for row in rows
            }
            with self.subTest(label=label):
                self.assertGreaterEqual(len({row["source"] for row in rows}), 4)
                self.assertGreaterEqual(len(stripped_skeletons), 4)

    def test_scene_paraphrases_do_not_contain_mechanical_clause_splices(self):
        forbidden_splices = (
            "开始地面该",
            "完成地面该",
            "开始垃圾该",
            "完成垃圾该",
            "开始家里该",
            "完成家里该",
            "完成准备好",
            "再开始开始",
            "一起完成把",
            "开始就寝环境",
            "开始学习环境",
            "完成就寝环境",
            "完成学习环境",
        )
        scene_rows = [row for row in self.examples if row["label_id"].startswith("scene:")]

        self.assertFalse(
            [
                row["text"]
                for row in scene_rows
                if any(fragment in row["text"] for fragment in forbidden_splices)
            ]
        )

    def test_object_examples_name_an_entity_alias_and_express_task_semantics(self):
        objects = {
            entry["label"]: entry
            for entry in self.catalog["entries"]
            if entry["entry_type"] == "object"
        }
        for row in self.examples:
            entry = objects.get(row["label_id"])
            if entry is None:
                continue
            with self.subTest(label=row["label_id"], text=row["text"]):
                self.assertGreaterEqual(len(normalize_semantic_text(row["text"])), 4)
                self.assertNotEqual(normalize_semantic_text(row["text"]), "这个目标物")

    def test_hash_split_is_stable_leak_free_and_keeps_every_training_label(self):
        train_a, dev_a = split_detailed_action_examples(self.examples)
        train_b, dev_b = split_detailed_action_examples(list(reversed(self.examples)))

        self.assertEqual(train_a, train_b)
        self.assertEqual(dev_a, dev_b)
        self.assertEqual({row["label_id"] for row in train_a}, self.labels)
        train_texts = {normalize_semantic_text(row["text"]) for row in train_a}
        dev_texts = {normalize_semantic_text(row["text"]) for row in dev_a}
        self.assertTrue(dev_a)
        self.assertTrue(train_texts.isdisjoint(dev_texts))
        self.assertEqual(len(train_a) + len(dev_a), len(self.examples))

    def test_dev_uses_expression_skeletons_absent_from_all_training_rows(self):
        train, dev = split_detailed_action_examples(self.examples)
        train_skeletons = {row["source"] for row in train}
        dev_skeletons = {row["source"] for row in dev}

        self.assertTrue(train_skeletons)
        self.assertTrue(dev_skeletons)
        self.assertTrue(train_skeletons.isdisjoint(dev_skeletons))

    def test_dev_has_content_level_slot_and_long_ngram_isolation(self):
        train, dev = split_detailed_action_examples(self.examples)
        train_by_label = defaultdict(list)
        for row in train:
            train_by_label[row["label_id"]].append(row)
        entries = {entry["label"]: entry for entry in self.catalog["entries"]}

        def ngrams(text, size=6):
            normalized = normalize_semantic_text(text)
            return {
                normalized[index : index + size]
                for index in range(max(0, len(normalized) - size + 1))
            }

        for row in dev:
            entry = entries[row["label_id"]]
            allowed_short_names = []
            if entry["entry_type"] == "object":
                formal_name = entry["object_name_zh"]
                if len(normalize_semantic_text(formal_name)) <= 2:
                    allowed_short_names.append(formal_name)
                else:
                    self.assertNotIn(
                        normalize_semantic_text(formal_name),
                        normalize_semantic_text(row["text"]),
                        msg=f"dev reuses formal object name: {row}",
                    )
            else:
                command = normalize_semantic_text(entry["command"])
                if len(command) >= 4:
                    self.assertNotIn(
                        command,
                        normalize_semantic_text(row["text"]),
                        msg=f"dev reuses scene theme verbatim: {row}",
                    )
                allowed_short_names.extend(
                    item["object_name_zh"]
                    for item in entry["objects"]
                    if len(normalize_semantic_text(item["object_name_zh"])) <= 2
                )

            def without_exemptions(text):
                normalized = normalize_semantic_text(text)
                for name in allowed_short_names:
                    normalized = normalized.replace(normalize_semantic_text(name), "")
                return normalized

            dev_ngrams = ngrams(without_exemptions(row["text"]))
            for train_row in train_by_label[row["label_id"]]:
                overlap = dev_ngrams & ngrams(without_exemptions(train_row["text"]))
                self.assertFalse(
                    overlap,
                    msg=(
                        f"content leakage for {row['label_id']}: {sorted(overlap)}; "
                        f"train={train_row['text']!r}; dev={row['text']!r}"
                    ),
                )

    def test_generated_files_and_checkpoint_match_the_catalog(self):
        train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
        dev = json.loads(DEV_PATH.read_text(encoding="utf-8"))
        report = json.loads(
            (ROOT / "models" / "detailed_action_semantic" / "training_report.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

        expected_train, expected_dev = split_detailed_action_examples(self.examples)
        self.assertEqual(train, expected_train)
        self.assertEqual(dev, expected_dev)

        self.assertTrue(train)
        self.assertTrue(dev)
        all_rows = [*train, *dev]
        self.assertTrue(
            all(
                set(row) == {"text", "label_id", "source"}
                and row["label_id"] in self.labels
                and isinstance(row["text"], str)
                and row["text"].strip()
                and isinstance(row["source"], str)
                and row["source"].strip()
                for row in all_rows
            )
        )
        self.assertEqual(
            Counter(row["label_id"] for row in train),
            Counter({label: 5 for label in self.labels}),
        )
        self.assertEqual(
            Counter(row["label_id"] for row in dev),
            Counter({label: 1 for label in self.labels}),
        )
        self.assertEqual(
            len({normalize_semantic_text(row["text"]) for row in all_rows}),
            len(all_rows),
        )
        self.assertEqual(
            {normalize_semantic_text(row["text"]) for row in train}
            & {normalize_semantic_text(row["text"]) for row in dev},
            set(),
        )
        self.assertEqual(
            checkpoint["format"], "semantic_action_character_tfidf_centroid_v2"
        )
        self.assertEqual(checkpoint["feature_ngram_range"], [1, 4])
        self.assertEqual(checkpoint["scoring"], "global_254_class_cosine")
        self.assertEqual(checkpoint["text_normalization"], "label_free_zh_v1")
        self.assertEqual(set(checkpoint["labels"]), self.labels)
        self.assertEqual(checkpoint["catalog_sha256"], catalog_sha256(CATALOGUE_PATH))
        self.assertEqual(checkpoint["train_examples"], len(train))
        self.assertEqual(checkpoint["dev_examples"], len(dev))
        expected_determinism = {
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
        self.assertEqual(checkpoint["deterministic_settings"], expected_determinism)
        self.assertEqual(checkpoint["cublas_workspace_config"], ":4096:8")
        self.assertEqual(report["deterministic_settings"], expected_determinism)
        self.assertEqual(report["cublas_workspace_config"], ":4096:8")
        model, vocabulary, loaded_labels = load_checkpoint(
            str(CHECKPOINT_PATH), device="cpu"
        )
        self.assertFalse(model.training)
        self.assertTrue(vocabulary)
        self.assertEqual(set(loaded_labels), self.labels)
        self.assertEqual(model.__class__.__name__, "CharacterTfidfCentroidClassifier")

    def test_checkpoint_features_are_train_observable_and_never_label_addressed(self):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        vocabulary = checkpoint["vocabulary"]
        forbidden_prefixes = ("lex_entity\t", "lex_scene\t", "lex_task\t")

        self.assertFalse(
            [token for token in vocabulary if token.startswith(forbidden_prefixes)]
        )
        self.assertFalse(
            [
                token
                for token in vocabulary
                if any(label in token for label in self.labels)
            ]
        )
        self.assertNotIn("semantic_feature_schema", checkpoint)
        self.assertTrue(
            all(
                token == "<unk>" or token.startswith(("n1:", "n2:", "n3:", "n4:"))
                for token in vocabulary
            )
        )

    def test_training_enables_and_reports_deterministic_torch_configuration(self):
        settings = training.configure_deterministic_training(20260813)

        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertEqual(
            settings,
            {
                "deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
        )

    def test_training_cli_has_the_required_reproducible_defaults(self):
        args = parse_args([])

        self.assertEqual(args.epochs, 40)
        self.assertEqual(args.seed, 20260813)
        self.assertEqual(args.train, TRAIN_PATH)
        self.assertEqual(args.dev, DEV_PATH)
        self.assertEqual(args.model_output, CHECKPOINT_PATH)

    def test_unseen_skeleton_metrics_are_computed_and_match_the_report(self):
        metrics = evaluate_semantic_checkpoint(
            CHECKPOINT_PATH, DEV_PATH, device="cpu", top_k=3
        )
        report = json.loads(
            (ROOT / "models" / "detailed_action_semantic" / "training_report.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertGreaterEqual(metrics["top1_accuracy"], 0.0)
        self.assertLessEqual(metrics["top1_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["top3_accuracy"], metrics["top1_accuracy"])
        self.assertEqual(metrics["examples"], 254)
        self.assertEqual(metrics["scene_examples"], 49)
        self.assertEqual(metrics["object_examples"], 205)
        self.assertIn("scene_top1_accuracy", metrics)
        self.assertIn("object_top1_accuracy", metrics)
        self.assertIn("major_confusions", metrics)
        self.assertEqual(metrics["top1_accuracy"], report["best_dev_accuracy"])
        self.assertEqual(metrics["top3_accuracy"], report["top3_accuracy"])
        self.assertEqual(metrics["scene_top1_accuracy"], report["scene_top1_accuracy"])
        self.assertEqual(metrics["object_top1_accuracy"], report["object_top1_accuracy"])


if __name__ == "__main__":
    unittest.main()
