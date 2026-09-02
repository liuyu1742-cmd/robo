"""Small trainable text classifier used by the hybrid semantic action parser."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import torch
from torch import nn


UNK_TOKEN = "<unk>"


def _features(text: str, *, min_n: int = 1, max_n: int = 2) -> list[str]:
    """Character unigrams and adjacent bigrams, robust to Chinese word spacing."""
    chars = [char for char in text.casefold() if not char.isspace()]
    if not chars:
        return [UNK_TOKEN]
    if (min_n, max_n) == (1, 2):
        padded = ["^"] + chars + ["$"]
        unigrams = [f"c:{char}" for char in chars]
        bigrams = [f"b:{left}{right}" for left, right in zip(padded, padded[1:])]
        return unigrams + bigrams
    return [
        f"n{size}:{''.join(chars[index:index + size])}"
        for size in range(min_n, max_n + 1)
        for index in range(max(0, len(chars) - size + 1))
    ]


def build_vocabulary(texts: Iterable[str], *, min_frequency: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_features(text))
    vocabulary = {UNK_TOKEN: 0}
    for token in sorted(token for token, count in counter.items() if count >= min_frequency):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def build_character_ngram_vocabulary(
    texts: Iterable[str], *, min_n: int = 1, max_n: int = 4
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_features(text, min_n=min_n, max_n=max_n))
    vocabulary = {UNK_TOKEN: 0}
    for token in sorted(counter):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(
    text: str, vocabulary: dict[str, int], *, normalize_text: bool = False
) -> list[int]:
    if normalize_text:
        from tools.semantic_text_normalization import normalize_for_semantic_model

        text = normalize_for_semantic_model(text)
    uses_v2_features = any(token.startswith("n3:") or token.startswith("n4:") for token in vocabulary)
    features = _features(text, min_n=1, max_n=4) if uses_v2_features else _features(text)
    encoded = [vocabulary.get(feature, vocabulary[UNK_TOKEN]) for feature in features]
    return encoded or [vocabulary[UNK_TOKEN]]


def vectorize_batch(
    texts: Sequence[str], vocabulary: dict[str, int], *, device: torch.device | None = None,
    normalize_text: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create EmbeddingBag's flattened feature tensor and per-example offsets."""
    encoded = [encode_text(text, vocabulary, normalize_text=normalize_text) for text in texts]
    offsets: list[int] = []
    values: list[int] = []
    for features in encoded:
        offsets.append(len(values))
        values.extend(features)
    return (
        torch.tensor(values, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


class CharacterNgramRelationClassifier(nn.Module):
    """Mean pooled character n-gram embeddings followed by a relation classifier."""

    def __init__(self, vocab_size: int, class_count: int, embedding_dim: int = 96) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.dropout = nn.Dropout(0.10)
        self.classifier = nn.Linear(embedding_dim, class_count)

    def forward(self, values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embedding(values, offsets)))


class CharacterTfidfCentroidClassifier(nn.Module):
    """Global cosine-style scorer compiled from per-label TF-IDF centroids."""

    def __init__(self, feature_class_weights: torch.Tensor) -> None:
        super().__init__()
        if feature_class_weights.ndim != 2:
            raise ValueError("feature-class weights must be a matrix")
        self.embedding = nn.EmbeddingBag.from_pretrained(
            feature_class_weights, freeze=True, mode="sum"
        )

    def forward(self, values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.embedding(values, offsets)


def make_tfidf_centroid_checkpoint(
    rows: Sequence[dict], vocabulary: dict[str, int], labels: Sequence[str], *,
    text_normalization: str = "none",
) -> dict:
    """Fit a deterministic 1-4 gram TF-IDF centroid model over all labels."""
    label_to_index = {label: index for index, label in enumerate(labels)}
    document_frequency: Counter[int] = Counter()
    documents: list[Counter[int]] = []
    for row in rows:
        features = Counter(
            encode_text(
                str(row["text"]),
                vocabulary,
                normalize_text=text_normalization == "label_free_zh_v1",
            )
        )
        documents.append(features)
        document_frequency.update(features)

    idf = torch.zeros(len(vocabulary), dtype=torch.float32)
    for feature_index in range(len(vocabulary)):
        idf[feature_index] = math.log(
            (1 + len(rows)) / (1 + document_frequency[feature_index])
        ) + 1.0

    centroids = torch.zeros((len(labels), len(vocabulary)), dtype=torch.float32)
    for row, features in zip(rows, documents):
        vector = torch.zeros(len(vocabulary), dtype=torch.float32)
        for feature_index, count in features.items():
            vector[feature_index] = float(count) * idf[feature_index]
        norm = torch.linalg.vector_norm(vector)
        if float(norm) > 0:
            vector /= norm
        centroids[label_to_index[str(row["label_id"])]] += vector
    centroids /= torch.linalg.vector_norm(centroids, dim=1, keepdim=True).clamp_min(1e-12)
    feature_class_weights = (centroids * idf.unsqueeze(0)).transpose(0, 1).contiguous()
    return {
        "format": "semantic_action_character_tfidf_centroid_v2",
        "model_state": {"feature_class_weights": feature_class_weights},
        "vocabulary": vocabulary,
        "labels": list(labels),
        "feature_ngram_range": [1, 4],
        "scoring": "global_254_class_cosine",
        "text_normalization": text_normalization,
    }


def make_checkpoint(
    model: CharacterNgramRelationClassifier,
    vocabulary: dict[str, int],
    labels: Sequence[str],
    *,
    embedding_dim: int,
) -> dict:
    return {
        "format": "semantic_action_character_ngram_v1",
        "model_state": model.state_dict(),
        "vocabulary": vocabulary,
        "labels": list(labels),
        "embedding_dim": embedding_dim,
    }


def load_checkpoint(path: str, *, device: torch.device | str = "cpu") -> tuple[
    nn.Module, dict[str, int], list[str]
]:
    payload = torch.load(path, map_location=device, weights_only=False)
    checkpoint_format = payload.get("format")
    labels = list(payload["labels"])
    vocabulary = dict(payload["vocabulary"])
    if checkpoint_format == "semantic_action_character_tfidf_centroid_v2":
        model = CharacterTfidfCentroidClassifier(
            payload["model_state"]["feature_class_weights"]
        ).to(device)
        model.text_normalization = str(payload.get("text_normalization", "none"))
        model.eval()
        return model, vocabulary, labels
    if checkpoint_format != "semantic_action_character_ngram_v1":
        raise ValueError("unexpected semantic action model checkpoint")
    model = CharacterNgramRelationClassifier(
        len(vocabulary), len(labels), int(payload["embedding_dim"])
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, vocabulary, labels


@torch.inference_mode()
def predict_relations(
    model: CharacterNgramRelationClassifier,
    vocabulary: dict[str, int],
    labels: Sequence[str],
    texts: Sequence[str],
    *,
    device: torch.device | str = "cpu",
) -> list[dict]:
    target_device = torch.device(device)
    values, offsets = vectorize_batch(texts, vocabulary, device=target_device)
    if getattr(model, "text_normalization", "none") == "label_free_zh_v1":
        values, offsets = vectorize_batch(
            texts, vocabulary, device=target_device, normalize_text=True
        )
    probabilities = torch.softmax(model(values, offsets), dim=1)
    confidences, indices = probabilities.max(dim=1)
    return [
        {"relation_key": labels[int(index)], "confidence": float(confidence)}
        for confidence, index in zip(confidences.cpu(), indices.cpu())
    ]
