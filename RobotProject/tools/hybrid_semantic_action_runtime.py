"""Trainable semantic relation prediction plus canonical action planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from tools.action_constraints import ActionConstraintResult
from tools.semantic_action_model import load_checkpoint, predict_relations


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
DEFAULT_MODEL = ROOT / "models" / "semantic_action" / "best.pt"


@dataclass(frozen=True)
class SemanticResolvedInstruction:
    """Resolved relation and its confidence, before action constraint handling."""

    task: str
    object: str
    target: str
    operation: str
    relation_key: str
    confidence: float


@lru_cache(maxsize=1)
def relation_catalog(path: str = str(DEFAULT_CATALOG)) -> dict[str, dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    catalog = {str(record["relation_key"]): dict(record) for record in records}
    if len(catalog) != len(records):
        raise ValueError("catalog relation keys must be unique")
    return catalog


def canonical_actions_for_relation(relation_key: str, catalog: dict[str, dict]) -> list[str]:
    try:
        return list(catalog[relation_key]["actions"])
    except KeyError as exc:
        raise KeyError(f"unknown relation: {relation_key}") from exc


def constrain_semantic_actions(
    resolved: SemanticResolvedInstruction,
    predicted_actions: list[str],
    catalog: dict[str, dict],
) -> ActionConstraintResult:
    """Constrain using the predicted relation's canonical plan.

    ``operation`` is an internal marker for this classifier and must never be
    passed into the legacy operation-specific planner.
    """
    expected_actions = canonical_actions_for_relation(resolved.relation_key, catalog)
    if predicted_actions == expected_actions:
        return ActionConstraintResult(True, list(predicted_actions), "semantic_relation_match", [])
    missing = [action for action in expected_actions if action not in set(predicted_actions)]
    unexpected = [action for action in predicted_actions if action not in set(expected_actions)]
    violations = (["missing_required:" + ",".join(missing)] if missing else [])
    violations += (["unexpected_action:" + ",".join(unexpected)] if unexpected else [])
    if not violations:
        violations = ["action_order_mismatch"]
    return ActionConstraintResult(
        False,
        expected_actions,
        "semantic_relation_canonicalized",
        violations,
    )


class HybridSemanticActionParser:
    """Classify a natural-language instruction, then retrieve its canonical plan."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        catalog_path: Path = DEFAULT_CATALOG,
        *,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vocabulary, self.labels = load_checkpoint(
            str(model_path), device=self.device
        )
        self.catalog = relation_catalog(str(catalog_path))
        if set(self.labels) != set(self.catalog):
            raise ValueError("model label set and current canonical catalog differ")

    def resolve(self, instruction: str) -> SemanticResolvedInstruction:
        prediction = predict_relations(
            self.model, self.vocabulary, self.labels, [instruction], device=self.device
        )[0]
        relation_key = prediction["relation_key"]
        record = self.catalog[relation_key]
        # Deliberately avoid operation-plan overrides: the canonical relation
        # sequence is the planner source of truth for the predicted relation.
        return SemanticResolvedInstruction(
            task=str(record["acceptance_task"]),
            object=str(record["object"]),
            target=str(record["target"]),
            operation="semantic_relation",
            relation_key=relation_key,
            confidence=float(prediction["confidence"]),
        )

    def generate_raw_actions(self, instruction: str) -> tuple[SemanticResolvedInstruction, list[str]]:
        resolved = self.resolve(instruction)
        return resolved, self.generate_raw_actions_for_resolved(resolved)

    def generate_raw_actions_for_resolved(
        self, resolved: SemanticResolvedInstruction
    ) -> list[str]:
        return canonical_actions_for_relation(resolved.relation_key, self.catalog)
