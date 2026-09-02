import unittest

from tools.hybrid_semantic_action_runtime import (
    SemanticResolvedInstruction,
    canonical_actions_for_relation,
    constrain_semantic_actions,
    relation_catalog,
)


class HybridSemanticActionRuntimeTest(unittest.TestCase):
    def test_catalog_exposes_canonical_actions_for_relation(self):
        catalog = relation_catalog()
        self.assertIn("cleaning::carpet", catalog)
        self.assertEqual(
            canonical_actions_for_relation("cleaning::carpet", catalog),
            ["locate(carpet)", "clean(carpet)", "inspect(carpet)"],
        )

    def test_unknown_relation_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unknown relation"):
            canonical_actions_for_relation("missing::object", {})

    def test_semantic_constraint_uses_relation_plan_not_operation_override(self):
        catalog = relation_catalog()
        resolved = SemanticResolvedInstruction(
            task="maintenance_management",
            object="fire_alarm",
            target="none",
            operation="semantic_relation",
            relation_key="maintenance_management::fire_alarm",
            confidence=0.99,
        )
        expected = canonical_actions_for_relation(resolved.relation_key, catalog)
        result = constrain_semantic_actions(resolved, expected, catalog)
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_actions, expected)


if __name__ == "__main__":
    unittest.main()
