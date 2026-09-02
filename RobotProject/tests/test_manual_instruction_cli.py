import unittest

import manual_instruction_test


class ManualInstructionCliTest(unittest.TestCase):
    def test_default_checkpoint_is_current_semantic_model(self):
        self.assertEqual(
            manual_instruction_test.DEFAULT_CHECKPOINT.name,
            "action_transformer_project_generation_semantic.pt",
        )


if __name__ == "__main__":
    unittest.main()
