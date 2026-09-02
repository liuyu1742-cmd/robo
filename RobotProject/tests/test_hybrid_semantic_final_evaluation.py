import tempfile
import unittest
from pathlib import Path

from evaluate_hybrid_semantic_final import ensure_not_previously_evaluated


class HybridSemanticFinalEvaluationTest(unittest.TestCase):
    def test_existing_lock_refuses_a_second_final_run(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "final.lock.json"
            lock.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already been run"):
                ensure_not_previously_evaluated(lock)


if __name__ == "__main__":
    unittest.main()
