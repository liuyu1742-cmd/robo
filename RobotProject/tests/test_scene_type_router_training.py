import unittest

from tools.train_scene_type_router import select_router_candidate


class SceneTypeRouterTrainingTest(unittest.TestCase):
    def test_selects_highest_scene_accuracy_without_object_regression(self):
        trials = [
            {"threshold": -0.5, "scene": 0.92, "object": 0.90, "overall": 0.91},
            {"threshold": 0.0, "scene": 0.86, "object": 0.915, "overall": 0.90},
            {"threshold": 0.5, "scene": 0.84, "object": 0.92, "overall": 0.90},
        ]
        selected = select_router_candidate(trials, minimum_object=0.9149)
        self.assertEqual(selected["threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
