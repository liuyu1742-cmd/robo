import unittest

from tools.calibrate_scene_intent import select_scene_candidate


class SceneIntentCalibrationTest(unittest.TestCase):
    def test_selects_highest_scene_accuracy_without_object_regression(self):
        trials = [
            {"bias": -1.0, "scene": 0.80, "object": 0.915, "overall": 0.89},
            {"bias": -0.5, "scene": 0.90, "object": 0.915, "overall": 0.91},
            {"bias": 0.0, "scene": 0.95, "object": 0.90, "overall": 0.91},
        ]
        selected = select_scene_candidate(trials, minimum_object=0.9149)
        self.assertEqual(selected["bias"], -0.5)


if __name__ == "__main__":
    unittest.main()
