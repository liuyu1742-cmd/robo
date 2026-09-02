import copy
import unittest

from tools.detailed_action_catalog import (
    build_detailed_action_catalog,
    load_detailed_action_catalog,
    validate_detailed_action_catalog,
)


class DetailedActionCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_detailed_action_catalog()

    def test_builds_single_catalog_with_all_254_stable_intents(self):
        entries = self.catalog["entries"]
        labels = [entry["label"] for entry in entries]
        scenes = [entry for entry in entries if entry["entry_type"] == "scene"]
        objects = [entry for entry in entries if entry["entry_type"] == "object"]

        self.assertEqual(self.catalog["summary"], {
            "scene_count": 49,
            "object_entry_count": 205,
            "intent_count": 254,
        })
        self.assertEqual(len(entries), 254)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(scenes), 49)
        self.assertEqual(len(objects), 205)
        self.assertTrue(all(entry["label"].startswith("scene:") for entry in scenes))
        self.assertTrue(all(entry["label"].startswith("object:") for entry in objects))

    def test_object_entries_keep_only_extra_detailed_actions_and_sources(self):
        objects = [
            entry for entry in self.catalog["entries"] if entry["entry_type"] == "object"
        ]

        self.assertEqual({entry["source"] for entry in objects}, {"original", "vla"})
        self.assertEqual(
            len({(entry["source"], entry["object_id"]) for entry in objects}), 205
        )
        self.assertTrue(all(entry["detailed_actions"] for entry in objects))
        self.assertTrue(
            all(
                "original_actions" not in entry and "final_actions" not in entry
                for entry in objects
            )
        )

    def test_scenes_refer_to_catalogued_objects_with_commands_and_controls(self):
        objects_by_label = {
            entry["label"]: entry
            for entry in self.catalog["entries"]
            if entry["entry_type"] == "object"
        }
        scenes = [
            entry for entry in self.catalog["entries"] if entry["entry_type"] == "scene"
        ]

        for scene in scenes:
            with self.subTest(scene=scene["scenario_id"]):
                self.assertTrue(scene["command"].strip())
                self.assertTrue(scene["objects"])
                for item in scene["objects"]:
                    self.assertIn(item["object_label"], objects_by_label)
                    self.assertTrue(item["detailed_actions"])
                    self.assertTrue(item["control_parameters"].strip())
                    referenced = objects_by_label[item["object_label"]]
                    self.assertEqual(item["object_name_zh"], referenced["object_name_zh"])
                    self.assertEqual(item["detailed_actions"], referenced["detailed_actions"])
                    self.assertEqual(
                        item["control_parameters"], referenced["control_parameters"]
                    )

    def test_validator_rejects_duplicate_labels_invalid_references_and_leaks(self):
        invalid = copy.deepcopy(self.catalog)
        invalid["entries"][1]["label"] = invalid["entries"][0]["label"]
        invalid["entries"][0]["objects"][0]["object_label"] = "object:original:missing"
        invalid["entries"][-1]["original_actions"] = ["leaked"]

        errors = validate_detailed_action_catalog(invalid)

        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("reference" in error for error in errors))
        self.assertTrue(any("leak" in error for error in errors))

    def test_validator_rejects_scene_object_content_mismatches(self):
        invalid = copy.deepcopy(self.catalog)
        scene = next(
            entry for entry in invalid["entries"] if entry["entry_type"] == "scene"
        )
        referenced = scene["objects"][0]
        donor = next(
            entry
            for entry in invalid["entries"]
            if entry["entry_type"] == "object"
            and entry["label"] != referenced["object_label"]
        )
        referenced["object_name_zh"] = donor["object_name_zh"]
        referenced["detailed_actions"] = donor["detailed_actions"]
        referenced["control_parameters"] = "mismatched controls"

        errors = validate_detailed_action_catalog(invalid)

        self.assertTrue(any("object name mismatch" in error for error in errors))
        self.assertTrue(any("detailed actions mismatch" in error for error in errors))
        self.assertTrue(any("control parameters mismatch" in error for error in errors))

    def test_loads_the_generated_catalogue(self):
        loaded = load_detailed_action_catalog()

        self.assertEqual(validate_detailed_action_catalog(loaded), [])
        self.assertEqual(loaded["summary"]["intent_count"], 254)


if __name__ == "__main__":
    unittest.main()
