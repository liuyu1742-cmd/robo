import json
import tempfile
import unittest
from pathlib import Path

from train_project_transformer_v2 import ProjectActionDatasetV2


class ProjectDatasetScopeTest(unittest.TestCase):
    def test_dataset_uses_acceptance_task_and_excludes_unknown_object(self):
        records = [
            {
                "id": "canonical-cup",
                "source": "epic_kitchens_100",
                "verb": "wash",
                "task": "dish_washing",
                "acceptance_task": "cleaning",
                "object": "cup",
                "target": "none",
                "actions": ["locate(cup)"],
            },
            {
                "id": "outside-scope-glove",
                "source": "epic_kitchens_100",
                "verb": "take",
                "task": "object_fetching",
                "acceptance_task": "object_fetching",
                "object": "glove",
                "target": "none",
                "actions": ["locate(glove)"],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            dataset = ProjectActionDatasetV2(path, visual_features=None)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.data[0]["task"], "cleaning")
        self.assertEqual(dataset.excluded_item_count, 1)


if __name__ == "__main__":
    unittest.main()
