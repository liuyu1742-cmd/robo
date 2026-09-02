import json
import torch
from torch.utils.data import Dataset

class HouseholdDataset(Dataset):

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            "task": item["task"],
            "scene": item["scene"],
            "object": item["object"],
            "target": item["target"],
            "actions": item["actions"]
        }


if __name__ == "__main__":
    ds = HouseholdDataset("datasets/final_train_dataset.json")
    print(len(ds))
    print(ds[0])