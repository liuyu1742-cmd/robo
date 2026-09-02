import json
import torch
from torch.utils.data import Dataset

from tools.encode import encode_actions

class HouseholdDataset(Dataset):

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        actions = encode_actions(item["actions"])

        return torch.tensor(actions)
