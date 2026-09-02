import json
import torch
from torch.utils.data import Dataset

from tools.encode import (
    encode_task,
    encode_object,
    encode_actions
)

class RobotDataset(Dataset):

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        task = encode_task(item["task"])
        obj = encode_object(item["object"])
        actions = encode_actions(item["actions"])

        return {
            "task": torch.tensor(task),
            "object": torch.tensor(obj),
            "actions": torch.tensor(actions)
        }