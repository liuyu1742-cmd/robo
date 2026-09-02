import torch
import torch.nn as nn

class RobotTransformer(nn.Module):

    def __init__(self,
                 task_size=10,
                 obj_size=20,
                 act_size=10,
                 d_model=64):

        super().__init__()

        self.task_emb = nn.Embedding(task_size, d_model)
        self.obj_emb = nn.Embedding(obj_size, d_model)
        self.act_emb = nn.Embedding(act_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(d_model, act_size)

    def forward(self, task, obj, actions):

        t = self.task_emb(task).unsqueeze(1)
        o = self.obj_emb(obj).unsqueeze(1)
        a = self.act_emb(actions)

        x = torch.cat([t, o, a], dim=1)

        x = self.encoder(x)

        out = self.fc(x)

        return out