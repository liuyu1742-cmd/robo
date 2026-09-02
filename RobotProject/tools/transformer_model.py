import torch
import torch.nn as nn

class ActionTransformer(nn.Module):

    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):

        # x: [batch, seq]
        x = self.embedding(x)

        x = self.transformer(x)

        out = self.fc(x)

        return out