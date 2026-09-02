import torch
import torch.nn as nn

class SimpleActionModel(nn.Module):

    def __init__(self, vocab_size=10, hidden=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden)

        self.rnn = nn.GRU(hidden, hidden, batch_first=True)

        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out