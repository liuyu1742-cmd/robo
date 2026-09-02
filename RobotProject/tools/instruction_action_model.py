"""Transformer used by the frozen instruction-to-action benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn

from tools.encode import ACTION_VOCAB, ARG_VOCAB, OBJECT_VOCAB, TARGET_VOCAB, TASK_VOCAB


class ActionTransformer(nn.Module):
    def __init__(
        self,
        task_size=None,
        obj_size=None,
        target_size=None,
        vocab_size=None,
        arg_vocab_size=None,
        d_model=64,
        nhead=4,
        max_seq_len=64,
    ):
        super().__init__()
        task_size = task_size or len(TASK_VOCAB)
        obj_size = obj_size or len(OBJECT_VOCAB)
        target_size = target_size or len(TARGET_VOCAB)
        vocab_size = vocab_size or len(ACTION_VOCAB)
        arg_vocab_size = arg_vocab_size or len(ARG_VOCAB)
        self.task_emb = nn.Embedding(task_size, d_model)
        self.obj_emb = nn.Embedding(obj_size, d_model)
        self.target_emb = nn.Embedding(target_size, d_model)
        self.act_emb = nn.Embedding(vocab_size, d_model)
        self.arg_emb = nn.Embedding(arg_vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.action_head = nn.Linear(d_model, vocab_size)
        self.argument_head = nn.Linear(d_model, arg_vocab_size)

    @staticmethod
    def generate_mask(size, device):
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, task, obj, target, actions, arguments):
        batch_size = task.size(0)
        conditioning = [
            self.cls.expand(batch_size, -1, -1),
            self.task_emb(task).unsqueeze(1),
            self.obj_emb(obj).unsqueeze(1),
            self.target_emb(target).unsqueeze(1),
        ]
        action_tokens = self.act_emb(actions) + self.arg_emb(arguments)
        values = torch.cat([*conditioning, action_tokens], dim=1)
        positions = torch.arange(values.size(1), device=values.device).unsqueeze(0)
        values = values + self.pos_emb(positions)
        conditioning_padding = torch.zeros(
            batch_size,
            4,
            dtype=torch.bool,
            device=values.device,
        )
        padding_mask = torch.cat([conditioning_padding, actions.eq(0)], dim=1)
        values = self.transformer(
            values,
            mask=self.generate_mask(values.size(1), values.device),
            src_key_padding_mask=padding_mask,
        )
        states = values[:, 4:, :]
        return self.action_head(states), self.argument_head(states)
