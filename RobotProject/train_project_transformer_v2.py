import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tools.project_encode import (
    ACTION_VOCAB, ARG_PAD_ID, ARG_VOCAB, OBJECT_VOCAB, PAD_ID,
    TARGET_VOCAB, TASK_VOCAB, encode_object, encode_plan, encode_target,
    encode_task,
)
from tools.project_generation import autoregressive_metrics


ROOT = Path(__file__).resolve().parent
with (ROOT / "meta" / "project_condition_vocab.json").open("r", encoding="utf-8") as file:
    _COND = json.load(file)
SOURCE_VOCAB = {value: index for index, value in enumerate(_COND["sources"])}
VERB_VOCAB = {value: index for index, value in enumerate(_COND["verbs"])}


class ProjectConditionedTransformer(nn.Module):
    def __init__(
        self, source_size, verb_size, task_size, obj_size, target_size,
        vocab_size, arg_vocab_size, d_model=96, nhead=4, max_seq_len=96,
        visual_dim=0,
    ):
        super().__init__()
        self.source_emb = nn.Embedding(source_size, d_model)
        self.verb_emb = nn.Embedding(verb_size, d_model)
        self.task_emb = nn.Embedding(task_size, d_model)
        self.obj_emb = nn.Embedding(obj_size, d_model)
        self.target_emb = nn.Embedding(target_size, d_model)
        self.act_emb = nn.Embedding(vocab_size, d_model)
        self.arg_emb = nn.Embedding(arg_vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.visual_dim = visual_dim
        if visual_dim > 0:
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Linear(visual_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.visual_missing = nn.Parameter(torch.zeros(1, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.action_head = nn.Linear(d_model, vocab_size)
        self.argument_head = nn.Linear(d_model, arg_vocab_size)

    @staticmethod
    def generate_mask(size, device):
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self, source, verb, task, obj, target, actions, arguments,
        visual=None, visual_present=None,
    ):
        batch_size = task.size(0)
        conditioning = [
            self.cls.expand(batch_size, -1, -1),
            self.source_emb(source).unsqueeze(1),
            self.verb_emb(verb).unsqueeze(1),
            self.task_emb(task).unsqueeze(1),
            self.obj_emb(obj).unsqueeze(1),
            self.target_emb(target).unsqueeze(1),
        ]
        if self.visual_dim > 0:
            if visual is None:
                visual = torch.zeros(batch_size, self.visual_dim, device=task.device)
            visual_token = self.visual_projection(visual)
            if visual_present is None:
                visual_present = torch.zeros(batch_size, dtype=torch.bool, device=task.device)
            visual_token = torch.where(
                visual_present.unsqueeze(1), visual_token, self.visual_missing.expand(batch_size, -1)
            )
            conditioning.append(visual_token.unsqueeze(1))
        condition_count = len(conditioning)
        action_tokens = self.act_emb(actions) + self.arg_emb(arguments)
        x = torch.cat([*conditioning, action_tokens], dim=1)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)
        conditioning_padding = torch.zeros(batch_size, condition_count, dtype=torch.bool, device=x.device)
        padding_mask = torch.cat([conditioning_padding, actions.eq(0)], dim=1)
        x = self.transformer(
            x,
            mask=self.generate_mask(x.size(1), x.device),
            src_key_padding_mask=padding_mask,
        )
        states = x[:, condition_count:, :]
        return self.action_head(states), self.argument_head(states)


class ProjectActionDatasetV2(Dataset):
    def __init__(self, path, max_samples=None, visual_features=None):
        with Path(path).open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
        self.excluded_reasons = Counter()
        scoped_data = []
        for raw_item in raw_data:
            item = dict(raw_item)
            item["task"] = item.get("acceptance_task", item["task"])
            reason = self._scope_exclusion_reason(item)
            if reason is not None:
                self.excluded_reasons[reason] += 1
                continue
            scoped_data.append(item)
        self.raw_item_count = len(raw_data)
        self.excluded_item_count = len(raw_data) - len(scoped_data)
        self.data = (
            scoped_data[:max_samples] if max_samples is not None else scoped_data
        )
        self.visual_dim = 0
        self.visual_by_id = {}
        if visual_features is not None and Path(visual_features).is_file():
            store = np.load(visual_features, allow_pickle=False)
            ids = store["segment_ids"].tolist()
            values = store["features"].astype(np.float32, copy=False)
            complete = store["completed"].astype(bool, copy=False)
            self.visual_dim = int(values.shape[1])
            self.visual_by_id = {
                segment_id: values[index]
                for index, segment_id in enumerate(ids)
                if complete[index]
            }

    @staticmethod
    def _scope_exclusion_reason(item):
        if item["source"] not in SOURCE_VOCAB:
            return "source"
        if item["verb"] not in VERB_VOCAB:
            return "verb"
        if item["task"] not in TASK_VOCAB:
            return "task"
        if item["object"] not in OBJECT_VOCAB:
            return "object"
        if item["target"] not in TARGET_VOCAB:
            return "target"
        try:
            encode_plan(item["actions"])
        except ValueError:
            return "action_or_argument"
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        action_ids, argument_ids = encode_plan(item["actions"])
        visual_value = self.visual_by_id.get(item["id"])
        visual_present = visual_value is not None
        if visual_value is None:
            visual_value = np.zeros(self.visual_dim, dtype=np.float32)
        return {
            "source": torch.tensor(SOURCE_VOCAB[item["source"]], dtype=torch.long),
            "verb": torch.tensor(VERB_VOCAB[item["verb"]], dtype=torch.long),
            "task": torch.tensor(encode_task(item["task"]), dtype=torch.long),
            "object": torch.tensor(encode_object(item["object"]), dtype=torch.long),
            "target": torch.tensor(encode_target(item["target"]), dtype=torch.long),
            "actions": torch.tensor(action_ids, dtype=torch.long),
            "arguments": torch.tensor(argument_ids, dtype=torch.long),
            "visual": torch.from_numpy(visual_value.copy()),
            "visual_present": torch.tensor(visual_present, dtype=torch.bool),
        }


def collate_batch(batch):
    max_length = max(item["actions"].size(0) for item in batch)
    actions = torch.full((len(batch), max_length), PAD_ID, dtype=torch.long)
    arguments = torch.full((len(batch), max_length), ARG_PAD_ID, dtype=torch.long)
    for index, item in enumerate(batch):
        actions[index, : item["actions"].size(0)] = item["actions"]
        arguments[index, : item["arguments"].size(0)] = item["arguments"]
    return {
        "source": torch.stack([item["source"] for item in batch]),
        "verb": torch.stack([item["verb"] for item in batch]),
        "task": torch.stack([item["task"] for item in batch]),
        "object": torch.stack([item["object"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "actions": actions,
        "arguments": arguments,
        "visual": torch.stack([item["visual"] for item in batch]),
        "visual_present": torch.stack([item["visual_present"] for item in batch]),
    }


@torch.no_grad()
def teacher_forced_accuracy(model, dataset, device, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    model.eval()
    action_correct = action_total = 0
    argument_correct = argument_total = 0
    sequence_correct = sequence_total = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        input_actions = batch["actions"][:, :-1]
        target_actions = batch["actions"][:, 1:]
        input_arguments = batch["arguments"][:, :-1]
        target_arguments = batch["arguments"][:, 1:]
        action_logits, argument_logits = model(
            batch["source"], batch["verb"], batch["task"], batch["object"],
            batch["target"], input_actions, input_arguments,
            batch["visual"], batch["visual_present"],
        )
        action_pred = action_logits.argmax(dim=-1)
        argument_pred = argument_logits.argmax(dim=-1)
        action_mask = target_actions.ne(PAD_ID)
        argument_mask = target_arguments.ne(ARG_PAD_ID)
        action_correct += (action_pred.eq(target_actions) & action_mask).sum().item()
        action_total += action_mask.sum().item()
        argument_correct += (argument_pred.eq(target_arguments) & argument_mask).sum().item()
        argument_total += argument_mask.sum().item()
        both = action_pred.eq(target_actions) & argument_pred.eq(target_arguments)
        valid = action_mask | argument_mask
        sequence_correct += ((both | ~valid).all(dim=1)).sum().item()
        sequence_total += target_actions.size(0)
    model.train()
    return {
        "action_token_accuracy": action_correct / action_total if action_total else 0.0,
        "argument_token_accuracy": argument_correct / argument_total if argument_total else 0.0,
        "teacher_forced_sequence_accuracy": sequence_correct / sequence_total if sequence_total else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train project v2 transformer with source+verb conditioning.")
    parser.add_argument("--train", type=Path, default=ROOT / "datasets" / "standard_generation_train.json")
    parser.add_argument("--val", type=Path, default=ROOT / "datasets" / "standard_generation_val.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "action_transformer_project_generation_final.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--visual-features", type=Path,
        default=ROOT / "datasets" / "epic_kitchens" / "processed" / "tier_a_visual_features.npz",
    )
    parser.add_argument(
        "--disable-visual",
        action="store_true",
        help="Train and evaluate without optional visual features.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    visual_features = None if args.disable_visual else args.visual_features
    train_dataset = ProjectActionDatasetV2(
        args.train, args.max_train_samples, visual_features
    )
    val_dataset = ProjectActionDatasetV2(
        args.val, args.max_val_samples, visual_features
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    model_config = {
        "source_size": len(SOURCE_VOCAB), "verb_size": len(VERB_VOCAB),
        "task_size": len(TASK_VOCAB), "obj_size": len(OBJECT_VOCAB),
        "target_size": len(TARGET_VOCAB), "vocab_size": len(ACTION_VOCAB),
        "arg_vocab_size": len(ARG_VOCAB), "d_model": 96,
        "visual_dim": train_dataset.visual_dim,
    }
    model = ProjectConditionedTransformer(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    action_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    argument_loss_fn = nn.CrossEntropyLoss(ignore_index=ARG_PAD_ID)
    best_score = -1.0
    print(f"training start: train={len(train_dataset)}, val={len(val_dataset)}, device={device}, output={args.output}")
    print(f"model_config={model_config}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            input_actions = batch["actions"][:, :-1]
            target_actions = batch["actions"][:, 1:]
            input_arguments = batch["arguments"][:, :-1]
            target_arguments = batch["arguments"][:, 1:]
            action_logits, argument_logits = model(
                batch["source"], batch["verb"], batch["task"], batch["object"],
                batch["target"], input_actions, input_arguments,
                batch["visual"], batch["visual_present"],
            )
            action_loss = action_loss_fn(action_logits.reshape(-1, action_logits.shape[-1]), target_actions.reshape(-1))
            argument_loss = argument_loss_fn(argument_logits.reshape(-1, argument_logits.shape[-1]), target_arguments.reshape(-1))
            loss = action_loss + argument_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / max(len(loader), 1)
        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
            metrics = teacher_forced_accuracy(model, val_dataset, device)
            generation_metrics = autoregressive_metrics(
                model,
                val_dataset.data,
                device,
                beam_width=args.beam_width,
                max_actions=args.max_actions,
            )
            score = generation_metrics["sequence_accuracy"]
            print(
                f"epoch {epoch + 1:02d}: loss={average_loss:.4f}, "
                f"action_acc={metrics['action_token_accuracy']:.2%}, "
                f"arg_acc={metrics['argument_token_accuracy']:.2%}, "
                f"teacher_seq_acc={metrics['teacher_forced_sequence_accuracy']:.2%}, "
                f"generation_seq_acc={score:.2%}, "
                f"truncated={generation_metrics['truncated_count']}"
            )
            if score > best_score:
                best_score = score
                torch.save({
                    "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_teacher_forced_sequence_accuracy": metrics[
                        "teacher_forced_sequence_accuracy"
                    ],
                    "val_metrics": metrics,
                    "val_autoregressive_sequence_accuracy": score,
                    "val_autoregressive_metrics": generation_metrics,
                    "model_config": model_config,
                    "source_vocab": SOURCE_VOCAB, "verb_vocab": VERB_VOCAB,
                    "task_vocab": TASK_VOCAB, "object_vocab": OBJECT_VOCAB,
                    "action_vocab": ACTION_VOCAB, "argument_vocab": ARG_VOCAB,
                    "target_vocab": TARGET_VOCAB,
                    "encoder": "project_v3_source_verb_visual" if train_dataset.visual_dim else "project_v2_source_verb",
                    "visual_features": str(visual_features) if train_dataset.visual_dim else None,
                }, args.output)
                print(f"  best checkpoint saved: {args.output}")
        else:
            print(f"epoch {epoch + 1:02d}: loss={average_loss:.4f}")
    print(f"training done: best val autoregressive sequence accuracy={best_score:.2%}")


if __name__ == "__main__":
    main()
