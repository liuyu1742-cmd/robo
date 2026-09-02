"""Train the frozen 15-task / 120-object instruction action planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tools.encode import (
    ACTION_VOCAB,
    ARG_PAD_ID,
    ARG_VOCAB,
    OBJECT_VOCAB,
    PAD_ID,
    TARGET_VOCAB,
    TASK_VOCAB,
    encode_object,
    encode_plan,
    encode_target,
    encode_task,
)
from tools.generation import beam_search
from tools.instruction_action_model import ActionTransformer


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "datasets" / "standard_instruction_action_benchmark.json"
DEFAULT_OUTPUT = ROOT / "models" / "instruction_action_transformer_v1.pt"
DEFAULT_REPORT = ROOT / "datasets" / "instruction_action_training_report.json"


class HouseholdInstructionDataset(Dataset):
    def __init__(self, path: Path, split: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.data = [item for item in data if item["split"] == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        action_ids, argument_ids = encode_plan(item["actions"])
        return {
            "task": torch.tensor(encode_task(item["task"]), dtype=torch.long),
            "object": torch.tensor(encode_object(item["object"]), dtype=torch.long),
            "target": torch.tensor(encode_target(item["target"]), dtype=torch.long),
            "actions": torch.tensor(action_ids, dtype=torch.long),
            "arguments": torch.tensor(argument_ids, dtype=torch.long),
        }


def collate_batch(batch):
    max_length = max(item["actions"].size(0) for item in batch)
    actions = torch.full((len(batch), max_length), PAD_ID, dtype=torch.long)
    arguments = torch.full(
        (len(batch), max_length),
        ARG_PAD_ID,
        dtype=torch.long,
    )
    for index, item in enumerate(batch):
        length = item["actions"].size(0)
        actions[index, :length] = item["actions"]
        arguments[index, :length] = item["arguments"]
    return {
        "task": torch.stack([item["task"] for item in batch]),
        "object": torch.stack([item["object"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "actions": actions,
        "arguments": arguments,
    }


def generated_exact_accuracy(model, dataset, device: str, beam_width: int) -> float:
    correct = 0
    model.eval()
    for item in dataset.data:
        prediction = beam_search(
            model,
            encode_task(item["task"]),
            encode_object(item["object"]),
            encode_target(item["target"]),
            beam_width=beam_width,
            max_actions=12,
            device=device,
        )
        verbs, arguments = encode_plan(item["actions"])
        expected = list(zip(verbs[1:-1], arguments[1:-1]))
        correct += prediction == expected
    model.train()
    return correct / len(dataset) if len(dataset) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HouseholdInstructionDataset(args.benchmark, "train")
    validation_dataset = HouseholdInstructionDataset(args.benchmark, "validation")
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    model_config = {
        "task_size": len(TASK_VOCAB),
        "obj_size": len(OBJECT_VOCAB),
        "target_size": len(TARGET_VOCAB),
        "vocab_size": len(ACTION_VOCAB),
        "arg_vocab_size": len(ARG_VOCAB),
        "d_model": 64,
    }
    model = ActionTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    action_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    argument_loss_fn = nn.CrossEntropyLoss(ignore_index=ARG_PAD_ID)
    best_accuracy = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"training: train={len(train_dataset)} validation={len(validation_dataset)} "
        f"device={device}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            action_logits, argument_logits = model(
                batch["task"],
                batch["object"],
                batch["target"],
                batch["actions"][:, :-1],
                batch["arguments"][:, :-1],
            )
            action_loss = action_loss_fn(
                action_logits.reshape(-1, action_logits.shape[-1]),
                batch["actions"][:, 1:].reshape(-1),
            )
            argument_loss = argument_loss_fn(
                argument_logits.reshape(-1, argument_logits.shape[-1]),
                batch["arguments"][:, 1:].reshape(-1),
            )
            loss = action_loss + argument_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        average_loss = total_loss / max(len(loader), 1)
        if epoch % args.validate_every and epoch != args.epochs:
            print(f"epoch {epoch:03d}: loss={average_loss:.4f}", flush=True)
            continue
        validation_accuracy = generated_exact_accuracy(
            model,
            validation_dataset,
            device,
            args.beam_width,
        )
        history.append(
            {
                "epoch": epoch,
                "loss": average_loss,
                "validation_exact_sequence_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch {epoch:03d}: loss={average_loss:.4f} "
            f"validation_exact={validation_accuracy:.2%}",
            flush=True,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_exact_match": validation_accuracy,
                    "task_vocab": TASK_VOCAB,
                    "object_vocab": OBJECT_VOCAB,
                    "action_vocab": ACTION_VOCAB,
                    "argument_vocab": ARG_VOCAB,
                    "target_vocab": TARGET_VOCAB,
                    "model_config": model_config,
                },
                args.output,
            )
            print(f"saved: {args.output.resolve()}", flush=True)
        if best_accuracy >= 1.0 and epoch >= 10:
            break

    report = {
        "format": "instruction_action_training_report_v1",
        "benchmark": str(args.benchmark.resolve()),
        "checkpoint": str(args.output.resolve()),
        "device": device,
        "train_cases": len(train_dataset),
        "validation_cases": len(validation_dataset),
        "best_validation_exact_sequence_accuracy": best_accuracy,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
