"""
Shared utilities for LLM Quantization on NLP Tasks.

Usage in Colab:
    # Upload this file to the same directory, then:
    from utils import set_seed, get_model_size_mb, save_checkpoint, load_checkpoint, collate_fn
"""

import os
import random
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, PyTorch, and CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_size_mb(model: nn.Module) -> float:
    """Return total model memory footprint (parameters + buffers) in megabytes."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1e6


def save_checkpoint(
    model, optimizer, scheduler, epoch, train_losses, eval_losses, filepath: str
) -> None:
    """Save a training checkpoint, creating parent directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "eval_losses": eval_losses,
        },
        filepath,
    )
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath: str):
    """Load a training checkpoint from filepath.

    Raises FileNotFoundError if the checkpoint does not exist.
    Returns: (model, optimizer, scheduler, epoch, train_losses, eval_losses)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Checkpoint not found: {filepath}\n"
            "Run the corresponding fine-tuning notebook first."
        )
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Checkpoint loaded: {filepath} (epoch {ckpt['epoch'] + 1})")
    return (
        model,
        optimizer,
        scheduler,
        ckpt["epoch"],
        ckpt["train_losses"],
        ckpt["eval_losses"],
    )


def collate_fn(batch):
    """Collate a batch of seq2seq examples into stacked tensors."""
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack(
        [torch.tensor(item["attention_mask"]) for item in batch]
    )
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
