"""
Internal training loops for ``Embedder.fit()``.

Not part of the public API — called by ``Embedder`` when ``mode`` is
``"head"`` or ``"full"``.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from oodkit.embeddings._guard import require_ml_deps

require_ml_deps()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402


def _run_epoch(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Single training epoch; returns mean loss.

    No per-batch progress bar: nested ``tqdm`` + ``set_postfix`` every step spams
    one line per batch in Jupyter and many non-TTY logs. Use the outer epoch bar
    for ``avg_loss`` (mean over the epoch).
    """
    backbone.train()
    head.train()
    running_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        with torch.set_grad_enabled(backbone.training and any(p.requires_grad for p in backbone.parameters())):
            features = backbone(images).last_hidden_state[:, 0]  # CLS token
        logits = head(features)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def train_head(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Train only the classifier head (backbone frozen).

    Args:
        backbone: Pretrained backbone; parameters are frozen.
        head: Linear classifier head to train.
        dataloader: Training data yielding ``(images, labels)`` batches.
        epochs: Number of training epochs.
        lr: Learning rate for Adam on the classifier head only.
        device: Target device.

    Returns:
        ``(backbone, head)`` after training.
    """
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    head.train()

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    epoch_bar = tqdm(range(epochs), desc="training (head)")
    for _ in epoch_bar:
        avg_loss = _run_epoch(backbone, head, dataloader, optimizer, loss_fn, device)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return backbone, head


def train_full(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    backbone_lr_ratio: float = 0.1,
) -> Tuple[nn.Module, nn.Module]:
    """Finetune backbone and head end-to-end.

    Args:
        backbone: Pretrained backbone; all parameters unfrozen.
        head: Linear classifier head.
        dataloader: Training data yielding ``(images, labels)`` batches.
        epochs: Number of training epochs.
        lr: Learning rate for Adam on the **classifier head**.
        device: Target device.
        backbone_lr_ratio: Positive multiplier applied to ``lr`` for backbone
            parameters (default ``0.1``). Use ``1.0`` for the same LR as the head.

    Returns:
        ``(backbone, head)`` after training.

    Raises:
        ValueError: If ``backbone_lr_ratio`` is not positive.
    """
    if backbone_lr_ratio <= 0:
        raise ValueError(f"backbone_lr_ratio must be positive, got {backbone_lr_ratio}")

    for p in backbone.parameters():
        p.requires_grad = True
    backbone.train()
    head.train()

    optimizer = torch.optim.Adam(
        [
            {"params": list(backbone.parameters()), "lr": lr * backbone_lr_ratio},
            {"params": list(head.parameters()), "lr": lr},
        ],
    )
    loss_fn = nn.CrossEntropyLoss()

    epoch_bar = tqdm(range(epochs), desc="training (full)")
    for _ in epoch_bar:
        avg_loss = _run_epoch(backbone, head, dataloader, optimizer, loss_fn, device)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return backbone, head


def save_checkpoint(
    backbone: nn.Module,
    head: Optional[nn.Module],
    save_path: str,
    meta: Dict,
) -> Path:
    """Persist model weights and metadata to disk.

    Args:
        backbone: Backbone model (state dict saved).
        head: Classifier head (state dict saved if not ``None``).
        save_path: Directory to write into (created if missing).
        meta: Dict with backbone preset name, n_classes, embed_dim, mode, epochs.

    Returns:
        ``Path`` to the saved directory.
    """
    out = Path(save_path)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), out / "backbone.pt")
    if head is not None:
        torch.save(head.state_dict(), out / "head.pt")
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out
