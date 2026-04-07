"""
Backbone preset registry for ``oodkit.embeddings``.

Loads pretrained vision transformers via HuggingFace ``transformers``.
**Default aliases** are **DINOv2** (public checkpoints). **DINOv3** presets
remain available but may be gated on Hugging Face (login + terms).

Only this module contains model-specific logic; the rest of the embeddings
package is backbone-agnostic.
"""

from dataclasses import dataclass
from typing import Any, Tuple

from oodkit.embeddings._guard import require_ml_deps

require_ml_deps()

import torch.nn as nn  # noqa: E402
from transformers import AutoImageProcessor, AutoModel  # noqa: E402

_HF_PREFIX = "facebook/"


@dataclass(frozen=True)
class BackbonePreset:
    """Immutable descriptor for a pretrained backbone."""

    hf_model_id: str
    embed_dim: int


PRESETS = {
    # DINOv2 — public on Hugging Face; default family for Embedder.
    "dinov2-small": BackbonePreset(
        hf_model_id=f"{_HF_PREFIX}dinov2-small",
        embed_dim=384,
    ),
    "dinov2-base": BackbonePreset(
        hf_model_id=f"{_HF_PREFIX}dinov2-base",
        embed_dim=768,
    ),
    "dinov2-large": BackbonePreset(
        hf_model_id=f"{_HF_PREFIX}dinov2-large",
        embed_dim=1024,
    ),
    # DINOv3 — may be gated; requires HF login + accepted terms on the model card.
    "dinov3-small": BackbonePreset(
        hf_model_id=f"{_HF_PREFIX}dinov3-vits16-pretrain-lvd1689m",
        embed_dim=384,
    ),
    "dinov3-base": BackbonePreset(
        hf_model_id=f"{_HF_PREFIX}dinov3-vitb16-pretrain-lvd1689m",
        embed_dim=768,
    ),
    "dinov3-large": BackbonePreset(
        hf_model_id=f"{_HF_PREFIX}dinov3-vitl16-pretrain-lvd1689m",
        embed_dim=1024,
    ),
}


def load_backbone(name: str) -> Tuple[nn.Module, Any, int]:
    """Load a pretrained backbone by short alias.

    Args:
        name: Key in ``PRESETS`` (e.g. ``"dinov2-small"``).

    Returns:
        ``(model, processor, embed_dim)`` where ``processor`` is a HuggingFace
        ``AutoImageProcessor`` that produces the correct input tensors for
        ``model``.

    Raises:
        ValueError: If ``name`` is not in ``PRESETS``.
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {sorted(PRESETS.keys())}"
        )
    preset = PRESETS[name]
    model = AutoModel.from_pretrained(preset.hf_model_id)
    processor = AutoImageProcessor.from_pretrained(preset.hf_model_id)
    return model, processor, preset.embed_dim
