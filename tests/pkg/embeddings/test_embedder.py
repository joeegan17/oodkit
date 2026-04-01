"""Smoke tests for ``oodkit.embeddings.Embedder`` (torch-gated).

These tests verify the API surface and error paths. They do NOT download real
models — tests that need a live backbone are skipped in CI via marks or mocks.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from oodkit.embeddings.embedder import Embedder, _resolve_device


def test_resolve_device_cpu():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_device():
    dev = _resolve_device("auto")
    assert isinstance(dev, torch.device)


def test_invalid_mode_raises():
    """Embedder.fit with bad mode raises before needing a real model."""

    class _FakeModel(torch.nn.Module):
        def forward(self, x):
            return x

    emb = Embedder.__new__(Embedder)
    emb._backbone_name = "dinov3-small"
    emb._device = torch.device("cpu")
    emb._model = _FakeModel()
    emb._processor = None
    emb._embed_dim = 384
    emb._head = None

    with pytest.raises(ValueError, match="mode must be"):
        emb.fit("nonexistent", mode="bogus")


def test_mode_none_is_noop():
    """mode='none' returns self without touching model."""

    emb = Embedder.__new__(Embedder)
    emb._backbone_name = "dinov3-small"
    emb._device = torch.device("cpu")
    emb._model = torch.nn.Linear(2, 2)
    emb._processor = None
    emb._embed_dim = 384
    emb._head = None

    result = emb.fit("nonexistent", mode="none")
    assert result is emb
