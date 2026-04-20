"""Smoke tests for ``oodkit.embeddings.Embedder`` (torch-gated).

These tests verify the API surface and error paths. They do NOT download real
models — tests that need a live backbone are skipped in CI via marks or mocks.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from oodkit.embeddings.embedder import Embedder, _resolve_device
from oodkit.embeddings.result import EmbeddingResult
from oodkit.embeddings.training import train_full


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
    emb._backbone_name = "dinov2-small"
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
    emb._backbone_name = "dinov2-small"
    emb._device = torch.device("cpu")
    emb._model = torch.nn.Linear(2, 2)
    emb._processor = None
    emb._embed_dim = 384
    emb._head = None

    result = emb.fit("nonexistent", mode="none")
    assert result is emb


def test_fit_full_rejects_nonpositive_backbone_lr_ratio():
    """Invalid backbone_lr_ratio fails before dataset resolution."""

    emb = Embedder.__new__(Embedder)
    emb._backbone_name = "dinov2-small"
    emb._device = torch.device("cpu")
    emb._model = torch.nn.Linear(2, 2)
    emb._processor = None
    emb._embed_dim = 384
    emb._head = None

    with pytest.raises(ValueError, match="backbone_lr_ratio"):
        emb.fit(
            "nonexistent",
            mode="full",
            backbone_lr_ratio=0.0,
        )


def test_fit_extract_forwards_backbone_lr_ratio():
    emb = Embedder.__new__(Embedder)
    captured: dict = {}

    def _fake_fit(dataset, mode="none", **kwargs):
        captured.update(kwargs)
        return emb

    def _fake_extract(dataset, **kwargs):
        return EmbeddingResult(embeddings=np.zeros((1, 2), dtype=np.float32))

    emb.fit = _fake_fit
    emb.extract = _fake_extract

    emb.fit_extract(
        "dummy",
        mode="full",
        lr=2e-3,
        backbone_lr_ratio=0.25,
        batch_size=8,
    )
    assert captured.get("lr") == 2e-3
    assert captured.get("backbone_lr_ratio") == 0.25


def test_train_full_adam_param_groups():
    class _FakeBackbone(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return SimpleNamespace(
                last_hidden_state=torch.zeros(b, 2, 4, device=x.device, dtype=x.dtype),
            )

    backbone = _FakeBackbone()
    head = torch.nn.Linear(4, 3)
    images = torch.randn(6, 3, 8, 8)
    labels = torch.randint(0, 3, (6,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels),
        batch_size=3,
    )

    adam_calls: list = []
    real_adam = torch.optim.Adam

    def _capture_adam(*args, **kwargs):
        adam_calls.append((args, kwargs))
        return real_adam(*args, **kwargs)

    with mock.patch.object(torch.optim, "Adam", side_effect=_capture_adam):
        train_full(
            backbone,
            head,
            loader,
            epochs=1,
            lr=1e-3,
            device=torch.device("cpu"),
            backbone_lr_ratio=0.2,
        )

    assert len(adam_calls) == 1
    param_groups = adam_calls[0][0][0]
    assert len(param_groups) == 2
    assert param_groups[0]["lr"] == pytest.approx(2e-4)
    assert param_groups[1]["lr"] == pytest.approx(1e-3)


def test_train_full_backbone_lr_ratio_must_be_positive():
    class _FakeBackbone(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return SimpleNamespace(
                last_hidden_state=torch.zeros(b, 2, 4, device=x.device, dtype=x.dtype),
            )

    backbone = _FakeBackbone()
    head = torch.nn.Linear(4, 3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(4, 3, 8, 8),
            torch.randint(0, 3, (4,)),
        ),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="backbone_lr_ratio"):
        train_full(
            backbone,
            head,
            loader,
            epochs=1,
            lr=1e-3,
            device=torch.device("cpu"),
            backbone_lr_ratio=-0.1,
        )


def test_extract_chip_metadata_returns_arrays_for_chip_dataset():
    ds = SimpleNamespace(
        chip_to_image=np.array([0, 0, 1], dtype=np.int64),
        boxes=np.array(
            [[0, 0, 10, 10], [5, 5, 20, 20], [0, 0, 8, 8]], dtype=np.float64
        ),
    )
    out = Embedder._extract_chip_metadata(ds)
    assert out is not None
    np.testing.assert_array_equal(out["chip_to_image"], [0, 0, 1])
    assert out["boxes"].shape == (3, 4)
    assert out["chip_to_image"].dtype == np.int64
    assert out["boxes"].dtype == np.float64


def test_extract_chip_metadata_none_for_plain_dataset():
    ds = SimpleNamespace(imgs=[("/a", 0)])
    assert Embedder._extract_chip_metadata(ds) is None


def test_extract_chip_metadata_rejects_length_mismatch():
    ds = SimpleNamespace(
        chip_to_image=np.array([0, 1], dtype=np.int64),
        boxes=np.zeros((3, 4), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="length mismatch"):
        Embedder._extract_chip_metadata(ds)


def test_extract_chip_metadata_rejects_bad_boxes_shape():
    ds = SimpleNamespace(
        chip_to_image=np.array([0], dtype=np.int64),
        boxes=np.zeros((1, 3), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="boxes"):
        Embedder._extract_chip_metadata(ds)
