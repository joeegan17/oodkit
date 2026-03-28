"""
Shared pytest fixtures for OODKit tests.

Place module-specific fixtures next to tests (e.g. ``tests/pkg/detectors/conftest.py``)
as the suite grows; keep cross-cutting synthetic data here.

Note: the test mirror lives under ``tests/pkg/`` (not ``tests/oodkit/``) so Python does
not import ``tests/oodkit`` as the real ``oodkit`` package.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from oodkit.data.features import Features


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible RNG for synthetic arrays."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def toy_dims() -> SimpleNamespace:
    """Default layout: batch × classes / features (used by most detector tests)."""
    return SimpleNamespace(n_samples=5, n_classes=3, n_features=4)


@pytest.fixture
def logits_2d(rng: np.random.Generator, toy_dims: SimpleNamespace) -> np.ndarray:
    """Random logits ``(n_samples, n_classes)``."""
    return rng.standard_normal((toy_dims.n_samples, toy_dims.n_classes))


@pytest.fixture
def embeddings_2d(rng: np.random.Generator, toy_dims: SimpleNamespace) -> np.ndarray:
    """Random embeddings ``(n_samples, n_features)``."""
    return rng.standard_normal((toy_dims.n_samples, toy_dims.n_features))


@pytest.fixture
def logits_zeros_small() -> np.ndarray:
    """``(2, 3)`` zeros — cheap batch for no-op ``fit`` / energy checks."""
    return np.zeros((2, 3), dtype=np.float64)


@pytest.fixture
def logits_sharp() -> np.ndarray:
    """Nearly one-hot row; MSP max prob ≈ 1 → score ≈ -1."""
    return np.array([[10.0, 0.0, 0.0]], dtype=np.float64)


@pytest.fixture
def logits_uniform_1x3() -> np.ndarray:
    """Uniform logits ``(1, 3)``; MSP max prob = 1/3."""
    return np.zeros((1, 3), dtype=np.float64)


@pytest.fixture
def logits_msp_predict_batch() -> np.ndarray:
    """Two rows, two classes — for MSP ``predict`` shape checks."""
    return np.array([[100.0, 0.0], [0.0, 0.0]], dtype=np.float64)


@pytest.fixture
def logits_1d_invalid() -> np.ndarray:
    """Invalid shape for detectors expecting ``(N, C)``."""
    return np.zeros(3, dtype=np.float64)


@pytest.fixture
def logits_temperature_compare() -> np.ndarray:
    """Single row for comparing MSP scores across temperatures."""
    return np.array([[2.0, 0.0]], dtype=np.float64)


@pytest.fixture
def features_logits_only(logits_2d: np.ndarray) -> Features:
    return Features(logits=logits_2d)


@pytest.fixture
def features_embeddings_only(embeddings_2d: np.ndarray) -> Features:
    return Features(embeddings=embeddings_2d)


@pytest.fixture
def features_both(logits_2d: np.ndarray, embeddings_2d: np.ndarray) -> Features:
    return Features(logits=logits_2d, embeddings=embeddings_2d)


@pytest.fixture
def features_logits_zeros(logits_zeros_small: np.ndarray) -> Features:
    return Features(logits=logits_zeros_small)


@pytest.fixture
def vim_linear_pack(rng: np.random.Generator) -> SimpleNamespace:
    """
    Consistent ViM toy problem: ``logits ≈ embeddings @ W.T + b``.

    Attributes: ``W``, ``b``, ``n_components``, ``embeddings``, ``logits``,
    ``n_classes``, ``n_features``, ``n_samples``.
    """
    n_classes, n_features, n_samples = 2, 3, 5
    W = rng.standard_normal((n_classes, n_features))
    b = np.array([0.1, -0.1], dtype=np.float64)
    n_components = 1
    embeddings = rng.standard_normal((n_samples, n_features))
    logits = embeddings @ W.T + b
    return SimpleNamespace(
        W=W,
        b=b,
        n_components=n_components,
        embeddings=embeddings,
        logits=logits,
        n_classes=n_classes,
        n_features=n_features,
        n_samples=n_samples,
    )


@pytest.fixture
def vim_origin_W() -> np.ndarray:
    return np.eye(2, 3)


@pytest.fixture
def vim_origin_b() -> np.ndarray:
    return np.array([1.0, 2.0], dtype=np.float64)


@pytest.fixture
def center_X_and_o() -> SimpleNamespace:
    X = np.ones((2, 4), dtype=np.float64)
    o = np.full(4, 0.5, dtype=np.float64)
    return SimpleNamespace(X=X, o=o, expected=np.full((2, 4), 0.5))


@pytest.fixture
def residual_Xc_bad_D(rng: np.random.Generator) -> np.ndarray:
    """Centered features for testing invalid ``D`` in residual projector."""
    return rng.standard_normal((4, 3))


@pytest.fixture
def logits_alpha_vim_score() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)


@pytest.fixture
def residual_norms_alpha_vim() -> np.ndarray:
    return np.array([0.5, 0.5], dtype=np.float64)


@pytest.fixture
def embeddings_two_cluster_train() -> np.ndarray:
    """Two compact 2D clusters for metric detector fitting."""
    return np.array(
        [
            [-1.0, -1.0],
            [-1.1, -0.9],
            [-0.9, -1.1],
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def labels_two_cluster_train() -> np.ndarray:
    """Class labels aligned with ``embeddings_two_cluster_train``."""
    return np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)


@pytest.fixture
def embeddings_near_far_queries() -> np.ndarray:
    """Query points: first near ID manifold, second intentionally far OOD."""
    return np.array(
        [
            [1.05, 0.95],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )
