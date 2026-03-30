"""
PCA-based OOD detection via reconstruction error (linear, cosine / CoP, RFF-cosine / CoRP).

Kernel PCA for OOD (CoP, CoRP): Fang et al., NeurIPS 2024.
https://proceedings.neurips.cc/paper_files/paper/2024/file/f2543511e5f4d4764857f9ad833a977d-Paper-Conference.pdf

Guan et al. ICCV 2023 PCA+energy fusion is implemented separately as ``PCAFusion``;
this module uses **reconstruction error only** (higher = more OOD).
"""

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.detectors.pca_common import fit_pca_subspace, reconstruction_errors_batch
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features

KernelArg = Literal["linear", "cosine", "rff_cosine"]


class PCA(BaseDetector):
    """Reconstruction-error OOD score in a PCA subspace (optionally kernelized).

    ``kernel="linear"``: mean-centered PCA on embeddings.

    ``kernel="cosine"``: CoP — PCA on mean-centered **row-normalized** embeddings
    (cosine feature map).

    ``kernel="rff_cosine"``: CoRP — RFF (Gaussian) features of normalized embeddings,
    then PCA in RFF space.

    ``fit`` and ``score`` use ``features.embeddings`` only. Higher scores indicate
    more OOD (larger reconstruction error).
    """

    def __init__(
        self,
        kernel: KernelArg = "linear",
        n_components: Optional[int] = None,
        pct_variance: float = 0.95,
        eps: float = 1e-12,
        rff_dim: int = 256,
        rff_gamma: float = 1.0,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Initialize PCA / kernel-PCA detector.

        Args:
            kernel: ``"linear"``, ``"cosine"`` (CoP), or ``"rff_cosine"`` (CoRP).
            n_components: Fixed number of components; if ``None``, ``pct_variance``
                selects ``k``.
            pct_variance: Variance threshold when ``n_components`` is ``None``.
            eps: Small constant for numerical stability (reserved for extensions).
            rff_dim: RFF feature dimension for ``rff_cosine`` (must be ``>= 2``).
            rff_gamma: RBF ``gamma`` in RFF for ``rff_cosine``.
            random_state: Seed or ``Generator`` for RFF weights (``rff_cosine`` only).

        Raises:
            ValueError: If ``eps <= 0`` or kernel is unknown.
        """
        if eps <= 0:
            raise ValueError("eps must be positive")
        if kernel not in ("linear", "cosine", "rff_cosine"):
            raise ValueError("kernel must be 'linear', 'cosine', or 'rff_cosine'")
        self.kernel = kernel
        self.n_components = n_components
        self.pct_variance = float(pct_variance)
        self.eps = float(eps)
        self.rff_dim = int(rff_dim)
        self.rff_gamma = float(rff_gamma)
        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)

    def fit(
        self,
        features_train: "Features",
        **kwargs: object,
    ) -> "PCA":
        """Fit PCA subspace on in-distribution embeddings.

        Args:
            features_train: Must provide ``embeddings`` with at least two samples and
                two feature dimensions (and for ``rff_cosine``, ``rff_dim >= 2``).
            **kwargs: Unused.

        Returns:
            ``self``.

        Raises:
            ValueError: If embeddings are missing or invalid.
        """
        if features_train.embeddings is None:
            raise ValueError("PCA.fit requires Features.embeddings")
        X = to_numpy(features_train.embeddings)
        if X.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {X.shape}")

        self._state = fit_pca_subspace(
            X,
            self.kernel,
            self.n_components,
            self.pct_variance,
            self.rff_dim,
            self.rff_gamma,
            self._rng,
        )
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Per-sample reconstruction error norms (higher = more OOD).

        Args:
            features_test: Must provide ``embeddings`` with feature dim matching ``fit``.
            **kwargs: Unused.

        Returns:
            Scores, shape ``(n_samples,)``, ``float64``.

        Raises:
            RuntimeError: If not fitted.
            ValueError: If embeddings are missing or wrong shape.
        """
        self._check_is_fitted()
        if features_test.embeddings is None:
            raise ValueError("PCA.score requires Features.embeddings")
        h = to_numpy(features_test.embeddings)
        return reconstruction_errors_batch(self._state, h).astype(np.float64)

    def predict(
        self,
        features: "Features",
        threshold: float,
        **kwargs: object,
    ) -> ArrayLike:
        """OOD (1) when ``score > threshold``."""
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)

    @property
    def n_components_fitted_(self) -> int:
        """Number of principal components retained after ``fit``."""
        self._check_is_fitted()
        return self._state.n_components_fitted_

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_state"):
            raise RuntimeError("PCA instance is not fitted yet.")
