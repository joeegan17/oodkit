"""
ViM (Virtual Logit Matching) OOD detector.

Paper: https://arxiv.org/pdf/2203.10807
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


def _select_principal_subspace_dim(
    evals_desc: np.ndarray,
    n_features: int,
    n_components: Optional[int],
    pct_variance: float,
) -> int:
    """Number of leading eigen-directions to discard (explicit count or variance rule)."""
    if n_components is not None:
        k = int(n_components)
        if k < 0 or k >= n_features:
            raise ValueError(
                f"n_components must satisfy 0 <= n_components < {n_features}, got {k}"
            )
        return k
    if not (0.0 < pct_variance <= 1.0):
        raise ValueError("pct_variance must be in (0, 1]")
    total = float(np.sum(evals_desc))
    if total <= 0:
        return 1
    cum_ratio = np.cumsum(evals_desc) / total
    k = int(np.searchsorted(cum_ratio, pct_variance, side="left")) + 1
    return int(max(1, min(k, n_features - 1)))


class ViM(BaseDetector):
    """Virtual logit matching using residual norm in the PCA complement.

    Principal subspace is defined from the covariance of **origin-centered** ID
    embeddings (same eigen ordering convention as ``PCAFusion``); ``n_components``
    is how many leading directions are **discarded** before residual norms are taken.

    Requires ``features.embeddings`` and ``features.logits`` for ``fit`` and ``score``.
    """

    def __init__(
        self,
        W: ArrayLike,
        b: ArrayLike,
        n_components: Optional[int] = None,
        pct_variance: float = 0.95,
    ) -> None:
        """Build detector head and subspace hyperparameters.

        Args:
            W: Classifier weights, shape ``(n_classes, n_features)``.
            b: Bias, shape ``(n_classes,)``.
            n_components: If set, discard this many leading principal directions.
                Overrides ``pct_variance``.
            pct_variance: If ``n_components`` is ``None``, choose smallest ``k`` so the
                top ``k`` eigenvalues explain at least this fraction of variance on
                origin-centered training embeddings.

        Raises:
            ValueError: If ``W`` or ``b`` shapes are invalid (see ``compute_origin``).
        """
        self.W = to_numpy(W)
        self.b = to_numpy(b)
        self.n_components = n_components
        self.pct_variance = float(pct_variance)
        self.origin = self.compute_origin(self.W, self.b)

    def fit(
        self,
        features_train: "Features",
        **kwargs: object,
    ) -> "ViM":
        """Fit residual subspace and scaling ``alpha`` on ID data.

        Args:
            features_train: ``embeddings`` ``(n_samples, n_features)`` and ``logits``
                ``(n_samples, n_classes)``.
            **kwargs: Unused.

        Returns:
            ``self``.

        Raises:
            ValueError: If inputs are missing, wrong shape, or feature dim ``< 2``.
        """
        embeddings = to_numpy(features_train.embeddings)
        logits = to_numpy(features_train.logits)

        X_centered = self.center_features(embeddings, self.origin)
        _, n_features = X_centered.shape
        if n_features < 2:
            raise ValueError("ViM.fit requires at least 2 feature dimensions")

        cov = X_centered.T @ X_centered
        evals, _ = np.linalg.eigh(cov)
        evals_desc = evals[::-1]
        k = _select_principal_subspace_dim(
            evals_desc,
            n_features,
            self.n_components,
            self.pct_variance,
        )
        self.n_components_fitted_ = k
        self.R = self.get_residual_projector(X_centered, k)
        residual_norms = self.compute_residual_norms(X_centered, self.R)
        self.alpha = self.compute_alpha(logits, residual_norms)
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Per-sample virtual-class softmax probability (higher ⇒ more OOD in typical use).

        Args:
            features_test: ``embeddings`` and ``logits`` after ``fit``.
            **kwargs: Unused.

        Returns:
            Probabilities in ``[0, 1]``, shape ``(n_samples,)``.

        Raises:
            RuntimeError: If not fitted.
        """
        self._check_is_fitted()

        embeddings = to_numpy(features_test.embeddings)
        logits = to_numpy(features_test.logits)

        X_centered = self.center_features(embeddings, self.origin)
        residual_norms = self.compute_residual_norms(X_centered, self.R)
        return self.compute_vim_score(logits, residual_norms, self.alpha)

    def predict(
        self,
        features: "Features",
        threshold: float = 0.5,
        **kwargs: object,
    ) -> ArrayLike:
        """OOD (1) when ``score > threshold`` (scores are virtual-class probabilities).

        Args:
            features: Embeddings and logits for ``score()``.
            threshold: Default ``0.5``; tune for your calibration.
            **kwargs: Forwarded to ``score()``.

        Returns:
            Labels ``{0, 1}``, shape ``(n_samples,)``.
        """
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)

    @staticmethod
    def compute_origin(W: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Minimum-norm point ``o`` with ``W @ o + b = 0`` (logits at origin).

        Args:
            W: Shape ``(n_classes, n_features)``.
            b: Shape ``(n_classes,)``.

        Returns:
            Origin ``o``, shape ``(n_features,)``.

        Raises:
            ValueError: If shapes are not 2D/1D compatible.
        """
        if W.ndim != 2:
            raise ValueError(f"W must have shape [C, F], got {W.shape}")
        if b.ndim != 1 or b.shape[0] != W.shape[0]:
            raise ValueError(
                f"b must have shape [C], got {b.shape}, expected ({W.shape[0]},)"
            )

        return -np.linalg.pinv(W) @ b

    @staticmethod
    def center_features(X: ArrayLike, o: ArrayLike) -> ArrayLike:
        """Subtract ViM origin from each row.

        Args:
            X: Embeddings, shape ``(n_samples, n_features)``.
            o: Origin, shape ``(n_features,)``.

        Returns:
            ``X - o``.

        Raises:
            ValueError: If shapes disagree.
        """
        if X.ndim != 2:
            raise ValueError(f"X must have shape [N, F], got {X.shape}")
        if o.ndim != 1 or o.shape[0] != X.shape[1]:
            raise ValueError(f"o must have shape [F], got {o.shape}, expected ({X.shape[1]},)")
        return X - o

    def _check_is_fitted(self) -> None:
        """Require ``R`` and ``alpha`` from ``fit``."""
        missing = [name for name in ("R", "alpha") if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"ViM instance is not fitted yet. Missing: {missing}")

    @staticmethod
    def get_residual_projector(X_centered: ArrayLike, k: int) -> ArrayLike:
        """Orthonormal basis for the complement of the top-``k`` principal directions.

        Args:
            X_centered: Origin-centered features, shape ``(n_samples, n_features)``.
            k: Number of leading eigenvectors of ``X_centered.T @ X_centered`` to remove.

        Returns:
            Matrix ``R`` with shape ``(n_features, n_features - k)``.

        Raises:
            ValueError: If ``X_centered`` is not 2D or ``k`` not in ``[0, F]``.
        """
        if X_centered.ndim != 2:
            raise ValueError(f"X_centered must have shape [N, F], got {X_centered.shape}")

        _, F = X_centered.shape
        if not (0 <= k <= F):
            raise ValueError(f"k must be between 0 and F, got {k}")

        cov = X_centered.T @ X_centered
        _, eigvecs = np.linalg.eigh(cov)

        eigvecs = eigvecs[:, ::-1]

        return eigvecs[:, k:]

    @staticmethod
    def compute_residual_norms(X_centered: ArrayLike, R: ArrayLike) -> ArrayLike:
        """Euclidean norm of each row projected onto columns of ``R``.

        Args:
            X_centered: Shape ``(n_samples, n_features)``.
            R: Residual basis, shape ``(n_features, n_residual)``.

        Returns:
            Norms, shape ``(n_samples,)``.

        Raises:
            ValueError: If shapes are invalid.
        """
        if X_centered.ndim != 2:
            raise ValueError(f"X_centered must have shape [N, F], got {X_centered.shape}")
        if R.ndim != 2 or R.shape[0] != X_centered.shape[1]:
            raise ValueError(
                f"R must have shape [F, K], got {R.shape}, expected first dim {X_centered.shape[1]}"
            )

        residuals = X_centered @ R
        return np.linalg.norm(residuals, axis=1)

    @staticmethod
    def compute_alpha(
        logits: ArrayLike,
        residual_norms: ArrayLike,
        eps: float = 1e-12,
    ) -> float:
        """Scale ``mean(max logit) / mean(residual norm)`` for virtual logit.

        Args:
            logits: Shape ``(n_samples, n_classes)``.
            residual_norms: Shape ``(n_samples,)``.
            eps: Denominator stabilization.

        Returns:
            Scalar ``alpha``.

        Raises:
            ValueError: If batch sizes or ranks disagree.
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {logits.shape}")
        if residual_norms.ndim != 1 or residual_norms.shape[0] != logits.shape[0]:
            raise ValueError(
                f"residual_norms must have shape [N], got {residual_norms.shape}, expected ({logits.shape[0]},)"
            )

        max_logits = np.max(logits, axis=1)
        return np.mean(max_logits) / (np.mean(residual_norms) + eps)

    @staticmethod
    def compute_vim_score(
        logits: ArrayLike,
        residual_norms: ArrayLike,
        alpha: float,
    ) -> ArrayLike:
        """Softmax probability of the virtual class with logit ``alpha * residual_norm``.

        Args:
            logits: Shape ``(n_samples, n_classes)``.
            residual_norms: Shape ``(n_samples,)``.
            alpha: Virtual-logit scale from ``compute_alpha``.

        Returns:
            Last column of softmax on augmented logits, shape ``(n_samples,)``.

        Raises:
            ValueError: If shapes disagree.
        """
        alpha = float(alpha)

        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {logits.shape}")
        if residual_norms.ndim != 1 or residual_norms.shape[0] != logits.shape[0]:
            raise ValueError(
                f"residual_norms must have shape [N], got {residual_norms.shape}, expected ({logits.shape[0]},)"
            )

        vim_logit = (alpha * residual_norms)[:, None]
        augmented_logits = np.concatenate([logits, vim_logit], axis=1)

        shifted = augmented_logits - np.max(augmented_logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        softmax = exps / np.sum(exps, axis=1, keepdims=True)

        return softmax[:, -1]
