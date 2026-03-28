"""
Class-conditional Mahalanobis OOD detector on embeddings.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class Mahalanobis(BaseDetector):
    """Minimum squared Mahalanobis distance to class means with shared covariance.

    Fit estimates per-class means and one pooled covariance (class-centered),
    regularized on the diagonal. Score is the minimum squared distance across
    classes (higher = more OOD).

    ``fit`` uses ``features.embeddings``; optional ``y`` gives class labels.
    If ``y`` is omitted, a warning is printed and a single Gaussian is fit.

    Note:
        Distance-based scores often work better with metric-aware embeddings
        (e.g. contrastive training).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialize with covariance regularization.

        Args:
            eps: Ridge added to the covariance diagonal for inversion stability.

        Raises:
            ValueError: If ``eps <= 0``.
        """
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def fit(
        self,
        features_train: "Features",
        y: Optional[ArrayLike] = None,
        **kwargs: object,
    ) -> "Mahalanobis":
        """Fit class means and shared inverse covariance on ID embeddings.

        Args:
            features_train: Must provide ``embeddings``, shape ``(n_samples, n_features)``.
            y: Training labels, shape ``(n_samples,)``. If ``None``, all samples
                are treated as one class (not recommended).
            **kwargs: Unused.

        Returns:
            ``self``.

        Raises:
            ValueError: If embeddings are missing, wrong shape, too few samples,
                or ``y`` length mismatches.
        """
        if features_train.embeddings is None:
            raise ValueError("Mahalanobis.fit requires Features.embeddings")
        embeddings = to_numpy(features_train.embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {embeddings.shape}")

        n_samples, n_features = embeddings.shape
        if n_samples < 2:
            raise ValueError("Mahalanobis.fit requires at least 2 samples")

        if y is None:
            print(
                "Mahalanobis.fit: y is None, fitting a single Gaussian. "
                "Class-conditional labels are recommended for best results."
            )
            y_np = np.zeros(n_samples, dtype=np.int64)
        else:
            y_np = np.asarray(to_numpy(y)).reshape(-1)
            if y_np.shape[0] != n_samples:
                raise ValueError(
                    f"y must have shape [N], got {y_np.shape}, expected ({n_samples},)"
                )

        classes, inverse = np.unique(y_np, return_inverse=True)
        n_classes = classes.shape[0]

        class_means = np.empty((n_classes, n_features), dtype=np.float64)
        for class_index in range(n_classes):
            mask = inverse == class_index
            class_means[class_index] = embeddings[mask].mean(axis=0)

        centered = embeddings - class_means[inverse]
        denom = max(n_samples - n_classes, 1)
        covariance = (centered.T @ centered) / float(denom)
        covariance += self.eps * np.eye(n_features, dtype=np.float64)
        inverse_covariance = np.linalg.pinv(covariance)

        self.classes_ = classes
        self.class_means_ = class_means
        self.inverse_covariance_ = inverse_covariance
        self.n_features_in_ = n_features
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Minimum squared Mahalanobis distance to any class mean.

        Args:
            features_test: ``embeddings`` with feature dim matching ``fit``.
            **kwargs: Unused.

        Returns:
            Distances, shape ``(n_samples,)``.

        Raises:
            RuntimeError: If not fitted.
            ValueError: If embeddings are missing or wrong shape.
        """
        self._check_is_fitted()

        if features_test.embeddings is None:
            raise ValueError("Mahalanobis.score requires Features.embeddings")
        embeddings = to_numpy(features_test.embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {embeddings.shape}")
        if embeddings.shape[1] != self.n_features_in_:
            raise ValueError(
                "embeddings feature dimension mismatch: "
                f"expected {self.n_features_in_}, got {embeddings.shape[1]}"
            )

        deltas = embeddings[:, None, :] - self.class_means_[None, :, :]
        projected = np.einsum("ncf,fg->ncg", deltas, self.inverse_covariance_)
        squared_distances = np.einsum("ncg,ncg->nc", projected, deltas)
        return np.min(squared_distances, axis=1)

    def predict(
        self,
        features: "Features",
        threshold: float,
        **kwargs: object,
    ) -> ArrayLike:
        """Predict OOD (1) when ``score > threshold``.

        Args:
            features: Embeddings for ``score()``.
            threshold: Validation-tuned cutoff.
            **kwargs: Forwarded to ``score()``.

        Returns:
            Labels ``{0, 1}``, shape ``(n_samples,)``.
        """
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)

    def _check_is_fitted(self) -> None:
        """Raise if ``fit`` has not set model attributes."""
        missing = [
            name
            for name in ("classes_", "class_means_", "inverse_covariance_", "n_features_in_")
            if not hasattr(self, name)
        ]
        if missing:
            raise RuntimeError(f"Mahalanobis instance is not fitted yet. Missing: {missing}")
