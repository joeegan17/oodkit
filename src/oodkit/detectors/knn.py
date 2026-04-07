"""
K-nearest neighbors OOD detector on embeddings.

Paper: https://arxiv.org/pdf/2204.06507
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class KNN(BaseDetector):
    """Distance to the ``k``-th nearest in-distribution embedding neighbor.

    Higher scores indicate more OOD. Requires ``features.embeddings`` only.

    With ``backend="auto"``, uses scikit-learn's ``NearestNeighbors`` when
    installed; otherwise falls back to a NumPy Euclidean implementation (non-Euclidean
    ``metric`` is ignored in that case).

    Note:
        KNN OOD scores often improve with metric-aware embeddings (e.g. contrastive).
    """

    def __init__(
        self,
        k: int = 10,
        backend: Literal["auto", "brute", "sklearn"] = "auto",
        metric: str = "euclidean",
    ) -> None:
        """Configure neighbor count and distance backend.

        Args:
            k: Number of neighbors (must be ``>= 1`` and at most training size at ``fit``).
            backend: ``"sklearn"``, ``"brute"`` (NumPy), or ``"auto"``.
            metric: Passed to scikit-learn when that backend is used.

        Raises:
            ValueError: If ``k < 1``, ``backend`` is invalid, or ``metric`` is empty.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if backend not in {"auto", "brute", "sklearn"}:
            raise ValueError("backend must be one of {'auto', 'brute', 'sklearn'}")
        if not metric:
            raise ValueError("metric must be a non-empty string")
        self.k = int(k)
        self.backend = backend
        self.metric = metric

    def fit(self, features_train: "Features", **kwargs: object) -> "KNN":
        """Store ID embeddings and optionally build sklearn index.

        Args:
            features_train: Must provide ``embeddings`` with at least ``k`` rows.
            **kwargs: Unused.

        Returns:
            ``self``.

        Raises:
            ValueError: If embeddings are missing, wrong shape, or fewer than ``k`` samples.
            ImportError: If ``backend="sklearn"`` and scikit-learn is not installed.
        """
        if features_train.embeddings is None:
            raise ValueError("KNN.fit requires Features.embeddings")
        train_embeddings = to_numpy(features_train.embeddings)
        if train_embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {train_embeddings.shape}")
        if train_embeddings.shape[0] < self.k:
            raise ValueError(
                f"k ({self.k}) cannot exceed number of training samples ({train_embeddings.shape[0]})"
            )

        self.train_embeddings_ = train_embeddings.astype(np.float32, copy=False)
        self.n_features_in_ = train_embeddings.shape[1]
        self.backend_ = self._resolve_backend()

        if self.backend_ == "sklearn":
            from sklearn.neighbors import NearestNeighbors

            self._nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto", metric=self.metric)
            self._nn.fit(self.train_embeddings_)
        elif self.backend == "brute" and self.metric != "euclidean":
            print(
                "KNN.fit: manual approach uses euclidean distance only "
                f"(requested metric='{self.metric}' is ignored)."
            )
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Distance to the ``k``-th nearest neighbor in the training bank.

        Args:
            features_test: ``embeddings`` with same feature dimension as ``fit``.
            **kwargs: Unused.

        Returns:
            Per-sample distance to the ``k``-th nearest neighbor, shape ``(n_samples,)``.

        Raises:
            RuntimeError: If not fitted.
            ValueError: If embeddings are missing or wrong shape.
        """
        self._check_is_fitted()

        if features_test.embeddings is None:
            raise ValueError("KNN.score requires Features.embeddings")
        query_embeddings = to_numpy(features_test.embeddings)
        if query_embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {query_embeddings.shape}")
        if query_embeddings.shape[1] != self.n_features_in_:
            raise ValueError(
                "embeddings feature dimension mismatch: "
                f"expected {self.n_features_in_}, got {query_embeddings.shape[1]}"
            )

        if self.backend_ == "sklearn":
            distances, _ = self._nn.kneighbors(query_embeddings, n_neighbors=self.k)
            # kneighbors returns distances sorted ascending; last column is k-th NN.
            return distances[:, -1]
        return self._score_bruteforce(query_embeddings)

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
        """Raise if ``fit`` has not run."""
        missing = [name for name in ("train_embeddings_", "n_features_in_", "backend_") if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"KNN instance is not fitted yet. Missing: {missing}")

    def _resolve_backend(self) -> Literal["brute", "sklearn"]:
        """Choose sklearn or NumPy neighbor search."""
        if self.backend == "brute":
            return "brute"
        if self.backend == "sklearn":
            try:
                import sklearn  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "backend='sklearn' requested but scikit-learn is not installed"
                ) from exc
            return "sklearn"

        try:
            import sklearn  # noqa: F401
        except ImportError:
            print("KNN.fit: sklearn not available, using manual approach with euclidean distance.")
            return "brute"
        return "sklearn"

    def _score_bruteforce(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Euclidean distance to the k-th nearest neighbor via expanded L2 norm formula."""
        train_sq = np.sum(self.train_embeddings_ ** 2, axis=1)[None, :]
        query_sq = np.sum(query_embeddings ** 2, axis=1)[:, None]
        pairwise_sq = query_sq + train_sq - 2.0 * (query_embeddings @ self.train_embeddings_.T)
        np.maximum(pairwise_sq, 0.0, out=pairwise_sq)
        pairwise = np.sqrt(pairwise_sq)

        partitioned = np.partition(pairwise, kth=self.k - 1, axis=1)
        return partitioned[:, self.k - 1]
