"""
K-nearest neighbors (KNN) OOD detector on embeddings.

Uses embeddings only. Fit stores an in-distribution embedding bank; score is
the average Euclidean distance to the k nearest neighbors in that bank.

Note
----
These distance-based methods often perform better when embeddings come from
contrastive objectives (or similarly metric-aware training), because in-class
neighbors become more semantically meaningful.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class KNN(BaseDetector):
    """
    KNN detector over embedding space.

    Expected inputs
    ---------------
    - `features.embeddings`, shape `(n_samples, n_features)`
    """

    def __init__(
        self,
        k: int = 10,
        backend: Literal["auto", "brute", "sklearn"] = "auto",
        metric: str = "euclidean",
    ) -> None:
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
        if features_train.embeddings is None:
            raise ValueError("KNN.fit requires Features.embeddings")
        train_embeddings = to_numpy(features_train.embeddings)
        if train_embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {train_embeddings.shape}")
        if train_embeddings.shape[0] < self.k:
            raise ValueError(
                f"k ({self.k}) cannot exceed number of training samples ({train_embeddings.shape[0]})"
            )

        self.train_embeddings_ = train_embeddings.astype(np.float64, copy=False)
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
            return distances.mean(axis=1)
        return self._score_bruteforce(query_embeddings)

    def predict(
        self,
        features: "Features",
        threshold: float,
        **kwargs: object,
    ) -> ArrayLike:
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)

    def _check_is_fitted(self) -> None:
        missing = [name for name in ("train_embeddings_", "n_features_in_", "backend_") if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"KNN instance is not fitted yet. Missing: {missing}")

    def _resolve_backend(self) -> Literal["brute", "sklearn"]:
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

        # auto: prefer sklearn if available, otherwise pure NumPy brute force.
        try:
            import sklearn  # noqa: F401
        except ImportError:
            print("KNN.fit: sklearn not available, using manual approach with euclidean distance.")
            return "brute"
        return "sklearn"

    def _score_bruteforce(self, query_embeddings: np.ndarray) -> np.ndarray:
        # Efficient pairwise L2 distances via ||x-y||^2 = ||x||^2 + ||y||^2 - 2x@y.
        train_sq = np.sum(self.train_embeddings_ ** 2, axis=1)[None, :]
        query_sq = np.sum(query_embeddings ** 2, axis=1)[:, None]
        pairwise_sq = query_sq + train_sq - 2.0 * (query_embeddings @ self.train_embeddings_.T)
        np.maximum(pairwise_sq, 0.0, out=pairwise_sq)
        pairwise = np.sqrt(pairwise_sq)

        nearest_k = np.partition(pairwise, kth=self.k - 1, axis=1)[:, : self.k]
        return nearest_k.mean(axis=1)
