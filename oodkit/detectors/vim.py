"""
ViM (Virtual Logit Matching) OOD detector.

Uses both logits and embeddings from Features.
"""

from typing import TYPE_CHECKING

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike

if TYPE_CHECKING:
    from oodkit.data.features import Features


class ViM(BaseDetector):
    """
    ViM detector.

    Expected inputs
    ----------------
    - `features.logits` for classification logits, shape `(n_samples, n_classes)`
    - `features.embeddings` for feature embeddings, shape `(n_samples, n_features)`

    """

    def __init__(self, W: ArrayLike, b: ArrayLike, D: int) -> None:
        """
        Parameters
        ----------
        W : ArrayLike
            Classifier weights used by ViM, shape `(n_classes, n_features)`.
        b : ArrayLike
            Classifier bias term from the classifier head, shape `(n_classes,)`.
        D : int
            Number of principal directions to discard.
        """
        self.W = self._to_numpy(W)
        self.b = self._to_numpy(b)
        self.D = D
        # ViM origin depends only on W and b, so compute once at construction time.
        self.origin = self.compute_origin(self.W, self.b)

    def fit(
        self,
        features_train: "Features",
        **kwargs: object,
    ) -> "ViM":
        """
        Fit ViM components from training data.

        Parameters
        ----------
        features_train : Features
            Container with at least:
            - `features_train.embeddings`: shape `(n_samples, n_features)`
            - `features_train.logits`: shape `(n_samples, n_classes)`
        **kwargs : object
            Reserved for future detector-specific options.

        Returns
        -------
        self : ViM
            The fitted detector instance.
        """
        # Convert once at boundary; internals operate on NumPy arrays.
        embeddings = self._to_numpy(features_train.embeddings)
        logits = self._to_numpy(features_train.logits)

        X_centered = self.center_features(embeddings, self.origin)
        self.R = self.get_residual_projector(X_centered, self.D)
        residual_norms = self.compute_residual_norms(X_centered, self.R)
        self.alpha = self.compute_alpha(logits, residual_norms)
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """
        Compute per-sample ViM OOD scores.

        Parameters
        ----------
        features_test : Features
            Container with:
            - `features_test.embeddings`: shape `(n_samples, n_features)`
            - `features_test.logits`: shape `(n_samples, n_classes)`
            The detector must have been fitted beforehand so that internal
            parameters (e.g. origin/subspace/scalar) are available.
        **kwargs : object
            Detector-specific options (reserved for future use).

        Returns
        -------
        scores : ArrayLike
            Per-sample ViM OOD scores, shape `(n_samples,)`.
        """
        self._check_is_fitted()

        # Convert once at boundary; internals operate on NumPy arrays.
        embeddings = self._to_numpy(features_test.embeddings)
        logits = self._to_numpy(features_test.logits)

        X_centered = self.center_features(embeddings, self.origin)
        residual_norms = self.compute_residual_norms(X_centered, self.R)
        return self.compute_vim_score(logits, residual_norms, self.alpha)

    def predict(
        self,
        features: "Features",
        threshold: float = 0.5,
        **kwargs: object,
    ) -> ArrayLike:
        """
        Predict binary ID (0) / OOD (1) labels from ViM scores.

        Parameters
        ----------
        features : Features
            Container with `features.logits` and `features.embeddings`.
        threshold : float, default=0.5
            If `score > threshold` then predict OOD (1), else ID (0).
        **kwargs : object
            Passed through to `score()`.

        Returns
        -------
        labels : ArrayLike
            Shape `(n_samples,)`, with values in `{0, 1}`.
        """
        scores = self.score(features, **kwargs)
        # ViM score is computed as a probability in [0, 1] (virtual-class softmax).
        return (scores > threshold).astype(int)

    @staticmethod
    def compute_origin(W: ArrayLike, b: ArrayLike) -> ArrayLike:
        """
        Compute the ViM origin in feature space.

        Parameters
        ----------
        W : ArrayLike
            Classifier weights, shape `(n_classes, n_features)`.
        b : ArrayLike
            Classifier bias term, shape `(n_classes,)`.

        Returns
        -------
        o : ArrayLike
            Origin vector in feature space, shape `(n_features,)`.
        """
        if W.ndim != 2:
            raise ValueError(f"W must have shape [C, F], got {W.shape}")
        if b.ndim != 1 or b.shape[0] != W.shape[0]:
            raise ValueError(
                f"b must have shape [C], got {b.shape}, expected ({W.shape[0]},)"
            )

        # With logits l = W x + b and W shape (C, F), choose the minimum-norm
        # origin o that satisfies W o + b = 0.
        return -np.linalg.pinv(W) @ b

    @staticmethod
    def center_features(X: ArrayLike, o: ArrayLike) -> ArrayLike:
        """
        Center features by subtracting the ViM origin.

        Parameters
        ----------
        X : ArrayLike
            Feature embeddings, shape `(n_samples, n_features)`.
        o : ArrayLike
            Origin vector, shape `(n_features,)`.

        Returns
        -------
        X_centered : ArrayLike
            Centered features, shape `(n_samples, n_features)`.
        """
        if X.ndim != 2:
            raise ValueError(f"X must have shape [N, F], got {X.shape}")
        if o.ndim != 1 or o.shape[0] != X.shape[1]:
            raise ValueError(f"o must have shape [F], got {o.shape}, expected ({X.shape[1]},)")
        return X - o

    def _check_is_fitted(self) -> None:
        """
        sklearn-like fittedness check.

        Raises
        ------
        RuntimeError
            If `fit()` has not been called yet.
        """
        # `origin` is computed at construction time (from W/b); training-dependent
        # fitted state is `R` and `alpha`.
        missing = [name for name in ("R", "alpha") if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"ViM instance is not fitted yet. Missing: {missing}")

    @staticmethod
    def get_residual_projector(X_centered: ArrayLike, D: int) -> ArrayLike:
        """
        Get basis vectors for the residual subspace (orthogonal complement
        of the top-D principal subspace).

        Parameters
        ----------
        X_centered : ArrayLike
            Centered training features, shape `(n_samples, n_features)`.
        D : int
            Number of top principal directions to discard.

        Returns
        -------
        R : ArrayLike
            Residual subspace basis, shape `(n_features, n_features - D)`.
        """
        if X_centered.ndim != 2:
            raise ValueError(f"X_centered must have shape [N, F], got {X_centered.shape}")

        _, F = X_centered.shape
        if not (0 <= D <= F):
            raise ValueError(f"D must be between 0 and F, got {D}")

        cov = X_centered.T @ X_centered  # No need to scale since we only need the eigenvectors
        _, eigvecs = np.linalg.eigh(cov)

        # eigh returns eigenvectors in ascending order of eigenvalues, so we need to reverse the order
        eigvecs = eigvecs[:, ::-1]

        # Residual subspace = orthogonal complement of the top-D eigenvectors
        return eigvecs[:, D:]

    @staticmethod
    def compute_residual_norms(X_centered: ArrayLike, R: ArrayLike) -> ArrayLike:
        """
        Compute residual norms ||x_P⊥|| for each sample.

        Parameters
        ----------
        X_centered : ArrayLike
            Centered features, shape `(n_samples, n_features)`.
        R : ArrayLike
            Residual subspace basis, shape `(n_features, k)`.

        Returns
        -------
        residual_norms : ArrayLike
            Residual norms per sample, shape `(n_samples,)`.
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
        """
        Compute alpha = mean(max logit) / mean(residual norm)

        Parameters
        ----------
        logits : ArrayLike
            Classifier logits, shape `(n_samples, n_classes)`.
        residual_norms : ArrayLike
            Residual norms, shape `(n_samples,)`.
        eps : float
            Small constant for numerical stability.

        Returns
        -------
        alpha : float
            Scalar computed from logits and residual norms.
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
        """
        Compute ViM probability by appending alpha * residual_norm as
        a virtual logit and applying softmax.

        Parameters
        ----------
        logits : ArrayLike
            Classifier logits, shape `(n_samples, n_classes)`.
        residual_norms : ArrayLike
            Residual norms, shape `(n_samples,)`.
        alpha : float
            Scalar scale factor.

        Returns
        -------
        vim_scores : ArrayLike
            Probability of the virtual class, shape `(n_samples,)`.
        """
        alpha = float(alpha)

        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {logits.shape}")
        if residual_norms.ndim != 1 or residual_norms.shape[0] != logits.shape[0]:
            raise ValueError(
                f"residual_norms must have shape [N], got {residual_norms.shape}, expected ({logits.shape[0]},)"
            )

        vim_logit = (alpha * residual_norms)[:, None]   # [N, 1]
        augmented_logits = np.concatenate([logits, vim_logit], axis=1)  # [N, C+1]

        # Numerically stable softmax
        shifted = augmented_logits - np.max(augmented_logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        softmax = exps / np.sum(exps, axis=1, keepdims=True)

        return softmax[:, -1]

    @staticmethod
    def _to_numpy(x: ArrayLike) -> np.ndarray:
        """
        Convert supported inputs to NumPy arrays.

        Accepts NumPy arrays directly and torch tensors via `detach().cpu().numpy()`.
        """
        if x is None:
            raise ValueError("Expected array-like input, got None")
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.detach().cpu().numpy()
        return np.asarray(x)
