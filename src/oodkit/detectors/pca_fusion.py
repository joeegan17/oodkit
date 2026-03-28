"""
PCA reconstruction fused with log-sum-exp of logits (Guan et al., ICCV 2023).

Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Guan_Revisit_PCA-based_Technique_for_Out-of-Distribution_Detection_ICCV_2023_paper.html
PDF: https://openaccess.thecvf.com/content/ICCV2023/papers/Guan_Revisit_PCA-based_Technique_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf
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
    """Resolve principal subspace dimension ``k`` from explicit count or variance threshold."""
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


class PCAFusion(BaseDetector):
    """Guan et al. (ICCV 2023) PCA reconstruction error fused with log-sum-exp logits.

    Fit uses mean-centered ID embeddings to build a top-``k`` PCA subspace.
    Scores follow the paper up to sign: this library returns ``-D`` so that
    **higher score means more OOD**, consistent with other detectors (the paper
    associates larger ``D`` with ID-like behavior).

    ``fit`` requires ``features.embeddings`` with shape ``(n_samples, n_features)``.
    ``score`` and ``predict`` require both ``embeddings`` and ``logits``.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        pct_variance: float = 0.95,
        temperature: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        """Configure PCA subspace selection and fusion numerics.

        Args:
            n_components: Fixed number of principal components to retain for
                reconstruction. If ``None``, ``pct_variance`` chooses ``k``.
            pct_variance: When ``n_components`` is ``None``, smallest ``k`` such that
                the top ``k`` eigenvalues explain at least this fraction of variance
                (on mean-centered training embeddings). Default ``0.95``.
            temperature: Logit temperature ``T`` in ``logits / T`` inside log-sum-exp
                (same convention as ``Energy``).
            eps: Added to ``||h||`` in the denominator of the normalized reconstruction
                error for stability.

        Raises:
            ValueError: If ``temperature <= 0`` or ``eps <= 0``.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.n_components = n_components
        self.pct_variance = float(pct_variance)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def fit(
        self,
        features_train: "Features",
        **kwargs: object,
    ) -> "PCAFusion":
        """Fit mean and top-``k`` PCA basis on in-distribution embeddings.

        Args:
            features_train: Must provide ``embeddings`` with at least two samples and
                two feature dimensions.
            **kwargs: Unused; reserved for future options.

        Returns:
            ``self``.

        Raises:
            ValueError: If embeddings are missing, wrong shape, or too small.
        """
        if features_train.embeddings is None:
            raise ValueError("PCAFusion.fit requires Features.embeddings")
        X = to_numpy(features_train.embeddings)
        if X.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {X.shape}")
        n_samples, n_features = X.shape
        if n_features < 2:
            raise ValueError("PCAFusion.fit requires at least 2 feature dimensions")
        if n_samples < 2:
            raise ValueError("PCAFusion.fit requires at least 2 samples")

        self.mean_ = X.mean(axis=0).astype(np.float64, copy=False)
        Xc = X - self.mean_
        cov = Xc.T @ Xc
        evals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, ::-1]
        evals_desc = evals[::-1]

        k = _select_principal_subspace_dim(
            evals_desc,
            n_features,
            self.n_components,
            self.pct_variance,
        )
        self.n_components_fitted_ = k
        self.principal_basis_ = eigvecs[:, :k] if k > 0 else np.zeros((n_features, 0), dtype=np.float64)
        self.n_features_in_ = n_features
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Compute per-sample OOD scores (negated paper ``D``).

        Args:
            features_test: Must provide ``embeddings`` and ``logits`` with matching
                batch size and embedding dim equal to ``n_features_in_`` from ``fit``.
            **kwargs: Unused.

        Returns:
            Scores, shape ``(n_samples,)``, ``float64``.

        Raises:
            RuntimeError: If ``fit`` has not been called.
            ValueError: If required fields are missing or shapes disagree.
        """
        self._check_is_fitted()

        if features_test.embeddings is None:
            raise ValueError("PCAFusion.score requires Features.embeddings")
        if features_test.logits is None:
            raise ValueError("PCAFusion.score requires Features.logits")

        h = to_numpy(features_test.embeddings)
        logits = to_numpy(features_test.logits)
        if h.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {h.shape}")
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {logits.shape}")
        if h.shape[0] != logits.shape[0]:
            raise ValueError(
                f"batch mismatch: embeddings {h.shape[0]} vs logits {logits.shape[0]}"
            )
        if h.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {h.shape[1]}"
            )

        mean = self.mean_
        k = self.n_components_fitted_
        V = self.principal_basis_
        centered = h - mean
        if k == 0:
            h_hat = np.broadcast_to(mean, h.shape).copy()
        else:
            coef = centered @ V
            h_hat = mean + coef @ V.T

        recon_err = np.linalg.norm(h - h_hat, axis=1)
        h_norm = np.linalg.norm(h, axis=1)
        r = np.clip(recon_err / (h_norm + self.eps), 0.0, 1.0)

        T = self.temperature
        scaled = logits / T
        m = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(scaled - m), axis=1)) + m.ravel()
        D = (1.0 - r) * log_sum_exp
        return (-D).astype(np.float64)

    def predict(
        self,
        features: "Features",
        threshold: float,
        **kwargs: object,
    ) -> ArrayLike:
        """Predict OOD where ``score > threshold``.

        Args:
            features: Embeddings and logits for ``score()``.
            threshold: Calibrated cutoff on validation data.
            **kwargs: Forwarded to ``score()``.

        Returns:
            Labels ``{0, 1}``, shape ``(n_samples,)``.
        """
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)

    def _check_is_fitted(self) -> None:
        """Ensure ``fit`` has populated PCA attributes."""
        needed = ("mean_", "principal_basis_", "n_components_fitted_", "n_features_in_")
        missing = [n for n in needed if not hasattr(self, n)]
        if missing:
            raise RuntimeError(f"PCAFusion is not fitted yet. Missing: {missing}")
