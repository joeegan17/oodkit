"""
PCA reconstruction fused with log-sum-exp of logits (Guan et al., ICCV 2023).

Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Guan_Revisit_PCA-based_Technique_for_Out-of-Distribution_Detection_ICCV_2023_paper.html
PDF: https://openaccess.thecvf.com/content/ICCV2023/papers/Guan_Revisit_PCA-based_Technique_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf

Kernel variants (linear / cosine CoP / RFF-cosine CoRP) follow Fang et al., NeurIPS 2024
for the reconstruction term; see ``oodkit.detectors.pca`` and ``pca_common``.
https://proceedings.neurips.cc/paper_files/paper/2024/file/f2543511e5f4d4764857f9ad833a977d-Paper-Conference.pdf
"""

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.detectors.pca_common import (
    fit_pca_subspace,
    guan_r,
    reconstruction_errors_batch,
)
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features

KernelArg = Literal["linear", "cosine", "rff_cosine"]


class PCAFusion(BaseDetector):
    """Guan et al. (ICCV 2023) fusion of PCA reconstruction error and log-sum-exp logits.

    Fit uses in-distribution embeddings to build a top-``k`` PCA subspace (linear,
    cosine / CoP, or RFF-cosine / CoRP — same kernels as ``PCA``). The ICCV paper
    defines a score ``D`` where larger values mean more in-distribution; this library
    returns **negative D** so that **higher returned values mean more OOD**, like
    other detectors.

    ``fit`` requires ``features.embeddings``. ``score`` and ``predict`` require
    ``embeddings`` and ``logits``.

    After ``fit``, ``explained_variance_ratio_`` and
    ``cumulative_explained_variance_ratio_`` describe the PCA working spectrum;
    see property docstrings.
    """

    def __init__(
        self,
        kernel: KernelArg = "linear",
        n_components: Optional[int] = None,
        pct_variance: float = 0.95,
        temperature: float = 1.0,
        eps: float = 1e-12,
        rff_dim: int = 256,
        rff_gamma: float = 1.0,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Configure kernel PCA, subspace selection, and fusion numerics.

        Args:
            kernel: ``"linear"``, ``"cosine"`` (CoP), or ``"rff_cosine"`` (CoRP).
            n_components: Fixed ``k``; if ``None``, ``pct_variance`` selects ``k``.
            pct_variance: Variance threshold on the working covariance when
                ``n_components`` is ``None``.
            temperature: Temperature scaling logits before log-sum-exp (divide logits by this).
            eps: Small constant added to the embedding row L2 norm in the denominator of ``r``.
            rff_dim: RFF dimension for ``rff_cosine``.
            rff_gamma: RBF ``gamma`` for RFF when ``kernel == "rff_cosine"``.
            random_state: RNG seed for RFF weights (``rff_cosine`` only).

        Raises:
            ValueError: If ``temperature <= 0``, ``eps <= 0``, or kernel invalid.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if kernel not in ("linear", "cosine", "rff_cosine"):
            raise ValueError("kernel must be 'linear', 'cosine', or 'rff_cosine'")
        self.kernel = kernel
        self.n_components = n_components
        self.pct_variance = float(pct_variance)
        self.temperature = float(temperature)
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
    ) -> "PCAFusion":
        """Fit PCA subspace on in-distribution embeddings.

        Args:
            features_train: Must provide ``embeddings`` with at least two samples and
                two feature dimensions.
            **kwargs: Unused.

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
        """Per-sample score is negative of the paper's D (see Guan et al. Eq. 13).

        Up to temperature scaling, D is (1 - r) times log-sum-exp of logits, with
        r from normalized reconstruction error; implementation returns ``-D``.

        Args:
            features_test: ``embeddings`` and ``logits`` with matching batch size.
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
        if h.shape[1] != self._state.n_features_in_:
            raise ValueError(
                f"expected {self._state.n_features_in_} features, got {h.shape[1]}"
            )

        recon_err = reconstruction_errors_batch(self._state, h)
        r = guan_r(recon_err, h, self.eps)

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

    @property
    def n_components_fitted_(self) -> int:
        """Number of principal components retained after ``fit``."""
        self._check_is_fitted()
        return self._state.n_components_fitted_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Per-component share of eigenvalue mass in the working space, descending order.

        Same convention as ``oodkit.detectors.pca.PCA.explained_variance_ratio_``.
        """
        self._check_is_fitted()
        return self._state.explained_variance_ratio_

    @property
    def cumulative_explained_variance_ratio_(self) -> np.ndarray:
        """Cumulative sum of ``explained_variance_ratio_``."""
        self._check_is_fitted()
        return np.cumsum(self._state.explained_variance_ratio_, dtype=np.float64)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_state"):
            raise RuntimeError("PCAFusion is not fitted yet.")
