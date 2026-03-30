"""
Shared PCA / kernel-PCA helpers for ``PCA`` and ``PCAFusion``.

CoP (cosine) and CoRP (RFF on cosine) follow Fang et al., NeurIPS 2024:
https://proceedings.neurips.cc/paper_files/paper/2024/file/f2543511e5f4d4764857f9ad833a977d-Paper-Conference.pdf

CoP: L2-normalize each embedding row to unit length, then linear PCA on mean-centered
features. CoRP: same normalization, then Random Fourier Features (Gaussian RBF kernel)
into a finite-dimensional space, then linear PCA there.

Implementation notes:
- ``linear``: mean-center raw embeddings, PCA in the original feature dimension.
- ``cosine`` (CoP): PCA on mean-centered **row-normalized** embeddings.
- ``rff_cosine`` (CoRP): RFF map of normalized embeddings, then PCA in RFF space.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

KernelName = Literal["linear", "cosine", "rff_cosine"]


def select_principal_subspace_dim(
    evals_desc: np.ndarray,
    n_features: int,
    n_components: Optional[int],
    pct_variance: float,
) -> int:
    """Choose ``k`` principal components from descending eigenvalues.

    For ``pct_variance``, that is cumulative eigenvalue mass of the working covariance
    (per kernel, see ``fit_pca_subspace``).
    """
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


def row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of ``X``."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def rff_gaussian_map(
    X: np.ndarray,
    n_components: int,
    gamma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random Fourier features for RBF kernel ``exp(-gamma ||x-y||^2)``.

    Uses ``sqrt(2/D) cos(w^T x + b)`` with ``w ~ N(0, 2*gamma I)``, ``b ~ U[0,2π]``.

    Returns:
        ``Z`` of shape ``(n_samples, n_components)``, and ``(W, b)`` for test-time reuse.
    """
    n_features = X.shape[1]
    W = rng.standard_normal((n_features, n_components))
    W *= np.sqrt(2.0 * gamma)
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    z = X @ W + b
    scale = np.sqrt(2.0 / n_components)
    return scale * np.cos(z), W, b


def apply_rff_gaussian_map(X: np.ndarray, W: np.ndarray, b: np.ndarray, n_components: int) -> np.ndarray:
    """Apply the same RFF map as ``rff_gaussian_map`` with fixed ``W``, ``b``."""
    z = X @ W + b
    scale = np.sqrt(2.0 / n_components)
    return scale * np.cos(z)


@dataclass
class PCASubspaceState:
    """Fitted subspace for reconstruction-based scoring."""

    kernel: KernelName
    n_features_in_: int
    n_components_fitted_: int
    mean_: np.ndarray
    principal_basis_: np.ndarray
    rff_W: Optional[np.ndarray] = None
    rff_b: Optional[np.ndarray] = None
    rff_n_components: int = 0
    rff_gamma: float = 1.0


def fit_pca_subspace(
    X: np.ndarray,
    kernel: KernelName,
    n_components: Optional[int],
    pct_variance: float,
    rff_dim: int,
    rff_gamma: float,
    rng: np.random.Generator,
) -> PCASubspaceState:
    """Fit mean-centered linear PCA in the chosen feature space.

    Each kernel applies a fixed map into a finite space, then covariance PCA, so
    ``pct_variance`` is cumulative eigenvalue mass in that working space—not raw
    embedding variance for cosine or RFF; for CoRP that space also depends on
    ``rff_dim`` and the RFF draw.
    """
    n_samples, n_features = X.shape
    if n_features < 2:
        raise ValueError("PCA requires at least 2 feature dimensions")
    if n_samples < 2:
        raise ValueError("PCA requires at least 2 samples")

    if kernel == "linear":
        mean = X.mean(axis=0).astype(np.float64, copy=False)
        Xw = X - mean
        cov = Xw.T @ Xw
        n_feat_pca = n_features
    elif kernel == "cosine":
        Xn = row_normalize(X)
        mean = Xn.mean(axis=0).astype(np.float64, copy=False)
        Xw = Xn - mean
        cov = Xw.T @ Xw
        n_feat_pca = n_features
    elif kernel == "rff_cosine":
        if rff_dim < 2:
            raise ValueError("rff_dim must be >= 2 for rff_cosine")
        Xn = row_normalize(X)
        Z, W, b = rff_gaussian_map(Xn, rff_dim, rff_gamma, rng)
        mean = Z.mean(axis=0).astype(np.float64, copy=False)
        Xw = Z - mean
        cov = Xw.T @ Xw
        n_feat_pca = rff_dim
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    evals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, ::-1]
    evals_desc = evals[::-1]

    k = select_principal_subspace_dim(
        evals_desc,
        n_feat_pca,
        n_components,
        pct_variance,
    )
    V = eigvecs[:, :k] if k > 0 else np.zeros((n_feat_pca, 0), dtype=np.float64)

    if kernel == "rff_cosine":
        assert W is not None and b is not None
        return PCASubspaceState(
            kernel=kernel,
            n_features_in_=n_features,
            n_components_fitted_=k,
            mean_=mean,
            principal_basis_=V,
            rff_W=W,
            rff_b=b,
            rff_n_components=rff_dim,
            rff_gamma=float(rff_gamma),
        )
    return PCASubspaceState(
        kernel=kernel,
        n_features_in_=n_features,
        n_components_fitted_=k,
        mean_=mean,
        principal_basis_=V,
        rff_W=None,
        rff_b=None,
        rff_n_components=0,
        rff_gamma=rff_gamma,
    )


def reconstruction_errors_batch(state: PCASubspaceState, h: np.ndarray) -> np.ndarray:
    """Per-sample reconstruction error norm (higher ⇒ more OOD)."""
    if h.ndim != 2 or h.shape[1] != state.n_features_in_:
        raise ValueError(
            f"embeddings must have shape [N, {state.n_features_in_}], got {h.shape}"
        )

    k = state.n_components_fitted_
    V = state.principal_basis_
    mean = state.mean_

    if state.kernel == "linear":
        centered = h - mean
        if k == 0:
            h_hat = np.broadcast_to(mean, h.shape).copy()
        else:
            coef = centered @ V
            h_hat = mean + coef @ V.T
        return np.linalg.norm(h - h_hat, axis=1)

    if state.kernel == "cosine":
        hn = row_normalize(h)
        centered = hn - mean
        if k == 0:
            h_hat = np.broadcast_to(mean, hn.shape).copy()
        else:
            coef = centered @ V
            h_hat = mean + coef @ V.T
        return np.linalg.norm(hn - h_hat, axis=1)

    assert state.rff_W is not None and state.rff_b is not None
    hn = row_normalize(h)
    Z = apply_rff_gaussian_map(hn, state.rff_W, state.rff_b, state.rff_n_components)
    centered = Z - mean
    if k == 0:
        z_hat = np.broadcast_to(mean, Z.shape).copy()
    else:
        coef = centered @ V
        z_hat = mean + coef @ V.T
    return np.linalg.norm(Z - z_hat, axis=1)


def guan_r(recon_err: np.ndarray, h_original: np.ndarray, eps: float) -> np.ndarray:
    """r = recon_err / (row L2 norm of original embedding + eps), clipped to 0..1 (Guan et al.)."""
    h_norm = np.linalg.norm(h_original, axis=1)
    return np.clip(recon_err / (h_norm + eps), 0.0, 1.0)
