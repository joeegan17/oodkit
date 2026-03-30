"""
WDiscOOD: LDA-based discriminative vs residual distances on embeddings.

Paper: https://arxiv.org/pdf/2303.07543 (Chen et al., ICCV 2023).

Scores are nonnegative; higher means more OOD. Train-set medians normalize the
two distance channels before combining with ``alpha``.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


def _matrix_sqrt_psd(sym_psd: np.ndarray, eval_floor: float) -> np.ndarray:
    """Return the symmetric PSD square root ``sym_psd ** 0.5`` via ``eigh``.

    Used as the whitening map ``Sw^{-1/2}`` when ``sym_psd = pinv(Sw)`` (so
    ``sqrt(pinv(Sw))`` matches the usual LDA whitening of the between scatter).
    """
    evals, evecs = np.linalg.eigh(sym_psd)
    evals = np.maximum(evals, eval_floor)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T


class WDiscOOD(BaseDetector):
    """Whitened LDA-style split: discriminative (WD) vs residual (WDR) distances.

    Fit requires ``Features.embeddings`` and per-sample class labels ``y``.
    LDA directions are orthonormalized (QR) so the WD projector is orthogonal;
    distances use **global-mean-centered** features in WD and WDR. The final
    score is ``d_wd_norm + alpha * d_wdr_norm`` where each distance term is
    divided by its training-set median (plus ``eps``) so ``alpha`` scales
    comparable channels.

    ``score`` uses embeddings only (class structure is fixed at fit time).
    """

    def __init__(
        self,
        n_discriminants: Optional[int] = None,
        ridge: float = 1e-3,
        alpha: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        """Configure subspace size and fusion.

        Args:
            n_discriminants: Number of leading LDA directions to keep. If ``None``,
                uses ``C - 1`` classes at fit time. Must satisfy
                ``1 <= n_discriminants <= C - 1`` when set.
            ridge: Added to the diagonal of the within-class scatter before inversion.
            alpha: Weight on the normalized WDR distance.
            eps: Small constant added to training medians when normalizing distances.

        Raises:
            ValueError: If ``ridge <= 0``, ``alpha < 0``, or ``eps <= 0``.
        """
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.n_discriminants = n_discriminants
        self.ridge = float(ridge)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def fit(
        self,
        features_train: "Features",
        y: Optional[ArrayLike] = None,
        **kwargs: object,
    ) -> "WDiscOOD":
        """Fit LDA subspaces and training-set distance medians.

        Args:
            features_train: Must provide ``embeddings`` ``(N, D)``.
            y: Integer class labels ``(N,)``. Required; must match ``N``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``y`` is missing, embeddings invalid, ``C < 2``, or
                ``n_discriminants`` is out of range.
        """
        if y is None:
            raise ValueError("WDiscOOD.fit requires y (class labels)")
        if features_train.embeddings is None:
            raise ValueError("WDiscOOD.fit requires Features.embeddings")
        X = np.asarray(to_numpy(features_train.embeddings), dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, D], got {X.shape}")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("WDiscOOD.fit requires at least 2 samples")

        y_np = np.asarray(to_numpy(y)).reshape(-1).astype(np.int64, copy=False)
        if y_np.shape[0] != n_samples:
            raise ValueError(
                f"y must have shape [N], got {y_np.shape}, expected ({n_samples},)"
            )

        _classes, inverse = np.unique(y_np, return_inverse=True)
        n_classes = int(_classes.shape[0])
        if n_classes < 2:
            raise ValueError("WDiscOOD.fit requires at least 2 distinct classes")

        max_k = n_classes - 1
        if self.n_discriminants is None:
            k = max_k
        else:
            k = int(self.n_discriminants)
            if k < 1 or k > max_k:
                raise ValueError(
                    f"n_discriminants must satisfy 1 <= n_discriminants <= C - 1 "
                    f"({max_k}), got {k}"
                )

        global_mean = X.mean(axis=0)
        counts = np.bincount(inverse, minlength=n_classes).astype(np.float64)
        order = np.argsort(inverse, kind="mergesort")
        sorted_x = X[order]
        starts = np.concatenate([[0], np.cumsum(counts[:-1], dtype=np.intp)])
        sums = np.add.reduceat(sorted_x, starts, axis=0)
        class_means = sums / counts[:, None]

        centered = X - class_means[inverse]
        sw = centered.T @ centered + self.ridge * np.eye(n_features, dtype=np.float64)

        deltas_cm = class_means - global_mean
        sb = deltas_cm.T @ np.diag(counts) @ deltas_cm

        sw_pinv = np.linalg.pinv(sw)
        inv_sqrt_sw = _matrix_sqrt_psd(sw_pinv, eval_floor=1e-8)
        m_sym = inv_sqrt_sw @ sb @ inv_sqrt_sw
        m_sym = 0.5 * (m_sym + m_sym.T)
        evals_m, evecs_m = np.linalg.eigh(m_sym)
        order = np.argsort(evals_m)[::-1]
        evecs_m = evecs_m[:, order]
        u = evecs_m[:, :k]
        v_wd_raw = inv_sqrt_sw @ u
        v_wd, _ = np.linalg.qr(v_wd_raw, mode="reduced")
        p = v_wd @ v_wd.T
        i_minus_p = np.eye(n_features, dtype=np.float64) - p

        self.global_mean_ = global_mean
        self.class_means_ = class_means
        self.v_wd_ = v_wd
        self.p_ = p
        self.i_minus_p_ = i_minus_p
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes
        self.k_fitted_ = k

        d_wd, d_wdr = self._raw_distances(X, v_wd, i_minus_p, class_means, global_mean)
        self.median_wd_ = float(np.median(d_wd))
        self.median_wdr_ = float(np.median(d_wdr))

        return self

    def _raw_distances(
        self,
        x: np.ndarray,
        v_wd: np.ndarray,
        i_minus_p: np.ndarray,
        class_means: np.ndarray,
        global_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-sample WD (min class L2) and WDR (L2 to origin) on centered features."""
        xc = x - global_mean
        z_wd = xc @ v_wd
        centroids_wd = (class_means - global_mean) @ v_wd
        diff = z_wd[:, None, :] - centroids_wd[None, :, :]
        d_wd = np.linalg.norm(diff, axis=2).min(axis=1)

        z_wdr = xc @ i_minus_p
        d_wdr = np.linalg.norm(z_wdr, axis=1)
        return d_wd, d_wdr

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Per-sample OOD scores (higher = more OOD).

        Args:
            features_test: ``embeddings`` with feature dimension matching ``fit``.

        Returns:
            Scores ``(n_samples,)``, ``float64``.
        """
        self._check_is_fitted()
        if features_test.embeddings is None:
            raise ValueError("WDiscOOD.score requires Features.embeddings")
        x = np.asarray(to_numpy(features_test.embeddings), dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, D], got {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                "embeddings feature dimension mismatch: "
                f"expected {self.n_features_in_}, got {x.shape[1]}"
            )

        d_wd, d_wdr = self._raw_distances(
            x,
            self.v_wd_,
            self.i_minus_p_,
            self.class_means_,
            self.global_mean_,
        )
        d_wd_n = d_wd / (self.median_wd_ + self.eps)
        d_wdr_n = d_wdr / (self.median_wdr_ + self.eps)
        return (d_wd_n + self.alpha * d_wdr_n).astype(np.float64)

    def predict(
        self,
        features: "Features",
        threshold: float,
        **kwargs: object,
    ) -> ArrayLike:
        """OOD (1) when ``score > threshold``."""
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)

    def _check_is_fitted(self) -> None:
        required = (
            "global_mean_",
            "class_means_",
            "v_wd_",
            "p_",
            "i_minus_p_",
            "median_wd_",
            "median_wdr_",
            "n_features_in_",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"WDiscOOD instance is not fitted yet. Missing: {missing}")
