"""
ViM (Virtual Logit Matching) OOD detector.

Uses both logits and embeddings from Features.
"""

from typing import TYPE_CHECKING

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike

if TYPE_CHECKING:
    from oodkit.data.features import Features


class ViM(BaseDetector):
    """
    ViM detector placeholder.

    Uses both logits and embeddings from Features.
    Implements fit, score, and predict via BaseDetector.
    """

    def fit(self, features: "Features", **kwargs) -> "ViM":
        """Fit ViM on in-distribution features (logits + embeddings)."""
        # TODO: implement
        return self

    def score(self, features: "Features", **kwargs) -> ArrayLike:
        """Compute ViM OOD scores."""
        # TODO: implement
        raise NotImplementedError("ViM.score not yet implemented")
