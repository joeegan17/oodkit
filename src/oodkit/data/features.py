"""
Features: unified container for logits and embeddings.

Can hold logits only, embeddings only, or both.
Detectors choose what they need (e.g., MSP → logits, KNN → embeddings, ViM → both).

Note: `Features` is a model-output container name and is not synonymous with
"embeddings only" in this library.
"""

from typing import Optional

from oodkit.types import ArrayLike


class Features:
    """
    Unified container for model outputs used by OOD detectors.

    Parameters
    ----------
    logits : ArrayLike, optional
        Class logits, shape (n_samples, n_classes).
    embeddings : ArrayLike, optional
        Feature embeddings, shape (n_samples, n_features).
    Notes
    -----
    At least one of logits or embeddings must be provided.
    """

    def __init__(
        self,
        logits: Optional[ArrayLike] = None,
        embeddings: Optional[ArrayLike] = None,
    ) -> None:
        if logits is None and embeddings is None:
            raise ValueError("At least one of logits or embeddings must be provided")
        self.logits = logits
        self.embeddings = embeddings
