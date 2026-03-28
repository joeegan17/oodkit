"""
Unified container for model logits and embeddings passed to OOD detectors.
"""

from typing import Optional

from oodkit.types import ArrayLike


class Features:
    """Holds logits and/or embeddings for one batch of model outputs.

    At least one of ``logits`` or ``embeddings`` must be set. Detectors read only
    the fields they need (e.g. MSP uses logits, KNN uses embeddings).

    Note:
        The name ``Features`` refers to general model outputs, not embeddings alone.
    """

    def __init__(
        self,
        logits: Optional[ArrayLike] = None,
        embeddings: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize a ``Features`` bundle.

        Args:
            logits: Class logits, shape ``(n_samples, n_classes)``, optional.
            embeddings: Embedding vectors, shape ``(n_samples, n_features)``, optional.

        Raises:
            ValueError: If both ``logits`` and ``embeddings`` are ``None``.
        """
        if logits is None and embeddings is None:
            raise ValueError("At least one of logits or embeddings must be provided")
        self.logits = logits
        self.embeddings = embeddings
