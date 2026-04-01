"""
Rich output container for extracted embeddings (and optional logits / labels).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from oodkit.data.features import Features


@dataclass
class EmbeddingResult:
    """Output of ``Embedder.extract`` or ``Embedder.fit_extract``.

    Always contains ``embeddings``; ``logits`` is present only when a classifier
    head was trained (``mode="head"`` or ``mode="full"``). ``labels`` stores
    ground-truth class indices when the dataset provides them, for convenient
    forwarding to ``detector.fit(..., y=result.labels)``.

    ``metadata`` is an extensible dict for per-sample info (e.g. image paths
    now, bounding boxes for future object-detection support).
    """

    embeddings: np.ndarray
    logits: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def to_features(self) -> Features:
        """Bridge to the detector API.

        Returns:
            A ``Features`` bundle with ``embeddings`` (always) and ``logits``
            (if available).
        """
        return Features(logits=self.logits, embeddings=self.embeddings)
