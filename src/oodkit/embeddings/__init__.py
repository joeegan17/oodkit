"""
Embedding extraction module for OODKit.

Requires PyTorch and HuggingFace ``transformers`` for extraction
(``Embedder``).  Install with:  ``pip install oodkit[ml]``

``load_embeddings`` works with just NumPy (no torch needed).

Public API::

    from oodkit.embeddings import Embedder, EmbeddingResult, load_embeddings

    emb = Embedder()                       # DINOv3-S by default
    result = emb.extract("path/to/images") # -> EmbeddingResult
    features = result.to_features()        # -> Features for detectors

    # Large datasets — save to disk, reload a subset later:
    emb.extract("path/to/images", save_to="my_embeddings/")
    result = load_embeddings("my_embeddings/", frac=0.5)
"""

from oodkit.embeddings.result import EmbeddingResult
from oodkit.embeddings.storage import load_embeddings

__all__ = ["Embedder", "EmbeddingResult", "load_embeddings"]


def __getattr__(name: str):
    """Lazy-load ``Embedder`` so ``EmbeddingResult`` works without torch."""
    if name == "Embedder":
        from oodkit.embeddings.embedder import Embedder as _Embedder

        return _Embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) + list(globals().keys()))
