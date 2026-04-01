"""
Lazy dependency gate for ``oodkit.embeddings``.

All modules in this subpackage call ``require_ml_deps()`` at import time so the
rest of oodkit stays importable without torch / transformers.
"""


def require_ml_deps() -> None:
    """Raise a clear ``ImportError`` if torch or transformers are missing."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        raise ImportError(
            f"oodkit.embeddings requires {', '.join(missing)}. "
            "Install with:  pip install oodkit[ml]"
        )
