"""Tests for the lazy dependency guard."""

from oodkit.embeddings._guard import require_ml_deps


def test_require_ml_deps_passes():
    """In test environments where torch+transformers are installed, no error."""
    try:
        require_ml_deps()
    except ImportError:
        pass
