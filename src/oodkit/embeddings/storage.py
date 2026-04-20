"""
Load saved embeddings from disk (written by ``Embedder.extract(save_to=...)``).

No torch dependency — works on any machine with just NumPy.
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from oodkit.embeddings.result import EmbeddingResult


def load_embeddings(
    path: Union[str, Path],
    frac: float = 1.0,
    seed: Optional[int] = None,
) -> EmbeddingResult:
    """Reload embeddings previously saved by ``Embedder.extract(save_to=...)``.

    Args:
        path: Directory containing ``manifest.json`` and ``.npy`` files.
        frac: Fraction of samples to load (``0.0`` < frac <= ``1.0``).
            Values below 1.0 select a random subset, keeping memory usage
            proportional.
        seed: Random seed for reproducible subsampling when ``frac < 1.0``.

    Returns:
        ``EmbeddingResult`` with the requested data loaded into memory.

    Raises:
        FileNotFoundError: If the manifest or array files are missing.
        ValueError: If ``frac`` is not in ``(0.0, 1.0]``.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0.0, 1.0], got {frac}")

    root = Path(path)
    with open(root / "manifest.json") as f:
        manifest = json.load(f)

    n = manifest["n_samples"]

    if frac >= 1.0:
        indices = None
    else:
        rng = np.random.default_rng(seed)
        k = max(1, int(n * frac))
        indices = np.sort(rng.choice(n, size=k, replace=False))

    embeddings = _load_array(root / "embeddings.npy", indices)
    logits = _load_array(root / "logits.npy", indices) if manifest["has_logits"] else None
    labels = _load_array(root / "labels.npy", indices) if manifest["has_labels"] else None

    metadata: dict = {}
    if manifest.get("has_image_paths"):
        ip_path = root / "image_paths.json"
        if ip_path.exists():
            with open(ip_path) as f:
                all_paths = json.load(f)
            metadata["image_paths"] = (
                [all_paths[i] for i in indices] if indices is not None else all_paths
            )

    if manifest.get("has_chip_to_image"):
        metadata["chip_to_image"] = _load_array(root / "chip_to_image.npy", indices)
    if manifest.get("has_boxes"):
        metadata["boxes"] = _load_array(root / "boxes.npy", indices)

    return EmbeddingResult(
        embeddings=embeddings,
        logits=logits,
        labels=labels,
        metadata=metadata,
    )


def _load_array(path: Path, indices: Optional[np.ndarray]) -> np.ndarray:
    """Memory-map a ``.npy`` file, optionally subset, and copy into RAM."""
    mmap = np.load(str(path), mmap_mode="r")
    if indices is not None:
        return np.array(mmap[indices])
    return np.array(mmap)
