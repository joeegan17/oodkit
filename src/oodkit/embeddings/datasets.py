"""
Dataset and DataLoader helpers for ``oodkit.embeddings``.

Accepts a PyTorch ``Dataset`` or a filesystem path (converted via
``torchvision.datasets.ImageFolder``).
"""

from pathlib import Path
from typing import Any, Optional, Union

from oodkit.embeddings._guard import require_ml_deps

require_ml_deps()

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402


def resolve_dataset(
    dataset: Union[str, Path, Dataset],
    processor: Any,
) -> Dataset:
    """Normalise user input into a PyTorch ``Dataset``.

    For **training** (``Embedder.fit``), the dataset must yield
    ``(image, label)`` tuples. The simplest way is to pass a path to an
    ``ImageFolder``-compatible directory tree::

        my_data/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg

    Alternatively, pass any PyTorch ``Dataset`` that returns
    ``(image_tensor, label_int)`` pairs.

    For **extraction only** (``Embedder.extract``), unlabeled datasets
    (returning just an image tensor) are also accepted.

    Args:
        dataset: A PyTorch ``Dataset``, or a string / ``Path`` pointing to an
            ``ImageFolder``-compatible directory tree.
        processor: HuggingFace ``AutoImageProcessor`` used as the transform
            when building an ``ImageFolder`` from a path.

    Returns:
        A PyTorch ``Dataset``.

    Raises:
        TypeError: If ``dataset`` is neither a path nor a ``Dataset``.
        FileNotFoundError: If a path is given but does not exist.
    """
    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")

        def _transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return ImageFolder(root=str(path), transform=_transform)

    if isinstance(dataset, Dataset):
        return dataset

    raise TypeError(
        f"dataset must be a path (str/Path) or a PyTorch Dataset, got {type(dataset).__name__}"
    )


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 1,
    shuffle: bool = False,
    pin_memory: Optional[bool] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """Wrap a ``Dataset`` in a ``DataLoader`` with sensible defaults.

    Args:
        dataset: PyTorch ``Dataset``.
        batch_size: Samples per batch.
        num_workers: Parallel data-loading workers.
        shuffle: Whether to shuffle every epoch.
        pin_memory: If ``None``, enabled when CUDA is available.
        persistent_workers: Keep worker processes alive between epochs (only
            applied when ``num_workers > 0``).

    Returns:
        A ``DataLoader``.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kw: dict = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0 and persistent_workers:
        kw["persistent_workers"] = True
    return DataLoader(**kw)
