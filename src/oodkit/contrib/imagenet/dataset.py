"""
PyTorch ``Dataset`` for ImageFolder-style trees with **canonical** ImageNet-1k labels.

Assumes one subdirectory per synset (``wnid``), same layout as standard class
folders. Raw ILSVRC validation (flat filenames) must be reorganized into
per-``wnid`` folders before using this loader, or use another pipeline.

Requires the optional ML stack (``torch``, ``torchvision``) like
``oodkit.embeddings``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from oodkit.contrib.imagenet.synset_table import SynsetTable

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SynsetImageDataset requires torch and torchvision. "
        'Install with: pip install "oodkit[ml]"'
    ) from exc


def _is_image_file(path: Path, extensions: Tuple[str, ...]) -> bool:
    return path.is_file() and path.suffix.lower() in extensions


class SynsetImageDataset(Dataset):
    """Labeled image dataset with **canonical** indices from :class:`SynsetTable`.

    Compatible with :class:`oodkit.embeddings.embedder.Embedder` when passed as
    ``dataset=`` (not as a bare path): samples are
    ``(processor_output_tensor, int_label)``. Exposes ``imgs`` like
    ``torchvision.datasets.ImageFolder`` so ``extract`` records
    ``metadata["image_paths"]``.

    Sample order is deterministic: **sorted synset folder names** (``wnid``), then
    **sorted image paths** within each folder. That matches canonical class order
    because ``LOC_synset_mapping.txt`` uses lexicographic ``wnid`` order. For
    **training**, use ``DataLoader(..., shuffle=True)`` so minibatches are not
    dominated by one class; ``Embedder.extract`` uses ``shuffle=False``, which
    only groups consecutive batches by class and does not change per-sample
    embeddings.

    Args:
        root: Root directory containing one subfolder per ``wnid``.
        synset_table: Mapping from :func:`SynsetTable.from_file`.
        processor: HuggingFace image processor (same as ``Embedder`` uses); called
            as ``processor(images=pil_img, return_tensors="pt")``.
        extensions: Filename suffixes to include (lowercase, with dot).
        strict: If True, raise when a subdirectory name is not a known ``wnid``.
        loader: PIL image loader (default: torchvision ``default_loader``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        synset_table: SynsetTable,
        processor: Any,
        *,
        extensions: Tuple[str, ...] = IMG_EXTENSIONS,
        strict: bool = True,
        loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root is not a directory: {self.root}")

        self.synset_table = synset_table
        self.processor = processor
        self.extensions = tuple(extensions)
        self.strict = strict
        self.loader = loader if loader is not None else default_loader

        self.imgs: List[Tuple[str, int]] = self._scan_samples()
        self.targets = [label for _, label in self.imgs]
        self.classes = list(synset_table.idx_to_name)
        self.wnids = list(synset_table.idx_to_wnid)

        if not self.imgs:
            raise ValueError(f"No images found under {self.root} with extensions {self.extensions}")

    def _scan_samples(self) -> List[Tuple[str, int]]:
        """Collect ``(path, canonical_label)`` in sorted wnid / sorted filename order."""
        wnid_to_idx = self.synset_table.wnid_to_idx
        samples: List[Tuple[str, int]] = []

        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name
            if name not in wnid_to_idx:
                msg = f"Unknown synset folder {name!r} under {self.root}"
                if self.strict:
                    raise ValueError(
                        f"{msg}. Expected only wnids from the synset table, or pass strict=False."
                    )
                warnings.warn(f"{msg}; skipping.", UserWarning, stacklevel=2)
                continue

            label = wnid_to_idx[name]
            for f in sorted(sub.iterdir()):
                if _is_image_file(f, self.extensions):
                    samples.append((str(f.resolve()), label))

        return samples

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, y = self.imgs[index]
        pil_img = self.loader(path)
        pixel_values = self.processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, y

    def sample_descriptor(self, index: int) -> dict:
        """Per-sample ids for notebooks (basename + synset + canonical index)."""
        path, y = self.imgs[index]
        p = Path(path)
        wnid = p.parent.name
        return {
            "path": path,
            "image_id": p.name,
            "wnid": wnid,
            "canonical_idx": y,
            "name": self.classes[y],
        }


def imagenet_variant_dataset(
    variant_root: Union[str, Path],
    mapping_path: Union[str, Path],
    processor: Any,
    *,
    strict: bool = True,
    extensions: Tuple[str, ...] = IMG_EXTENSIONS,
    loader: Optional[Callable[..., Any]] = None,
) -> SynsetImageDataset:
    """Load :class:`SynsetTable` from file and build a :class:`SynsetImageDataset`.

    Args:
        variant_root: Root of one variant (e.g. ImageNet-O).
        mapping_path: Path to ``LOC_synset_mapping.txt``.
        processor: HuggingFace processor for the backbone used by ``Embedder``.
        strict: Forwarded to :class:`SynsetImageDataset`.
        extensions: Included image suffixes.
        loader: Optional PIL loader.

    Returns:
        Ready-to-use dataset for ``Embedder.extract`` / ``fit``.
    """
    table = SynsetTable.from_file(mapping_path)
    return SynsetImageDataset(
        variant_root,
        table,
        processor,
        strict=strict,
        extensions=extensions,
        loader=loader,
    )
