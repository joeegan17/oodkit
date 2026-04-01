"""
High-level ``Embedder`` for extracting image embeddings (and optional logits).

Wraps a pretrained vision backbone (DINOv3 by default) loaded from HuggingFace,
with optional classifier-head or full-model finetuning.
"""

import json
from pathlib import Path
from typing import Optional, Union

from oodkit.embeddings._guard import require_ml_deps

require_ml_deps()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from numpy.lib.format import open_memmap  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from oodkit.embeddings.backbones import load_backbone  # noqa: E402
from oodkit.embeddings.datasets import make_dataloader, resolve_dataset  # noqa: E402
from oodkit.embeddings.result import EmbeddingResult  # noqa: E402
from oodkit.embeddings.training import (  # noqa: E402
    save_checkpoint,
    train_full,
    train_head,
)

_VALID_MODES = ("none", "head", "full")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Embedder:
    """Extract embeddings (and optionally logits) from images.

    Default backbone is DINOv3-S via HuggingFace. No training is required for
    pure embedding extraction (``mode="none"``); optional ``mode="head"`` or
    ``mode="full"`` trains a classifier so that ``logits`` are available for
    logit-based OOD detectors (MSP, Energy, ViM, etc.).

    Outputs are returned as ``EmbeddingResult`` with a ``.to_features()``
    bridge to ``oodkit.detectors``.
    """

    def __init__(
        self,
        backbone: str = "dinov3-small",
        device: str = "auto",
    ) -> None:
        """Initialize with a backbone preset and target device.

        Args:
            backbone: Short alias for a registered backbone (e.g. ``"dinov3-small"``).
            device: ``"cpu"``, ``"cuda"``, or ``"auto"`` (CUDA when available).

        Raises:
            ValueError: If ``backbone`` is not in the preset registry.
        """
        self._backbone_name = backbone
        self._device = _resolve_device(device)
        self._model, self._processor, self._embed_dim = load_backbone(backbone)
        self._model.to(self._device)
        self._model.eval()
        self._head: Optional[nn.Linear] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: Union[str, Path, Dataset],
        mode: str = "none",
        epochs: int = 10,
        batch_size: int = 64,
        num_workers: int = 1,
        lr: float = 1e-3,
        save: bool = True,
        save_path: str = "oodkit_checkpoint/",
    ) -> "Embedder":
        """Optionally train a classifier head or finetune the backbone.

        The dataset must yield ``(image, label)`` tuples. The easiest way is to
        pass a path to an ``ImageFolder``-compatible directory tree (one
        subfolder per class) or a PyTorch ``Dataset`` that returns
        ``(image_tensor, label_int)`` pairs.

        Args:
            dataset: A labeled PyTorch ``Dataset`` yielding ``(image, label)``
                tuples, or a path to an ``ImageFolder``-compatible directory
                tree (one subfolder per class).
            mode: ``"none"`` (pretrained only), ``"head"`` (train classifier,
                backbone frozen), or ``"full"`` (finetune everything).
            epochs: Training epochs (ignored when ``mode="none"``).
            batch_size: Samples per batch during training.
            num_workers: DataLoader workers.
            lr: Learning rate for Adam.
            save: Whether to save a checkpoint after training.
            save_path: Directory for the checkpoint.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``mode`` is invalid or the dataset does not contain
                labels / has fewer than 2 classes.
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'")
        if mode == "none":
            return self

        ds = resolve_dataset(dataset, self._processor)
        n_classes = self._infer_n_classes(ds)
        if n_classes < 2:
            raise ValueError("Training requires at least 2 classes")

        self._head = nn.Linear(self._embed_dim, n_classes).to(self._device)

        loader = make_dataloader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        if mode == "head":
            self._model, self._head = train_head(
                self._model, self._head, loader, epochs, lr, self._device,
            )
        else:
            self._model, self._head = train_full(
                self._model, self._head, loader, epochs, lr, self._device,
            )

        if save:
            meta = {
                "backbone": self._backbone_name,
                "embed_dim": self._embed_dim,
                "n_classes": n_classes,
                "mode": mode,
                "epochs": epochs,
            }
            save_checkpoint(self._model, self._head, save_path, meta)

        self._model.eval()
        if self._head is not None:
            self._head.eval()
        return self

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(
        self,
        dataset: Union[str, Path, Dataset],
        batch_size: int = 64,
        num_workers: int = 1,
        save_to: Optional[Union[str, Path]] = None,
    ) -> EmbeddingResult:
        """Run the backbone (and optional head) over a dataset.

        Args:
            dataset: PyTorch ``Dataset`` or path to images.
            batch_size: Samples per batch.
            num_workers: DataLoader workers.
            save_to: If set, write arrays to this directory incrementally
                using memory-mapped files instead of accumulating in RAM.
                Useful for datasets too large to hold in memory. Use
                ``load_embeddings`` to reload (partially or fully) later.

        Returns:
            ``EmbeddingResult`` with embeddings (always), logits (if head
            exists), and labels / metadata when available. When ``save_to``
            is set, arrays are memory-mapped (near-zero RAM footprint).
        """
        ds = resolve_dataset(dataset, self._processor)
        loader = make_dataloader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        self._model.eval()
        if self._head is not None:
            self._head.eval()

        if save_to is not None:
            return self._extract_to_disk(ds, loader, Path(save_to))

        all_embeddings: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        has_labels = self._dataset_has_labels(ds)

        with torch.no_grad():
            for batch in tqdm(loader, desc="extracting", leave=False):
                if has_labels:
                    images, labels_batch = batch[0].to(self._device), batch[1]
                    all_labels.append(labels_batch.numpy() if isinstance(labels_batch, torch.Tensor) else np.asarray(labels_batch))
                else:
                    images = batch[0].to(self._device) if isinstance(batch, (list, tuple)) else batch.to(self._device)

                # CLS token (position 0) as the image-level embedding
                emb = self._model(images).last_hidden_state[:, 0]
                all_embeddings.append(emb.cpu().numpy())

                if self._head is not None:
                    logits = self._head(emb)
                    all_logits.append(logits.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        logits = np.concatenate(all_logits, axis=0).astype(np.float32) if all_logits else None
        labels_out = np.concatenate(all_labels, axis=0) if all_labels else None

        metadata = {}
        if hasattr(ds, "imgs"):
            metadata["image_paths"] = [p for p, _ in ds.imgs]

        return EmbeddingResult(
            embeddings=embeddings,
            logits=logits,
            labels=labels_out,
            metadata=metadata,
        )

    def _extract_to_disk(
        self,
        ds: Dataset,
        loader: "torch.utils.data.DataLoader",
        save_dir: Path,
    ) -> EmbeddingResult:
        """Stream extraction results to memory-mapped ``.npy`` files.

        Creates ``save_dir`` with ``embeddings.npy``, optional ``logits.npy``
        and ``labels.npy``, plus a ``manifest.json`` for ``load_embeddings``.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        n = len(ds)
        has_labels = self._dataset_has_labels(ds)
        has_head = self._head is not None

        emb_mmap = open_memmap(
            str(save_dir / "embeddings.npy"),
            mode="w+", dtype=np.float32, shape=(n, self._embed_dim),
        )
        logits_mmap = None
        if has_head:
            logits_mmap = open_memmap(
                str(save_dir / "logits.npy"),
                mode="w+", dtype=np.float32, shape=(n, self._head.out_features),
            )
        labels_mmap = None
        if has_labels:
            labels_mmap = open_memmap(
                str(save_dir / "labels.npy"),
                mode="w+", dtype=np.int64, shape=(n,),
            )

        idx = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="extracting → disk", leave=False):
                if has_labels:
                    images, labels_batch = batch[0].to(self._device), batch[1]
                else:
                    images = (
                        batch[0].to(self._device)
                        if isinstance(batch, (list, tuple))
                        else batch.to(self._device)
                    )

                # CLS token (position 0) as the image-level embedding
                emb = self._model(images).last_hidden_state[:, 0]
                bs = emb.shape[0]
                emb_mmap[idx : idx + bs] = emb.cpu().numpy().astype(np.float32)

                if has_head:
                    logits = self._head(emb)
                    logits_mmap[idx : idx + bs] = logits.cpu().numpy().astype(np.float32)

                if has_labels:
                    lbl = (
                        labels_batch.numpy()
                        if isinstance(labels_batch, torch.Tensor)
                        else np.asarray(labels_batch)
                    )
                    labels_mmap[idx : idx + bs] = lbl

                idx += bs

        emb_mmap.flush()
        del emb_mmap
        if logits_mmap is not None:
            logits_mmap.flush()
            del logits_mmap
        if labels_mmap is not None:
            labels_mmap.flush()
            del labels_mmap

        metadata: dict = {}
        if hasattr(ds, "imgs"):
            paths_list = [p for p, _ in ds.imgs]
            metadata["image_paths"] = paths_list
            with open(save_dir / "image_paths.json", "w") as f:
                json.dump(paths_list, f)

        manifest = {
            "n_samples": n,
            "embed_dim": self._embed_dim,
            "has_logits": has_head,
            "has_labels": has_labels,
            "has_image_paths": bool(metadata.get("image_paths")),
        }
        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return EmbeddingResult(
            embeddings=np.load(str(save_dir / "embeddings.npy"), mmap_mode="r"),
            logits=np.load(str(save_dir / "logits.npy"), mmap_mode="r") if has_head else None,
            labels=np.load(str(save_dir / "labels.npy"), mmap_mode="r") if has_labels else None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # fit_extract
    # ------------------------------------------------------------------

    def fit_extract(
        self,
        dataset: Union[str, Path, Dataset],
        mode: str = "none",
        **kwargs: object,
    ) -> EmbeddingResult:
        """Convenience: ``fit`` then ``extract`` on the same dataset.

        Args:
            dataset: A labeled PyTorch ``Dataset`` or path (see ``fit``).
            mode: Training mode (see ``fit``).
            **kwargs: Forwarded to ``fit`` (``epochs``, ``lr``, etc.) and
                ``extract`` (``batch_size``, ``num_workers``).

        Returns:
            ``EmbeddingResult``.
        """
        fit_keys = {"epochs", "lr", "save", "save_path"}
        shared_keys = {"batch_size", "num_workers"}
        extract_keys = {"save_to"}
        fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_keys | shared_keys}
        extract_kwargs = {k: v for k, v in kwargs.items() if k in shared_keys | extract_keys}

        self.fit(dataset, mode=mode, **fit_kwargs)
        return self.extract(dataset, **extract_kwargs)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "Embedder":
        """Restore an ``Embedder`` from a saved checkpoint.

        Args:
            path: Directory containing ``backbone.pt``, ``meta.json``, and
                optionally ``head.pt``.
            device: Target device (``"auto"`` / ``"cpu"`` / ``"cuda"``).

        Returns:
            A ready-to-use ``Embedder`` with trained weights loaded.

        Raises:
            FileNotFoundError: If checkpoint files are missing.
        """
        ckpt = Path(path)
        with open(ckpt / "meta.json") as f:
            meta = json.load(f)

        emb = cls(backbone=meta["backbone"], device=device)
        emb._model.load_state_dict(torch.load(ckpt / "backbone.pt", map_location=emb._device, weights_only=True))
        emb._model.eval()

        head_path = ckpt / "head.pt"
        if head_path.exists():
            n_classes = meta["n_classes"]
            emb._head = nn.Linear(emb._embed_dim, n_classes).to(emb._device)
            emb._head.load_state_dict(torch.load(head_path, map_location=emb._device, weights_only=True))
            emb._head.eval()

        return emb

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dataset_has_labels(ds: Dataset) -> bool:
        """Heuristic: dataset yields ``(image, label)`` tuples."""
        try:
            sample = ds[0]
            return isinstance(sample, (list, tuple)) and len(sample) >= 2
        except Exception:
            return False

    @staticmethod
    def _infer_n_classes(ds: Dataset) -> int:
        """Determine class count from dataset metadata.

        Raises:
            ValueError: If the dataset has no ``.classes`` or ``.targets``
                attribute (i.e. it does not carry label information).
        """
        if hasattr(ds, "classes"):
            return len(ds.classes)
        if hasattr(ds, "targets"):
            return len(set(ds.targets))
        raise ValueError(
            "Cannot determine number of classes. The dataset must carry "
            "labels — pass an ImageFolder-compatible path (one subfolder per "
            "class) or a Dataset with a .classes / .targets attribute."
        )
