"""
ImageNet-style folder datasets (optional; for demos and benchmarks).

Requires ``torch`` and ``torchvision`` (same as ``oodkit.embeddings``).

Example::

    from oodkit.contrib.imagenet import imagenet_variant_dataset
    from oodkit.embeddings.backbones import load_backbone
    from oodkit.embeddings.embedder import Embedder

    _, processor, _ = load_backbone("dinov2-small")
    emb = Embedder()
    ds = imagenet_variant_dataset(
        "path/to/imagenet-o",
        "path/to/LOC_synset_mapping.txt",
        processor,
    )
    result = emb.extract(ds)
"""

from oodkit.contrib.imagenet.synset_table import RootValidation, SynsetTable

__all__ = [
    "RootValidation",
    "SynsetTable",
    "SynsetImageDataset",
    "imagenet_variant_dataset",
]


def __getattr__(name: str):
    if name in ("SynsetImageDataset", "imagenet_variant_dataset"):
        from oodkit.contrib.imagenet.dataset import SynsetImageDataset, imagenet_variant_dataset

        return {"SynsetImageDataset": SynsetImageDataset, "imagenet_variant_dataset": imagenet_variant_dataset}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | {"__doc__", "__file__"})
