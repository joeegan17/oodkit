# %% [markdown]
# # OODKit showcase: ImageNet val (ID) vs ImageNet-O (OOD)
#
# Run cell-by-cell in **VS Code / Cursor** (“Run Current Cell”) or **Jupyter** (`# %%` markers).
#
# **Requirements:** `pip install -e ".[ml]"` plus HuggingFace access for the backbone if needed.
#
# **Data layout**
# - **ID (val):** one folder per WordNet id (`n01498041/…`), same as `SynsetImageDataset`.
# - **OOD (e.g. ImageNet-O):** same folder layout (synset subfolders + images).
# - **`LOC_synset_mapping.txt`:** ILSVRC 1000-class list (line order = canonical index).
#
# Edit the **paths** and **training** constants in the first code cell below.

# %%
from __future__ import annotations

import sys
from pathlib import Path

# --- Edit these paths for your machine ---------------------------------------
DATASETS_ROOT = Path(r"C:\dev\datasets")
IMAGENET_VAL_ROOT = DATASETS_ROOT / "imagenet-val"
IMAGENET_O_ROOT = DATASETS_ROOT / "O"
LOC_SYNSET_MAPPING = DATASETS_ROOT / "LOC_synset_mapping.txt"

BACKBONE = "dinov2-small"
HEAD_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4
# DataLoader: use pinned host memory and keep workers alive between epochs (GPU training).
PIN_MEMORY = True
PERSISTENT_WORKERS = True

SEED = 42
# Fraction of **full val** used to train the linear head and fit detectors; rest is ID test in the mix.
TRAIN_FRACTION = 0.9

_DL_KW = dict(
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
)


def _check_paths() -> None:
    missing = [p for p in (IMAGENET_VAL_ROOT, IMAGENET_O_ROOT, LOC_SYNSET_MAPPING) if not p.exists()]
    if missing:
        print("Missing paths — update the constants at the top of this cell:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        raise FileNotFoundError("Fix IMAGENET_VAL_ROOT, IMAGENET_O_ROOT, and LOC_SYNSET_MAPPING.")


_check_paths()

# %%
import numpy as np
import torch
from torch.utils.data import Dataset

from oodkit.contrib.imagenet import SynsetTable, SynsetImageDataset
from oodkit.data.features import Features
from oodkit.detectors import Energy, KNN, MSP, Mahalanobis, PCA, PCAFusion, ViM, WDiscOOD
from oodkit.embeddings.backbones import load_backbone
from oodkit.embeddings.embedder import Embedder
from oodkit.evaluation import ScoreBank, concatenate_embedding_results, evaluate

# %%
class IndexedDatasetSubset(Dataset):
    """Subset by indices while keeping ``imgs`` / ``targets`` / ``classes`` for ``Embedder``."""

    def __init__(self, base: SynsetImageDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = list(indices)
        self.classes = base.classes
        self.wnids = base.wnids
        self.imgs = [base.imgs[i] for i in self.indices]
        self.targets = [lbl for _, lbl in self.imgs]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


# %%
synset_table = SynsetTable.from_file(LOC_SYNSET_MAPPING)
_, processor, _ = load_backbone(BACKBONE)

val_full = SynsetImageDataset(IMAGENET_VAL_ROOT, synset_table, processor, strict=True)
ood_full = SynsetImageDataset(IMAGENET_O_ROOT, synset_table, processor, strict=False)

rng = np.random.default_rng(SEED)
n_val = len(val_full)
if n_val < 2:
    raise RuntimeError(f"Need at least 2 val images; got {n_val} under {IMAGENET_VAL_ROOT}")

n_train = max(1, min(int(TRAIN_FRACTION * n_val), n_val - 1))
n_id_test = n_val - n_train

perm = rng.permutation(n_val)
train_idx = perm[:n_train].tolist()
id_test_idx = perm[n_train : n_train + n_id_test].tolist()

train_ds = IndexedDatasetSubset(val_full, train_idx)
id_test_ds = IndexedDatasetSubset(val_full, id_test_idx)
ood_ds = ood_full

print(f"Val: train={len(train_ds)}  id_test={len(id_test_ds)}  |  ImageNet-O (full): {len(ood_ds)}")

# %%
# Train classifier head on ID train split (backbone frozen in mode="head")
embedder = Embedder(backbone=BACKBONE)
embedder.fit(
    train_ds,
    mode="head",
    epochs=HEAD_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=1e-3,
    save=False,
    **_DL_KW,
)

# %%
# Extract embeddings + logits once per split (full ID test + full ImageNet-O)
train_res = embedder.extract(train_ds, batch_size=BATCH_SIZE, **_DL_KW)
id_test_res = embedder.extract(id_test_ds, batch_size=BATCH_SIZE, **_DL_KW)
ood_res = embedder.extract(ood_ds, batch_size=BATCH_SIZE, **_DL_KW)

assert train_res.logits is not None and train_res.labels is not None

combined, ood_labels = concatenate_embedding_results([id_test_res, ood_res], [0, 1])
comb_feat = combined.to_features()

id_train_feat = train_res.to_features()
y_train = train_res.labels
y_combined_class = combined.labels

# Free per-split results (data lives on in combined / id_train_feat)
del id_test_res, ood_res

# Release GPU memory back to system after extraction
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Combined:", combined.embeddings.shape, "ood_labels:", ood_labels.shape)

# %%
# Build detectors: fit on ID **train** only, score on ID test + OOD
head = embedder._head  # noqa: SLF001 — showcase script; no public accessor yet
assert head is not None
W = head.weight.detach().cpu().numpy()
b = head.bias.detach().cpu().numpy()

detectors: dict = {}

msp = MSP()
msp.fit(id_train_feat)
detectors["MSP"] = msp

energy = Energy()
energy.fit(id_train_feat)
detectors["Energy"] = energy

maha = Mahalanobis()
maha.fit(id_train_feat, y=y_train)
detectors["Mahalanobis"] = maha

knn = KNN(k=min(10, len(train_ds)), backend="auto")
knn.fit(id_train_feat)
detectors["KNN"] = knn

pca = PCA(kernel="linear")
pca.fit(id_train_feat)
detectors["PCA"] = pca

pcaf = PCAFusion(kernel="linear")
pcaf.fit(id_train_feat)
detectors["PCAFusion"] = pcaf

vim = ViM(W, b)
vim.fit(id_train_feat)
detectors["ViM"] = vim

wd = WDiscOOD()
wd.fit(id_train_feat, y=y_train)
detectors["WDiscOOD"] = wd

# Free training features/labels — detectors keep only what they need internally
del id_train_feat, y_train, train_res, W, b

# %%
bank = ScoreBank(ood_labels=ood_labels, class_labels=y_combined_class)
for name, det in detectors.items():
    bank.add(name, det.score(comb_feat))

table = evaluate(bank)
print(table)

# %% [markdown]
# Higher scores ⇒ more OOD. **AUROC** / **FPR@95** / **AUPR** use `ood_labels`: ID test = 0, ImageNet-O = 1.

# %%
# from oodkit.evaluation import evaluate_by_class
# by_cls = evaluate_by_class(bank)
