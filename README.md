# OODKit

Out-of-distribution detection library for computer vision.

Architecture, module relationships, and agent onboarding: [`REPO_GUIDE.md`](REPO_GUIDE.md).

## Installation

```bash
pip install -e .
```

## Usage

```python
from oodkit.detectors import ViM, MSP, Energy, Mahalanobis, KNN, PCA, PCAFusion, BaseDetector
from oodkit.data import Features

# Create features (logits, embeddings, or both)
features = Features(logits=..., embeddings=...)

# ViM: fit on ID data; default keeps components until 95% cumulative variance (pct_variance)
detector = ViM(W, b)
detector.fit(features)
scores = detector.score(features)

# PCA: reconstruction error only (linear, cosine/CoP, or rff_cosine/CoRP); higher score = more OOD
pca = PCA(kernel="linear")
pca.fit(Features(embeddings=...))
scores_pca_only = pca.score(Features(embeddings=...))

# PCAFusion (Guan et al. ICCV 2023): same kernels as PCA + log-sum-exp fusion; score negated so higher = more OOD
pca_f = PCAFusion(kernel="linear")
pca_f.fit(Features(embeddings=...))
scores_pca = pca_f.score(Features(embeddings=..., logits=...))
labels_pca = pca_f.predict(Features(embeddings=..., logits=...), threshold=...)  # calibrate threshold on validation

# MSP / Energy: logits only; fit() is a no-op; optional temperature
msp = MSP(temperature=1.0)
msp.fit(features)
scores_msp = msp.score(features)

# Mahalanobis: embeddings + optional class labels at fit time
maha = Mahalanobis(eps=1e-6)
maha.fit(Features(embeddings=...), y=...)  # y defaults to single Gaussian if omitted
scores_maha = maha.score(Features(embeddings=...))

# KNN: embeddings only, score = distance to k-th nearest ID embedding
knn = KNN(k=10, backend="auto", metric="cosine")  # metric passed through when sklearn backend is used
knn.fit(Features(embeddings=...))
scores_knn = knn.score(Features(embeddings=...))

# Distance-based detectors require an explicit calibrated threshold for predict()
labels_maha = maha.predict(Features(embeddings=...), threshold=...)
labels_knn = knn.predict(Features(embeddings=...), threshold=...)
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Some tests require **PyTorch** (`torch`); they are skipped if it is not installed. To run the full suite:

```bash
pip install -e ".[dev,ml]"
pytest
```

With Docker: `docker compose run --rm dev pytest` (torch tests skipped unless you install `torch` in the image or extend the Dockerfile with `.[ml]`).

Tests mirror `src/oodkit/` under `tests/pkg/` (e.g. `tests/pkg/detectors/test_msp.py`). The folder is named `pkg` so it does not shadow the installed `oodkit` package on `sys.path`. Shared synthetic data and `Features` bundles live in `tests/conftest.py`.

## Demo script

- [`notebooks/imagenet_ood_showcase.py`](notebooks/imagenet_ood_showcase.py) — train a classifier head on ImageNet val, run multiple OOD detectors, and compare ID vs ImageNet-O with `ScoreBank` / `evaluate` (see [`notebooks/README.md`](notebooks/README.md)).

## Package structure

- `oodkit/detectors/` — OOD detectors (ViM, MSP, Energy, Mahalanobis, KNN, PCA, PCAFusion, …) with sklearn-style fit/score/predict
- `oodkit/data/` — Features container for logits and embeddings
- `oodkit/embeddings/` — `Embedder`, disk-backed extraction, optional training (`pip install ".[ml]"`)
- `oodkit/evaluation/` — `ScoreBank`, metrics, plots, and helpers to merge ID/OOD `EmbeddingResult` blocks (`concatenate_embedding_results`, `ood_labels_from_counts`, …)
- `oodkit/contrib/` — Optional cookbook code (not required for core usage). `oodkit.contrib.imagenet` — `LOC_synset_mapping.txt` → `SynsetTable`, `SynsetImageDataset` / `imagenet_variant_dataset` for ImageFolder-style ImageNet variants with **canonical** class indices (needs torch/torchvision)
- `oodkit/utils/` — Linear algebra and other helpers
