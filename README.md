# OODKit

Out-of-distribution detection library for computer vision.

## Installation

```bash
pip install -e .
```

## Usage

```python
from oodkit.detectors import ViM, MSP, Energy, Mahalanobis, KNN, BaseDetector
from oodkit.data import Features

# Create features (logits, embeddings, or both)
features = Features(logits=..., embeddings=...)

# ViM: fit on ID data, then score
detector = ViM(W, b, D)
detector.fit(features)
scores = detector.score(features)

# MSP / Energy: logits only; fit() is a no-op; optional temperature
msp = MSP(temperature=1.0)
msp.fit(features)
scores_msp = msp.score(features)

# Mahalanobis: embeddings + optional class labels at fit time
maha = Mahalanobis(eps=1e-6)
maha.fit(Features(embeddings=...), y=...)  # y defaults to single Gaussian if omitted
scores_maha = maha.score(Features(embeddings=...))

# KNN: embeddings only, score = avg distance to k nearest ID embeddings
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

## Package structure

- `oodkit/detectors/` — OOD detectors (ViM, MSP, Energy, Mahalanobis, KNN, …) with sklearn-style fit/score/predict
- `oodkit/data/` — Features container for logits and embeddings
- `oodkit/utils/` — Linear algebra and other helpers
