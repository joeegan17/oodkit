# OODKit

Out-of-distribution detection library for computer vision.

## Installation

```bash
pip install -e .
```

## Usage

```python
from oodkit.detectors import ViM, MSP, Energy, BaseDetector
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

- `oodkit/detectors/` — OOD detectors (ViM, MSP, Energy, …) with sklearn-style fit/score/predict
- `oodkit/data/` — Features container for logits and embeddings
- `oodkit/utils/` — Linear algebra and other helpers
