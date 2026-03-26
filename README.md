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

## Package structure

- `oodkit/detectors/` — OOD detectors (ViM, MSP, Energy, …) with sklearn-style fit/score/predict
- `oodkit/data/` — Features container for logits and embeddings
- `oodkit/utils/` — Linear algebra and other helpers
