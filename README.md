# OODKit

Out-of-distribution detection library for computer vision.

## Installation

```bash
pip install -e .
```

## Usage

```python
from oodkit.detectors import ViM, BaseDetector
from oodkit.data import Features

# Create features (logits, embeddings, or both)
features = Features(logits=..., embeddings=...)

# Fit detector on in-distribution data
detector = ViM()
detector.fit(features)

# Score and predict
scores = detector.score(features)
```

## Package structure

- `oodkit/detectors/` — OOD detectors (ViM, etc.) with sklearn-style fit/score/predict
- `oodkit/data/` — Features container for logits and embeddings
- `oodkit/utils/` — Linear algebra and other helpers
