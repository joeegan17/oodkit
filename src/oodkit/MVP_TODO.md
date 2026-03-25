# OODKit MVP TODO

This file tracks improvements we want after the initial MVP scaffold.

## API and Naming
- Keep `Features` as the unified model-output container (logits and/or embeddings).
- Continue using explicit local names like `embeddings` when operating only on embedding tensors/arrays.
- Revisit whether `BaseDetector.predict()` should define a default threshold contract across all detectors.

## Input Type Flexibility
- Keep detector internals NumPy-first for MVP simplicity.
- Accept torch tensors at detector API boundaries and convert once to NumPy.
- Standardize conversion behavior in shared utilities (not per-detector) to avoid drift.
- Decide and document expected output types (currently NumPy arrays).

## Detector Roadmap
- Add detectors: `Energy`, `MSP`, `Mahalanobis`, and KNN-style embedding methods.
- Define minimal per-detector required inputs (`logits`, `embeddings`, or both) in docstrings.
- Add simple threshold selection helpers (global percentile / validation-driven), while keeping raw scores first-class.

## Data and Feature Generation
- Add feature/logit generation utilities for foundation models (planned: DINOv3).
- Add optional supervised contrastive fine-tuning for classifier heads to improve embeddings.

## Object Detection Use Cases
- Add utilities to chip images to GT boxes with minimum size (25x25).
- Run OOD scoring on chips and aggregate to image-level scores (mean/max/other reducers).
- Document assumptions for detection datasets and model outputs.

## Reliability and Tests
- Add basic unit tests for shape validation and fittedness checks.
- Add small synthetic tests for `fit()`, `score()`, and `predict()` behavior.
- Add smoke tests for NumPy input and torch-tensor input conversion paths.

## Completed
(none yet)

## Later / Needs confirmation
