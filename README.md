# OODKit

A small Python library for **out-of-distribution (OOD) detection** in computer
vision. Fit a detector on in-distribution features, score new samples (higher
score = more OOD), and evaluate or compare methods — with a consistent
sklearn-style API.

OODKit works on either image-classification or object-detection data, and can
take you from raw images to scored samples in a few steps.

## Install

```bash
pip install -e .              # runtime dependencies for detectors, embeddings, plots, and examples
pip install -e ".[dev]"       # adds test + notebook tooling
```

## What's in the box

- **Detectors** (`oodkit.detectors`) — a family of OOD scorers (MSP, Energy,
  Mahalanobis, KNN, PCA / CoP / CoRP, PCAFusion, ViM, WDiscOOD) with a shared
  `fit` / `score` / `predict` contract.
- **Features** (`oodkit.data`) — a single `Features(logits=..., embeddings=...)`
  container that every detector accepts.
- **Embedder** (`oodkit.embeddings`) — optional helper that turns a dataset of
  images into logits/embeddings via a pretrained backbone, with disk-backed
  extraction so analysis can run on machines without a GPU.
- **Evaluation** (`oodkit.evaluation`) — `ScoreBank` aligns scores from multiple
  detectors with labels and metadata; metrics (`evaluate`, `evaluate_by_class`)
  and plots (`score_distributions`, `rank_grid`, ROC / PR / correlation) all
  consume a bank.
- **Object-detection support** — chip utilities (`oodkit.data.chips`,
  `ChipDataset`), image-level pooling (`pool_image_scores`), and
  `oodkit.contrib.coco` for COCO / COCO-O ingestion without `pycocotools`.

## Minimal example

```python
from oodkit.data import Features
from oodkit.detectors import Energy
from oodkit.evaluation import ScoreBank, evaluate

# You supply logits / embeddings however you like (numpy, torch, Embedder, ...).
id_feat  = Features(logits=id_logits,  embeddings=id_embeddings)
ood_feat = Features(logits=ood_logits, embeddings=ood_embeddings)

det = Energy()
det.fit(id_feat)

scores_id  = det.score(id_feat)
scores_ood = det.score(ood_feat)

bank = ScoreBank(ood_labels=[0] * len(scores_id) + [1] * len(scores_ood))
bank.add("Energy", [*scores_id, *scores_ood])
print(evaluate(bank))   # AUROC, FPR@95, AUPR, ...
```

For a full end-to-end run (images → extraction → multiple detectors →
comparison plots), see [`notebooks/`](notebooks/README.md):

- [`imagenet_ood_showcase.ipynb`](notebooks/imagenet_ood_showcase.ipynb) —
  classification OOD on ImageNet vs ImageNet-O.
- [`coco_ood_showcase.ipynb`](notebooks/coco_ood_showcase.ipynb) —
  object-detection OOD on COCO vs COCO-O (chips, image-level pooling,
  per-domain breakdowns, ranked chip galleries).

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests mirror `src/oodkit/` under `tests/pkg/`. Tests that need torch skip
cleanly if the runtime ML stack isn't installed. The recommended development
workflow uses Docker: `docker compose run --rm dev pytest`.

## Design notes and deeper reading

- [`REPO_GUIDE.md`](REPO_GUIDE.md) — architecture, module boundaries, and
  extension points. Start here if you're contributing or exploring internals.
- [`ROADMAP.md`](ROADMAP.md) — what's shipped and what's on deck.
- [`OBJECT_DETECTION_PLAN.md`](OBJECT_DETECTION_PLAN.md) — design notes for the
  OD pipeline, including deferred research directions.
