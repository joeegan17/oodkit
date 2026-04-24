# Notebooks

## ImageNet val vs ImageNet-O (`imagenet_ood_showcase.ipynb`)

Public notebook with runnable cells. The ignored `.py` sibling is the agent-friendly editing source.

Edit **paths and hyperparameters** at the top of the first code cell (`DATASETS_ROOT`, `IMAGENET_VAL_ROOT`, `IMAGENET_O_ROOT`, `LOC_SYNSET_MAPPING`, `HEAD_EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, `PERSISTENT_WORKERS`, `TRAIN_FRACTION`).

Uses the **full** ImageNet-O tree. Val is split **90% train / 10% ID test** by default (`TRAIN_FRACTION`).

```bash
pip install -e ".[dev]"
# open imagenet_ood_showcase.ipynb and run cells top to bottom
```

Trains a **linear head** for 5 epochs (by default) on **DINOv2** (`dinov2-small`), fits detectors on the val train split, scores **held-out val + full ImageNet-O**, and prints `evaluate` metrics for all detectors. To try **DINOv3**, set `BACKBONE = "dinov3-small"` after Hugging Face access is set up (see roadmap).

## COCO vs COCO-O object-detection domain (`coco_ood_showcase.ipynb`)

Chip-level demo for the object-detection domain. Reads COCO train / val annotations and one or more COCO-O OOD domains (`cartoon`, `tattoo`, `weather`, ...), crops square chips around ground-truth boxes (no stretching), and evaluates **Energy**, **WDiscOOD**, and **ViM** at two granularities:

- **chip-level** — every box is one sample.
- **image-level** — chip scores pooled via `pool_image_scores` (`mean` / `max` / `topk_mean`), with a per-OOD-domain breakdown using the `group` metadata.
- **per (class, OOD group)** — chip-level AUROC table + heatmap for the most frequent classes, so you can tell which domain × class pairs each detector struggles with.
- **score KDE grid** — `score_kde_grid(detector, classes=..., groups=...)` renders a `class x OOD group` grid of ID-vs-group KDE overlays (rows share x-axes) so you can see *why* the AUROC numbers look the way they do. One figure per detector.
- **visual inspection** — `display_chips(detector, group=..., class_name=..., truth=..., rank_range=..., direction=...)` renders a grid of chip crops ranked by a detector's score, with titles showing class / domain / true ID-OOD / score. One sample cell per detector.

Edit **paths and hyperparameters** at the top of the first code cell (`COCO_ROOT`, `COCO_O_ROOT`, `OOD_DOMAINS`, `MIN_BOX_SIDE`, `MIN_CHIP_SIZE`, `TRAIN_IMAGE_FRACTION`, `VAL_IMAGE_FRACTION`, `POOL_TOPK`).

Expected directory layout:

```
<COCO_ROOT>/
  coco_annotations/
    instances_train2017.json
    instances_val2017.json
  coco_train/                       # image files
  coco_val/                         # image files

<COCO_O_ROOT>/
  <domain>/                         # e.g. cartoon, tattoo, weather
    annotations/instances_val2017.json
    images/                         # image files (renamed from val2017/)
```

```bash
pip install -e ".[dev]"
# open coco_ood_showcase.ipynb and run cells top to bottom
```

No `pycocotools` dependency — `oodkit.contrib.coco` parses the JSON directly. See `OBJECT_DETECTION_PLAN.md` for design details and deferred research directions.
