# Notebooks

## ImageNet Val Vs ImageNet-O (`imagenet_ood_showcase.ipynb`)

Public notebook with runnable cells. The `.ipynb` file is the source of truth.

Edit **paths and hyperparameters** at the top of the first code cell (`DATASETS_ROOT`, `IMAGENET_VAL_ROOT`, `IMAGENET_O_ROOT`, `LOC_SYNSET_MAPPING`, `HEAD_EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, `PERSISTENT_WORKERS`, `TRAIN_FRACTION`).

Uses the **full** ImageNet-O tree. Val is split **90% train / 10% ID test** by default (`TRAIN_FRACTION`).

```bash
pip install -e ".[dev]"
# open imagenet_ood_showcase.ipynb and run cells top to bottom
```

Trains a **linear head** for 5 epochs by default on **DINOv2** (`dinov2-small`), fits detectors on the val train split, scores **held-out val + full ImageNet-O**, and prints `evaluate` metrics for all detectors. To try **DINOv3**, set `BACKBONE = "dinov3-small"` after Hugging Face access is set up.

## COCO Vs COCO-O Object-Detection Domain (`coco_ood_showcase.ipynb`)

Chip-level demo for the object-detection domain. Reads COCO train / val annotations and one or more COCO-O OOD domains (`cartoon`, `tattoo`, `weather`, ...), crops square chips around ground-truth boxes without stretching, and evaluates **Energy**, **WDiscOOD**, and **ViM** at two granularities:

- **chip-level** - every box is one sample.
- **image-level** - chip scores pooled via `pool_image_scores` (`mean` / `max` / `topk_mean`), with a per-OOD-domain breakdown using the `group` metadata.
- **per (class, OOD group)** - chip-level AUROC table and heatmap for the most frequent classes, so you can tell which domain x class pairs each detector struggles with.
- **score KDE grid** - `score_kde_grid(detector, classes=..., groups=...)` renders a `class x OOD group` grid of ID-vs-group KDE overlays. One figure per detector.
- **visual inspection** - `display_chips(detector, group=..., class_name=..., truth=..., rank_range=..., direction=...)` renders chip crops ranked by detector score.

Edit **paths and hyperparameters** at the top of the first code cell (`COCO_ROOT`, `COCO_O_ROOT`, `OOD_DOMAINS`, `MIN_BOX_SIDE`, `MIN_CHIP_SIZE`, `TRAIN_IMAGE_FRACTION`, `VAL_IMAGE_FRACTION`, `POOL_TOPK`).

Expected directory layout:

```text
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

No `pycocotools` dependency - `oodkit.contrib.coco` parses the JSON directly.

## Geometry-Aware Pooling Research (`coco_geometry_pooling_vim.ipynb`)

Focused COCO / COCO-O research notebook for comparing image-level pooling methods on top of **ViM** chip scores.

It keeps the pipeline narrow: train/extract COCO chips, fit ViM, then compare:

- `mean`
- `topk_mean`
- `GeometryAwarePooler`

The notebook prints image-level metrics and visualizes ranked examples where simple pooling and geometry-aware pooling agree or disagree. It is intended for iterating on scene-level OOD scoring, not as a broad detector tutorial.

## Global + Chip OOD Fusion (`coco_global_chip_fusion.ipynb`)

Focused COCO / COCO-O research notebook for combining whole-image embedding OOD scores with chip-level **ViM** scores.

It trains the chip head once, extracts embeddings for both chips and full parent images, compares non-logit global image detectors (`KNN`, `Mahalanobis`, `WDiscOOD`), then evaluates:

- mean chip pooling
- top-k chip pooling
- global image embedding OOD
- global + mean chip
- global + top-k chip
- global + chip + geometry
- geometry only

This is the preferred notebook for testing whether global image context helps COCO-O before moving to datasets with stronger scene-layout shift.

## Detection Failure Prediction (`coco_detection_failure_prediction.ipynb`)

Focused COCO / COCO-O notebook for the image-class failure-prediction pipeline.

It trains/reference-fits on a COCO train ID subset, then evaluates a pretrained torchvision detector on COCO val plus the COCO-O `cartoon` domain. Detections are evaluated against GT into canonical `image_id x class_id` rows, DINOv2 + ViM scores are computed at both full-image and detection-chip scale, and grouped logistic baselines compare:

- failure-prior baseline
- class identity only
- confidence only
- OOD only
- confidence + OOD
- confidence + OOD + class

It also includes per-domain metrics, high-confidence failures, same-confidence OOD quantile comparisons, and a qualitative gallery. This is the first milestone for the claim that inference-time multi-scale OOD signals can surface the classes within shifted images where an object detector is likely to fail.
