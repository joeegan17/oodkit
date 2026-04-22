# OODKit repository guide

Living orientation for humans and agents: **what this repo is**, **how packages fit together**, and **where to change things**. Update this file when you make structural or API changes that would mislead a reader of an older version.

---

## Purpose

**OODKit** is a Python library for **out-of-distribution (OOD) detection** in computer vision. Users obtain model outputs (class logits and/or embedding vectors) on in-distribution (ID) and OOD data, fit detectors on ID features, score new samples (higher score = more OOD), optionally threshold to binary ID/OOD predictions, and evaluate or compare methods.

---

## Implementation goals

- **Usable:** One sklearn-style contract for detectors (`fit` / `score` / `predict` on `Features`), a single evaluation handoff (`ScoreBank`), and clear bridges from embedding extraction to detectors.
- **Efficient and practical:** NumPy-first internals; `ScoreBank` normalizes to compact dtypes; optional heavy ML stack isolated behind `pip install oodkit[ml]`; disk-backed embedding extraction and **reload without torch** via `load_embeddings`.

---

## Repository layout

| Path | Role |
|------|------|
| `src/oodkit/` | Installable package (setuptools `package-dir = src`). |
| `tests/pkg/` | Tests mirroring `src/oodkit/`; folder name **`pkg`** avoids shadowing the installed `oodkit` on `sys.path`. |
| `notebooks/` | End-to-end demos (e.g. ImageNet OOD showcase). |
| `docker/` + `docker-compose.yml` | Dev / Jupyter GPU environment. |
| `pyproject.toml` | Core deps: `numpy`, `matplotlib`. Optional `[ml]` adds torch, transformers, sklearn, etc. |

---

## Package map and data flow

### 1. `oodkit.data` — `Features`

**`Features(logits=..., embeddings=...)`** is the universal input to detectors. At least one field is required. Naming: “features” means general model outputs, not embeddings only. Normalization/preprocessing is **caller-controlled** unless a specific detector defines otherwise.

### 2. `oodkit.detectors` — OOD scores

All detectors subclass **`BaseDetector`**: learn from ID `Features` in `fit`, return per-sample scores in `score` (convention: **higher = more OOD**), optional `predict` with an explicit threshold where needed.

Concrete detectors live under `src/oodkit/detectors/` (e.g. **MSP**, **Energy**, **ViM**, **Mahalanobis**, **KNN**, **PCA**, **PCAFusion**, **WDiscOOD**). Each implementation chooses logits vs embeddings (or both) from `Features`.

**Public re-exports** from `oodkit` root (`src/oodkit/__init__.py`) are the main ergonomic entry for the core API.

### 3. `oodkit.embeddings` — from images to `Features`

- **`EmbeddingResult`** (`embeddings`, optional `logits` / `labels` / `metadata`) is the output of extraction. **`to_features()`** bridges to **`Features`** for detectors.
- **`Embedder`** is **lazy-imported** (`embeddings/__init__.py` uses `__getattr__`) so importing `EmbeddingResult` or `load_embeddings` does not require torch.
- **`oodkit.embeddings._guard`** calls **`require_ml_deps()`** at import time in torch-dependent modules so the rest of the package stays importable without `[ml]`.
- **`load_embeddings`** (`storage.py`) reads manifests + `.npy` written by `Embedder.extract(..., save_to=...)` using **only NumPy**.

Supporting pieces: **`backbones`** (HF presets), **`datasets`** / dataloaders, **`training`** (optional head or full finetune).

### 4. `oodkit.evaluation` — metrics and comparisons

**`ScoreBank`** is the hub: register multiple detector score vectors plus optional sample-aligned metadata — `ood_labels`, `class_labels`, `class_names` (int-label → name), `groups` (per-sample string tags, e.g. OOD domain names), and arbitrary `sample_metrics`. **`evaluate`**, **`evaluate_by_class`**, low-level **`roc_curve`** / **`auroc`**, **`compare`** helpers (`rank_samples` supports `rank_range`), **`performance`**, and **`plots`** all consume a bank so callers do not manually align arrays. **`by_class(name_or_int)`** and **`by_group(name)`** return sliced banks.

**Visualization:** **`plots.score_distributions(kind="hist"|"kde", standardize=...)`** overlays ID vs OOD score distributions per detector, optionally z-scored against the ID pool so axes are comparable across detectors. **`plots.rank_grid(bank, detector, images=..., rank_range=..., class_name=..., group=..., truth=...)`** renders a ranked sample grid with filters; `images` can be a list of PILs/paths or any `__getitem__`-indexable loader (the COCO notebook plugs in an on-the-fly chip cropper in three lines).

**`combine`** provides **`concatenate_embedding_results`** and helpers to build **`ood_labels`** vectors when ID and OOD runs are separate blocks. It also merges OD metadata (`chip_to_image` with per-block offsets, `boxes`, `object_ids`, `group`, `image_ids`) so chip-level analysis survives multi-block concatenation.

**`pool_image_scores`** (`oodkit.evaluation.pooling`) aggregates chip-level scores into image-level scores (`mean`, `max`, `topk_mean`) using `metadata["chip_to_image"]`. Compose with `ScoreBank` twice — once per chip, once per image — to evaluate both granularities in a single pipeline.

### 5. Object-detection chip pipeline (`oodkit.data` + `oodkit.contrib.coco`)

OD support is layered so most of the library stays image-agnostic:

- **`oodkit.data.chips`** — pure-NumPy primitives (`to_xyxy`, `filter_small_boxes`, `square_chip_regions`, `crop_chip`). No torch, no PIL. Chipping rule: every chip is a **square** with side `max(longest_box_side, min_chip_size)` (default `25`), centered on the box, zero-padded at edges.
- **`oodkit.data.chip_dataset.ChipDataset`** — `torch.utils.data.Dataset` that flattens per-image boxes into a stream of square chips. Exposes `chip_to_image`, `boxes`, `labels`, `groups`, `image_ids`, and `object_ids` for downstream propagation. `object_ids` follow `{image_id}[_{class_name}][_{group}]_{order}` (e.g. `00067_cat_cartoon_0`).
- **`Embedder.extract`** duck-types `ChipDataset`-shaped inputs via `_extract_chip_metadata` and forwards all OD fields into `EmbeddingResult.metadata` (arrays for `chip_to_image` / `boxes`; Python lists for `image_paths` / `image_ids` / `group` / `object_ids`). `save_to=...` serializes the lists as JSON and updates the manifest so `load_embeddings` round-trips them without torch.
- **`oodkit.contrib.coco`** — COCO / COCO-O ingestion (no `pycocotools` dependency). Pure-Python pieces (`CocoCategoryTable`, `load_coco`, `discover_coco_id`, `discover_coco_ood`) are importable without `[ml]`; chip-dataset builders (`coco_chip_dataset`, `coco_id_chip_datasets`, `coco_ood_chip_datasets`) lazy-import torch. The OOD builder tags each domain (`cartoon`, `tattoo`, ...) into `ChipDataset.groups`, which survives all the way into `EmbeddingResult.metadata["group"]` for per-domain slicing.

See `OBJECT_DETECTION_PLAN.md` for the phase-by-phase spec, assumed COCO / COCO-O directory layouts, and deferred research directions (context-aware pooling).

### 6. `oodkit.contrib`

Optional, cookbook-style code **not required** for core library use. **`oodkit.contrib.imagenet`** maps ImageNet LOC synsets to canonical indices and offers dataset helpers (expects torch/torchvision where used). **`oodkit.contrib.coco`** covers object-detection ingestion — see the section above.

### 7. `oodkit.utils` and `oodkit.types`

Shared helpers (**array** coercion, **linalg**) and typing aliases (**`ArrayLike`**) used across detectors and evaluation.

---

## Typical workflows (mental model)

1. **Already have logits/embeddings:** Build **`Features`** → **`detector.fit`** on ID → **`detector.score`** on ID/OOD → optional **`predict`** → feed scores into **`ScoreBank`** → **`evaluation.evaluate`** / plots.
2. **Start from images:** **`pip install oodkit[ml]`** → **`Embedder`** (`fit` optional) → **`extract`** → **`EmbeddingResult.to_features()`** → same as (1). Large runs: **`save_to`** + **`load_embeddings`** on analysis machines without GPU.

---

## Tests and quality

- **`pytest`**; config in `pyproject.toml` (`testpaths = ["tests"]`).
- Tests that need torch are skipped when torch is missing; full run: **`pip install -e ".[dev,ml]"`** then **`pytest`**.
- Shared fixtures and synthetic **`Features`** live in **`tests/conftest.py`**.

---

## Extension points (for implementers)

- **New detector:** Implement **`BaseDetector`**; accept **`Features`**; document logits vs embedding requirements; add tests under **`tests/pkg/detectors/`**.
- **New backbone preset:** Register in **`embeddings/backbones.py`** and document in README / here if behavior or deps change.
- **New evaluation surface:** Prefer extending **`ScoreBank`** ingestion or functions that take **`ScoreBank`** so multi-detector workflows stay aligned.

---

## Related docs

- **`README.md`** — install, quick API examples, test notes, high-level package structure.
- **`ROADMAP.md`** — shipped MVP summary and backlog (do not use closed `MVP_TODO.md` for new items).
- **`notebooks/README.md`** — demo entrypoints.

When in doubt, **grep `from oodkit` in `tests/pkg/`** for usage patterns.
