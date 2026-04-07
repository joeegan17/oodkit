# Notebooks

## ImageNet val vs ImageNet-O (`imagenet_ood_showcase.py`)

Interactive script with `# %%` cells (open in VS Code / Cursor and use **Run Cell**, or paste into Jupyter).

Edit **paths and hyperparameters** at the top of the first code cell (`DATASETS_ROOT`, `IMAGENET_VAL_ROOT`, `IMAGENET_O_ROOT`, `LOC_SYNSET_MAPPING`, `HEAD_EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, `PERSISTENT_WORKERS`, `TRAIN_FRACTION`).

Uses the **full** ImageNet-O tree. Val is split **90% train / 10% ID test** by default (`TRAIN_FRACTION`).

```bash
pip install -e ".[ml]"
# open imagenet_ood_showcase.py and run cells top to bottom
```

Trains a **linear head** for 5 epochs (by default) on **DINOv2** (`dinov2-small`), fits detectors on the val train split, scores **held-out val + full ImageNet-O**, and prints `evaluate` metrics for all detectors. To try **DINOv3**, set `BACKBONE = "dinov3-small"` after Hugging Face access is set up (see roadmap).
