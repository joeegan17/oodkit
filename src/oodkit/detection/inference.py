"""Lightweight object-detection inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd


def run_torchvision_detector(
    image_paths: Sequence[Union[str, Path]],
    *,
    model_name: str = "fasterrcnn_resnet50_fpn_v2",
    score_threshold: float = 0.05,
    device: str = "auto",
    batch_size: int = 1,
    image_ids: Optional[Sequence[object]] = None,
    category_table: Optional[Any] = None,
    detector_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Run a pretrained torchvision detector and return a prediction table.

    The output is intentionally plain: one row per detection with ``xyxy``
    boxes, confidence, class id/name, source image metadata, and a stable
    ``detection_id``. When a COCO ``CocoCategoryTable`` is supplied, torchvision
    class names are remapped to OODKit's contiguous class ids.
    """
    if score_threshold < 0:
        raise ValueError("score_threshold must be nonnegative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    try:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F
    except ImportError as exc:  # pragma: no cover - exercised by env
        raise ImportError(
            "run_torchvision_detector requires torch, torchvision, and pillow"
        ) from exc

    paths = [Path(p) for p in image_paths]
    if image_ids is None:
        ids = [p.stem for p in paths]
    else:
        if len(image_ids) != len(paths):
            raise ValueError("image_ids length must match image_paths length")
        ids = list(image_ids)

    resolved_device = _resolve_device(device, torch)
    model, categories = _load_torchvision_model(model_name)
    model.to(resolved_device)
    model.eval()

    name_to_idx = _category_name_to_idx(category_table)
    detector_label = detector_name or model_name

    rows: list[dict[str, object]] = []
    det_counter = 0
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            images = []
            sizes: list[tuple[int, int]] = []
            for path in batch_paths:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    sizes.append((int(rgb.width), int(rgb.height)))
                    images.append(F.to_tensor(rgb).to(resolved_device))

            outputs = model(images)
            for image_id, path, (width, height), output in zip(
                batch_ids, batch_paths, sizes, outputs
            ):
                boxes = output.get("boxes").detach().cpu().numpy()
                scores = output.get("scores").detach().cpu().numpy()
                labels = output.get("labels").detach().cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    conf = float(score)
                    if conf < score_threshold:
                        continue
                    raw_label = int(label)
                    class_name = _torchvision_class_name(raw_label, categories)
                    class_id = name_to_idx.get(class_name, raw_label)
                    rows.append(
                        {
                            "image_id": image_id,
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "bbox": [float(v) for v in box.tolist()],
                            "confidence": conf,
                            "detection_id": f"{image_id}_det_{det_counter}",
                            "image_path": str(path),
                            "image_width": width,
                            "image_height": height,
                            "detector_name": detector_label,
                            "dataset_name": dataset_name,
                        }
                    )
                    det_counter += 1

    return pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "class_id",
            "class_name",
            "bbox",
            "confidence",
            "detection_id",
            "image_path",
            "image_width",
            "image_height",
            "detector_name",
            "dataset_name",
        ],
    )


def _resolve_device(device: str, torch_module: Any) -> Any:
    if device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device)


def _load_torchvision_model(model_name: str) -> tuple[Any, Sequence[str]]:
    try:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            RetinaNet_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
            retinanet_resnet50_fpn_v2,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torchvision detection models are required") from exc

    key = model_name.lower()
    if key in {"fasterrcnn", "fasterrcnn_resnet50_fpn_v2"}:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        return fasterrcnn_resnet50_fpn_v2(weights=weights), weights.meta["categories"]
    if key in {"retinanet", "retinanet_resnet50_fpn_v2"}:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        return retinanet_resnet50_fpn_v2(weights=weights), weights.meta["categories"]
    raise ValueError(
        "model_name must be one of {'fasterrcnn_resnet50_fpn_v2', "
        "'retinanet_resnet50_fpn_v2'}"
    )


def _category_name_to_idx(category_table: Optional[Any]) -> Mapping[str, int]:
    if category_table is None:
        return {}
    if hasattr(category_table, "idx_to_name"):
        return {str(name): i for i, name in enumerate(category_table.idx_to_name)}
    if isinstance(category_table, Mapping):
        return {str(name): int(idx) for name, idx in category_table.items()}
    raise TypeError("category_table must be a CocoCategoryTable or name->idx mapping")


def _torchvision_class_name(label: int, categories: Sequence[str]) -> str:
    if 0 <= int(label) < len(categories):
        return str(categories[int(label)])
    return str(int(label))


__all__ = ["run_torchvision_detector"]
