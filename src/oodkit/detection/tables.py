"""Tabular object-detection evaluation and OOD feature aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from oodkit.data.chip_dataset import ChipImageAnn


@dataclass
class DetectionEvaluationTables:
    """Output tables for object-detection failure analysis."""

    image_class_metrics: pd.DataFrame
    detections_enriched: pd.DataFrame
    ground_truth_enriched: pd.DataFrame


_PRED_COLUMNS = ("image_id", "class_id", "bbox")
_GT_COLUMNS = ("image_id", "class_id", "bbox")


def evaluate_detection_tables(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
    backend: str = "auto",
    class_names: Optional[Mapping[int, str]] = None,
    detector_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    low_precision_threshold: float = 0.5,
    low_recall_threshold: float = 0.5,
) -> DetectionEvaluationTables:
    """Match detections to GT and build canonical failure-analysis tables.

    ``backend="auto"`` uses FiftyOne when available and the inputs contain
    image-size columns required for normalized boxes; otherwise it uses the
    built-in greedy per-image/per-class matcher. The built-in matcher is kept
    as a deterministic fallback for tests and lightweight demos.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    pred = _prepare_predictions(predictions, class_names=class_names)
    gt = _prepare_ground_truth(ground_truth, class_names=class_names)

    if backend not in {"auto", "simple", "fiftyone"}:
        raise ValueError("backend must be one of {'auto', 'simple', 'fiftyone'}")

    use_fiftyone = backend == "fiftyone" or (
        backend == "auto" and _can_use_fiftyone(pred, gt)
    )
    if use_fiftyone:
        try:
            det_enriched, gt_enriched = _evaluate_with_fiftyone(
                pred, gt, iou_threshold=iou_threshold
            )
        except Exception:
            if backend == "fiftyone":
                raise
            det_enriched, gt_enriched = _evaluate_simple(
                pred, gt, iou_threshold=iou_threshold
            )
    else:
        det_enriched, gt_enriched = _evaluate_simple(
            pred, gt, iou_threshold=iou_threshold
        )

    metrics = aggregate_image_class_metrics(
        det_enriched,
        gt_enriched,
        detector_name=detector_name,
        dataset_name=dataset_name,
        low_precision_threshold=low_precision_threshold,
        low_recall_threshold=low_recall_threshold,
    )
    return DetectionEvaluationTables(metrics, det_enriched, gt_enriched)


def aggregate_image_class_metrics(
    detections_enriched: pd.DataFrame,
    ground_truth_enriched: pd.DataFrame,
    *,
    detector_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    low_precision_threshold: float = 0.5,
    low_recall_threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate enriched detection/GT rows to one row per image/class."""
    det = detections_enriched.copy()
    gt = ground_truth_enriched.copy()

    keys = set()
    if not det.empty:
        keys.update((r.image_id, int(r.class_id)) for r in det.itertuples())
    if not gt.empty:
        keys.update((r.image_id, int(r.class_id)) for r in gt.itertuples())

    rows = []
    for image_id, class_id in sorted(keys, key=lambda x: (str(x[0]), int(x[1]))):
        d = det[(det["image_id"] == image_id) & (det["class_id"] == class_id)]
        g = gt[(gt["image_id"] == image_id) & (gt["class_id"] == class_id)]
        num_det = int(len(d))
        num_gt = int(len(g))
        tp = int(np.sum(d["match_status"].to_numpy(dtype=object) == "TP")) if num_det else 0
        fp = int(np.sum(d["match_status"].to_numpy(dtype=object) == "FP")) if num_det else 0
        fn = int(np.sum(g["match_status"].to_numpy(dtype=object) == "FN")) if num_gt else 0

        precision = tp / num_det if num_det else np.nan
        recall = tp / num_gt if num_gt else np.nan
        if np.isfinite(precision) and np.isfinite(recall):
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan

        conf = d["confidence"].astype(float).to_numpy() if num_det and "confidence" in d else np.array([])
        class_name = _first_non_null(
            list(g.get("class_name", [])) + list(d.get("class_name", []))
        )

        rows.append(
            {
                "image_id": image_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "num_gt": num_gt,
                "num_detections": num_det,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "has_failure": bool(fp > 0 or fn > 0),
                "missed_object": bool(fn > 0),
                "false_positive": bool(fp > 0),
                "low_precision": (precision < low_precision_threshold) if num_det else pd.NA,
                "low_recall": (recall < low_recall_threshold) if num_gt else pd.NA,
                "mean_detection_confidence": float(np.mean(conf)) if conf.size else np.nan,
                "max_detection_confidence": float(np.max(conf)) if conf.size else np.nan,
                "min_detection_confidence": float(np.min(conf)) if conf.size else np.nan,
                "detector_name": detector_name if detector_name is not None else _first_non_null(d.get("detector_name", [])),
                "dataset_name": dataset_name if dataset_name is not None else _first_non_null(
                    list(g.get("dataset_name", [])) + list(d.get("dataset_name", []))
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_image_class_metrics()
    out["low_precision"] = out["low_precision"].astype("boolean")
    out["low_recall"] = out["low_recall"].astype("boolean")
    return out


def aggregate_chip_scores(
    detections_enriched: pd.DataFrame,
    *,
    chip_score_col: str = "chip_ood_score",
) -> pd.DataFrame:
    """Aggregate detection/chip OOD scores by ``image_id x class_id``."""
    if detections_enriched.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "class_id",
                "chip_ood_mean",
                "chip_ood_max",
                "chip_ood_p90",
                "chip_ood_std",
                "num_chips",
            ]
        )
    if chip_score_col not in detections_enriched:
        raise ValueError(f"detections_enriched is missing {chip_score_col!r}")

    rows = []
    for (image_id, class_id), group in detections_enriched.groupby(["image_id", "class_id"], sort=True):
        scores = pd.to_numeric(group[chip_score_col], errors="coerce").to_numpy(dtype=float)
        scores = scores[np.isfinite(scores)]
        rows.append(
            {
                "image_id": image_id,
                "class_id": int(class_id),
                "chip_ood_mean": float(np.mean(scores)) if scores.size else np.nan,
                "chip_ood_max": float(np.max(scores)) if scores.size else np.nan,
                "chip_ood_p90": float(np.percentile(scores, 90)) if scores.size else np.nan,
                "chip_ood_std": float(np.std(scores)) if scores.size else np.nan,
                "num_chips": int(scores.size),
            }
        )
    return pd.DataFrame(rows)


def attach_ood_features(
    image_class_metrics: pd.DataFrame,
    *,
    global_scores: Optional[pd.DataFrame] = None,
    detections_enriched: Optional[pd.DataFrame] = None,
    global_score_col: str = "global_ood_score",
    chip_score_col: str = "chip_ood_score",
    embedding_model_name: Optional[str] = None,
    ood_detector_name: Optional[str] = None,
) -> pd.DataFrame:
    """Attach global and chip OOD features to image-class metrics."""
    out = image_class_metrics.copy()
    if global_scores is not None:
        if "image_id" not in global_scores or global_score_col not in global_scores:
            raise ValueError(
                f"global_scores must contain 'image_id' and {global_score_col!r}"
            )
        out = out.merge(
            global_scores[["image_id", global_score_col]].drop_duplicates("image_id"),
            on="image_id",
            how="left",
        )
    elif global_score_col not in out:
        out[global_score_col] = np.nan

    if detections_enriched is not None:
        chip = aggregate_chip_scores(detections_enriched, chip_score_col=chip_score_col)
        out = out.merge(chip, on=["image_id", "class_id"], how="left")
    for col in ["chip_ood_mean", "chip_ood_max", "chip_ood_p90", "chip_ood_std"]:
        if col not in out:
            out[col] = np.nan
    if "num_chips" not in out:
        out["num_chips"] = 0
    out["num_chips"] = out["num_chips"].fillna(0).astype(int)

    if embedding_model_name is not None:
        out["embedding_model_name"] = embedding_model_name
    elif "embedding_model_name" not in out:
        out["embedding_model_name"] = None
    if ood_detector_name is not None:
        out["ood_detector_name"] = ood_detector_name
    elif "ood_detector_name" not in out:
        out["ood_detector_name"] = None
    return out


def detection_chips_from_table(
    detections: pd.DataFrame,
    *,
    image_paths: Mapping[object, str],
    image_sizes: Optional[Mapping[object, Tuple[float, float]]] = None,
) -> Tuple[Sequence[ChipImageAnn], Sequence[object]]:
    """Build chip annotations from detection rows.

    Returns the annotations plus flattened detection IDs in the same order that
    ``ChipDataset`` will emit chips.
    """
    pred = _prepare_predictions(detections)
    annotations = []
    ordered_detection_ids = []
    for image_id, group in pred.groupby("image_id", sort=False):
        if image_id not in image_paths:
            raise KeyError(f"missing image path for image_id={image_id!r}")
        annotations.append(
            ChipImageAnn(
                image_path=str(image_paths[image_id]),
                boxes=np.vstack(group["bbox"].to_numpy()).astype(np.float64),
                labels=group["class_id"].to_numpy(dtype=np.int64),
                image_id=str(image_id),
                image_size=image_sizes.get(image_id) if image_sizes is not None else None,
            )
        )
        ordered_detection_ids.extend(group["detection_id"].tolist())
    return annotations, ordered_detection_ids


def _prepare_predictions(
    predictions: pd.DataFrame,
    *,
    class_names: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    pred = _ensure_columns(predictions.copy(), _PRED_COLUMNS, "predictions")
    if pred.empty:
        return _empty_predictions()
    pred["class_id"] = pred["class_id"].astype(int)
    pred["bbox"] = [_validate_box(x) for x in pred["bbox"]]
    if "detection_id" not in pred:
        pred["detection_id"] = [f"det_{i}" for i in range(len(pred))]
    pred["detection_id"] = pred["detection_id"].astype(str)
    if "confidence" not in pred:
        pred["confidence"] = np.nan
    pred["confidence"] = pd.to_numeric(pred["confidence"], errors="coerce")
    if "class_name" not in pred:
        pred["class_name"] = [
            _class_name(int(x), class_names) for x in pred["class_id"].tolist()
        ]
    return pred


def _prepare_ground_truth(
    ground_truth: pd.DataFrame,
    *,
    class_names: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    gt = _ensure_columns(ground_truth.copy(), _GT_COLUMNS, "ground_truth")
    if gt.empty:
        return _empty_ground_truth()
    gt["class_id"] = gt["class_id"].astype(int)
    gt["bbox"] = [_validate_box(x) for x in gt["bbox"]]
    if "gt_id" not in gt:
        gt["gt_id"] = [f"gt_{i}" for i in range(len(gt))]
    gt["gt_id"] = gt["gt_id"].astype(str)
    if "class_name" not in gt:
        gt["class_name"] = [_class_name(int(x), class_names) for x in gt["class_id"].tolist()]
    return gt


def _evaluate_simple(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    iou_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    det_rows = []
    gt_rows = []
    keys = set()
    if not pred.empty:
        keys.update((r.image_id, int(r.class_id)) for r in pred.itertuples())
    if not gt.empty:
        keys.update((r.image_id, int(r.class_id)) for r in gt.itertuples())

    for image_id, class_id in sorted(keys, key=lambda x: (str(x[0]), int(x[1]))):
        d = pred[(pred["image_id"] == image_id) & (pred["class_id"] == class_id)].copy()
        g = gt[(gt["image_id"] == image_id) & (gt["class_id"] == class_id)].copy()
        if not d.empty:
            d = d.sort_values("confidence", ascending=False, na_position="last")
        matched_gt = set()
        gt_match: dict[str, tuple[str, float]] = {}

        gt_boxes = list(g["bbox"]) if not g.empty else []
        gt_ids = list(g["gt_id"]) if not g.empty else []
        for det in d.itertuples():
            ious = np.array([_iou(det.bbox, b) for b in gt_boxes], dtype=float)
            if ious.size:
                order = np.argsort(ious)[::-1]
                best_all = float(ious[order[0]])
            else:
                order = np.array([], dtype=int)
                best_all = 0.0
            match_idx = None
            for idx in order:
                candidate = str(gt_ids[int(idx)])
                if candidate not in matched_gt and ious[int(idx)] >= iou_threshold:
                    match_idx = int(idx)
                    break
            row = det._asdict()
            row.pop("Index", None)
            if match_idx is None:
                row.update(
                    {
                        "match_status": "FP",
                        "matched_gt_id": None,
                        "iou_to_match": best_all,
                    }
                )
            else:
                gt_id = str(gt_ids[match_idx])
                iou = float(ious[match_idx])
                matched_gt.add(gt_id)
                gt_match[gt_id] = (str(det.detection_id), iou)
                row.update(
                    {
                        "match_status": "TP",
                        "matched_gt_id": gt_id,
                        "iou_to_match": iou,
                    }
                )
            det_rows.append(row)

        pred_boxes = list(d["bbox"]) if not d.empty else []
        for truth in g.itertuples():
            row = truth._asdict()
            row.pop("Index", None)
            gt_id = str(truth.gt_id)
            best_iou = max([_iou(truth.bbox, b) for b in pred_boxes], default=0.0)
            if gt_id in gt_match:
                det_id, iou = gt_match[gt_id]
                row.update(
                    {
                        "match_status": "TP",
                        "matched_detection_id": det_id,
                        "best_iou": iou,
                    }
                )
            else:
                row.update(
                    {
                        "match_status": "FN",
                        "matched_detection_id": None,
                        "best_iou": best_iou,
                    }
                )
            gt_rows.append(row)

    return _finalize_detections(det_rows), _finalize_ground_truth(gt_rows)


def _evaluate_with_fiftyone(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    iou_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import fiftyone as fo

    pred, gt = pred.copy(), gt.copy()
    if not _has_image_size_columns(pred, gt):
        raise ValueError("FiftyOne backend requires image_width and image_height columns")

    eval_key = "oodkit_eval"
    dataset = fo.Dataset()
    dataset.persistent = False
    samples = []
    image_ids = sorted(set(pred["image_id"]).union(set(gt["image_id"])), key=str)
    for image_id in image_ids:
        d = pred[pred["image_id"] == image_id]
        g = gt[gt["image_id"] == image_id]
        ref = d.iloc[0] if not d.empty else g.iloc[0]
        w, h = float(ref["image_width"]), float(ref["image_height"])
        filepath = str(ref.get("image_path", Path(str(image_id)).with_suffix(".jpg")))
        sample = fo.Sample(filepath=filepath, oodkit_image_id=str(image_id))
        sample["predictions"] = fo.Detections(
            detections=[
                fo.Detection(
                    label=str(r.class_name),
                    bounding_box=_to_fo_box(r.bbox, w, h),
                    confidence=None if pd.isna(r.confidence) else float(r.confidence),
                    oodkit_detection_id=str(r.detection_id),
                    oodkit_class_id=int(r.class_id),
                )
                for r in d.itertuples()
            ]
        )
        sample["ground_truth"] = fo.Detections(
            detections=[
                fo.Detection(
                    label=str(r.class_name),
                    bounding_box=_to_fo_box(r.bbox, w, h),
                    oodkit_gt_id=str(r.gt_id),
                    oodkit_class_id=int(r.class_id),
                )
                for r in g.itertuples()
            ]
        )
        samples.append(sample)
    dataset.add_samples(samples)
    dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key=eval_key,
        method="coco",
        iou=iou_threshold,
        classwise=True,
    )

    det_rows, gt_rows = [], []
    for sample in dataset:
        image_id = sample["oodkit_image_id"]
        for det in sample["predictions"].detections:
            det_id = det["oodkit_detection_id"]
            base = pred[pred["detection_id"].astype(str) == str(det_id)].iloc[0].to_dict()
            base.update(
                {
                    "match_status": str(det.get(eval_key, "fp")).upper(),
                    "matched_gt_id": det.get(f"{eval_key}_id", None),
                    "iou_to_match": float(det.get(f"{eval_key}_iou", 0.0) or 0.0),
                }
            )
            det_rows.append(base)
        for truth in sample["ground_truth"].detections:
            gt_id = truth["oodkit_gt_id"]
            base = gt[gt["gt_id"].astype(str) == str(gt_id)].iloc[0].to_dict()
            status = str(truth.get(eval_key, "fn")).upper()
            base.update(
                {
                    "match_status": status,
                    "matched_detection_id": truth.get(f"{eval_key}_id", None),
                    "best_iou": float(truth.get(f"{eval_key}_iou", 0.0) or 0.0),
                }
            )
            gt_rows.append(base)
    dataset.delete()
    return _finalize_detections(det_rows), _finalize_ground_truth(gt_rows)


def _can_use_fiftyone(pred: pd.DataFrame, gt: pd.DataFrame) -> bool:
    try:
        import fiftyone  # noqa: F401
    except Exception:
        return False
    return _has_image_size_columns(pred, gt)


def _has_image_size_columns(pred: pd.DataFrame, gt: pd.DataFrame) -> bool:
    cols = {"image_width", "image_height"}
    frames = [df for df in (pred, gt) if not df.empty]
    return bool(frames) and all(cols.issubset(df.columns) for df in frames)


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        if len(df) == 0:
            for col in missing:
                df[col] = []
            return df
        raise ValueError(f"{name} is missing required columns: {missing}")
    return df


def _validate_box(box) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float64).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"bbox must have shape (4,), got {arr.shape}")
    if arr[2] <= arr[0] or arr[3] <= arr[1]:
        raise ValueError(f"bbox must have positive width and height, got {arr.tolist()}")
    return arr


def _iou(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def _to_fo_box(box, width: float, height: float) -> list[float]:
    box = np.asarray(box, dtype=np.float64)
    return [
        float(box[0] / width),
        float(box[1] / height),
        float((box[2] - box[0]) / width),
        float((box[3] - box[1]) / height),
    ]


def _class_name(class_id: int, class_names: Optional[Mapping[int, str]]) -> str:
    if class_names is None:
        return str(class_id)
    return str(class_names.get(int(class_id), int(class_id)))


def _first_non_null(values) -> Optional[object]:
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None


def _finalize_detections(rows) -> pd.DataFrame:
    cols = [
        "image_id",
        "class_id",
        "class_name",
        "bbox",
        "confidence",
        "detection_id",
        "match_status",
        "matched_gt_id",
        "iou_to_match",
    ]
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df:
            df[col] = []
    return df


def _finalize_ground_truth(rows) -> pd.DataFrame:
    cols = [
        "image_id",
        "class_id",
        "class_name",
        "bbox",
        "gt_id",
        "match_status",
        "matched_detection_id",
        "best_iou",
    ]
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df:
            df[col] = []
    return df


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["image_id", "class_id", "class_name", "bbox", "confidence", "detection_id"]
    )


def _empty_ground_truth() -> pd.DataFrame:
    return pd.DataFrame(columns=["image_id", "class_id", "class_name", "bbox", "gt_id"])


def _empty_image_class_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "image_id",
            "class_id",
            "class_name",
            "num_gt",
            "num_detections",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "has_failure",
            "missed_object",
            "false_positive",
            "low_precision",
            "low_recall",
            "mean_detection_confidence",
            "max_detection_confidence",
            "min_detection_confidence",
            "detector_name",
            "dataset_name",
        ]
    )


__all__ = [
    "DetectionEvaluationTables",
    "aggregate_chip_scores",
    "aggregate_image_class_metrics",
    "attach_ood_features",
    "detection_chips_from_table",
    "evaluate_detection_tables",
]
