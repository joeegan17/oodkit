"""Tests for object-detection evaluation tables."""

import numpy as np
import pandas as pd

from oodkit.detection import (
    attach_ood_features,
    detection_chips_from_table,
    evaluate_detection_tables,
)


def _pred(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "class_id",
            "bbox",
            "confidence",
            "detection_id",
            "class_name",
        ],
    )


def _gt(rows):
    return pd.DataFrame(
        rows,
        columns=["image_id", "class_id", "bbox", "gt_id", "class_name"],
    )


def test_gt_only_image_class_row():
    tables = evaluate_detection_tables(
        _pred([]),
        _gt([("img1", 1, [0, 0, 10, 10], "gt1", "car")]),
        backend="simple",
    )

    row = tables.image_class_metrics.iloc[0]
    assert row["num_gt"] == 1
    assert row["num_detections"] == 0
    assert row["tp"] == 0
    assert row["fp"] == 0
    assert row["fn"] == 1
    assert np.isnan(row["precision"])
    assert row["recall"] == 0.0
    assert bool(row["has_failure"])
    assert bool(row["missed_object"])
    assert not bool(row["false_positive"])
    assert pd.isna(row["low_precision"])
    assert bool(row["low_recall"])
    assert tables.ground_truth_enriched.iloc[0]["match_status"] == "FN"


def test_detection_only_image_class_row():
    tables = evaluate_detection_tables(
        _pred([("img1", 2, [0, 0, 10, 10], 0.8, "det1", "truck")]),
        _gt([]),
        backend="simple",
    )

    row = tables.image_class_metrics.iloc[0]
    assert row["num_gt"] == 0
    assert row["num_detections"] == 1
    assert row["tp"] == 0
    assert row["fp"] == 1
    assert row["fn"] == 0
    assert row["precision"] == 0.0
    assert np.isnan(row["recall"])
    assert bool(row["has_failure"])
    assert not bool(row["missed_object"])
    assert bool(row["false_positive"])
    assert bool(row["low_precision"])
    assert pd.isna(row["low_recall"])
    assert tables.detections_enriched.iloc[0]["match_status"] == "FP"


def test_mixed_tp_fp_fn_row():
    tables = evaluate_detection_tables(
        _pred(
            [
                ("img1", 1, [0, 0, 10, 10], 0.9, "det1", "car"),
                ("img1", 1, [50, 50, 60, 60], 0.7, "det2", "car"),
            ]
        ),
        _gt(
            [
                ("img1", 1, [0, 0, 10, 10], "gt1", "car"),
                ("img1", 1, [20, 20, 30, 30], "gt2", "car"),
            ]
        ),
        backend="simple",
        iou_threshold=0.5,
    )

    row = tables.image_class_metrics.iloc[0]
    assert row["tp"] == 1
    assert row["fp"] == 1
    assert row["fn"] == 1
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert row["f1"] == 0.5
    assert list(tables.detections_enriched["match_status"]) == ["TP", "FP"]
    assert sorted(tables.ground_truth_enriched["match_status"]) == ["FN", "TP"]


def test_multi_class_image_stays_separated():
    tables = evaluate_detection_tables(
        _pred(
            [
                ("img1", 1, [0, 0, 10, 10], 0.9, "det1", "car"),
                ("img1", 2, [40, 40, 50, 50], 0.8, "det2", "truck"),
            ]
        ),
        _gt(
            [
                ("img1", 1, [0, 0, 10, 10], "gt1", "car"),
                ("img1", 2, [20, 20, 30, 30], "gt2", "truck"),
            ]
        ),
        backend="simple",
    )

    metrics = tables.image_class_metrics.sort_values("class_id")
    assert metrics["class_id"].tolist() == [1, 2]
    assert metrics["tp"].tolist() == [1, 0]
    assert metrics["fn"].tolist() == [0, 1]
    assert metrics["fp"].tolist() == [0, 1]


def test_attach_ood_features_preserves_missing_chips():
    tables = evaluate_detection_tables(
        _pred([("img1", 1, [0, 0, 10, 10], 0.9, "det1", "car")]),
        _gt(
            [
                ("img1", 1, [0, 0, 10, 10], "gt1", "car"),
                ("img2", 2, [0, 0, 10, 10], "gt2", "truck"),
            ]
        ),
        backend="simple",
    )
    det = tables.detections_enriched.copy()
    det["chip_ood_score"] = [3.0]
    global_scores = pd.DataFrame(
        {"image_id": ["img1", "img2"], "global_ood_score": [1.0, 2.0]}
    )

    out = attach_ood_features(
        tables.image_class_metrics,
        global_scores=global_scores,
        detections_enriched=det,
        embedding_model_name="dinov2-small",
        ood_detector_name="Mahalanobis",
    ).sort_values(["image_id", "class_id"])

    det_row = out[(out["image_id"] == "img1") & (out["class_id"] == 1)].iloc[0]
    missing_row = out[(out["image_id"] == "img2") & (out["class_id"] == 2)].iloc[0]
    assert det_row["num_chips"] == 1
    assert det_row["chip_ood_mean"] == 3.0
    assert missing_row["num_chips"] == 0
    assert np.isnan(missing_row["chip_ood_mean"])
    assert np.isnan(missing_row["mean_detection_confidence"])
    assert missing_row["global_ood_score"] == 2.0
    assert missing_row["ood_detector_name"] == "Mahalanobis"


def test_detection_chips_from_table_preserves_detection_order(tmp_path):
    image_path = tmp_path / "img1.jpg"
    image_path.write_bytes(b"placeholder")
    det = _pred(
        [
            ("img1", 1, [0, 0, 10, 10], 0.9, "det1", "car"),
            ("img1", 1, [10, 10, 20, 20], 0.8, "det2", "car"),
        ]
    )

    anns, ordered = detection_chips_from_table(
        det,
        image_paths={"img1": str(image_path)},
        image_sizes={"img1": (100, 80)},
    )

    assert ordered == ["det1", "det2"]
    assert len(anns) == 1
    assert anns[0].image_id == "img1"
    assert anns[0].image_size == (100, 80)
    assert anns[0].boxes.shape == (2, 4)
