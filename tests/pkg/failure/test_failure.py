"""Tests for failure-prediction baselines."""

import numpy as np
import pandas as pd

from oodkit.failure import (
    calibration_bins,
    evaluate_failure_baselines,
    grouped_train_test_split,
)


def _failure_table(n_images=10):
    rows = []
    for image_idx in range(n_images):
        for class_id in (1, 2):
            has_det = not (image_idx % 5 == 0 and class_id == 2)
            has_failure = (image_idx + class_id) % 3 == 0
            rows.append(
                {
                    "image_id": f"img{image_idx}",
                    "class_id": class_id,
                    "class_name": f"class{class_id}",
                    "has_failure": has_failure,
                    "global_ood_score": float(image_idx) / n_images,
                    "chip_ood_mean": np.nan if not has_det else float(class_id),
                    "chip_ood_max": np.nan if not has_det else float(class_id + 1),
                    "chip_ood_p90": np.nan if not has_det else float(class_id + 0.5),
                    "chip_ood_std": np.nan if not has_det else 0.1,
                    "num_chips": 1 if has_det else 0,
                    "mean_detection_confidence": 0.8 if has_det else np.nan,
                    "max_detection_confidence": 0.9 if has_det else np.nan,
                    "min_detection_confidence": 0.7 if has_det else np.nan,
                    "num_detections": 1 if has_det else 0,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_train_test_split_has_no_group_overlap():
    table = _failure_table()
    train_idx, test_idx = grouped_train_test_split(
        table,
        group_col="image_id",
        test_size=0.3,
        random_state=0,
    )

    train_groups = set(table.iloc[train_idx]["image_id"])
    test_groups = set(table.iloc[test_idx]["image_id"])
    assert train_groups.isdisjoint(test_groups)
    assert len(train_idx) + len(test_idx) == len(table)


def test_evaluate_failure_baselines_handles_missing_features():
    table = _failure_table()
    feature_sets = {
        "class prior": [],
        "global + chip + confidence + class": [
            "global_ood_score",
            "chip_ood_mean",
            "chip_ood_max",
            "chip_ood_p90",
            "chip_ood_std",
            "num_chips",
            "mean_detection_confidence",
            "max_detection_confidence",
            "min_detection_confidence",
            "num_detections",
            "class_id",
        ],
    }

    metrics, predictions = evaluate_failure_baselines(
        table,
        feature_sets=feature_sets,
        random_state=1,
    )

    assert metrics["model"].tolist() == list(feature_sets)
    assert set(predictions["model"]) == set(feature_sets)
    assert predictions["predicted_failure_probability"].between(0, 1).all()
    assert {"AUROC", "AUPR", "Brier"}.issubset(metrics.columns)


def test_calibration_bins_counts_all_examples():
    bins = calibration_bins(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        n_bins=2,
    )

    assert bins["count"].sum() == 4
    assert bins.loc[0, "observed_rate"] == 0.0
    assert bins.loc[1, "observed_rate"] == 1.0
