"""Object-detection utilities for failure prediction workflows."""

from oodkit.detection.inference import run_torchvision_detector
from oodkit.detection.tables import (
    DetectionEvaluationTables,
    aggregate_chip_scores,
    attach_ood_features,
    detection_chips_from_table,
    evaluate_detection_tables,
)

__all__ = [
    "DetectionEvaluationTables",
    "aggregate_chip_scores",
    "attach_ood_features",
    "detection_chips_from_table",
    "evaluate_detection_tables",
    "run_torchvision_detector",
]
