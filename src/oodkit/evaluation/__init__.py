"""
OOD evaluation module.

Provides tools for comparing multiple detectors, computing standard OOD
metrics, and visualizing detector behavior.

The primary interface is ``ScoreBank``:  add detector scores once, then pass
the bank to any evaluation function — no manual array alignment required.

Typical workflow::

    from oodkit.evaluation import ScoreBank, evaluate, plots

    bank = ScoreBank(ood_labels=ood_gt, class_labels=pred_classes)
    bank.add("MSP", msp.score(features))
    bank.add("Energy", energy.score(features))
    bank.add("ViM", vim.score(features))

    # Supervised metrics
    table = evaluate(bank)
    print(table)

    # Visualization
    plots.roc_curves(bank)
    plots.score_distributions(bank)
    plots.correlation_heatmap(bank)
"""

from oodkit.evaluation.score_bank import ScoreBank
from oodkit.evaluation.metrics import (
    auroc,
    aupr,
    detection_error,
    evaluate,
    evaluate_by_class,
    fpr_at_tpr,
    MetricsTable,
    pr_curve,
    roc_curve,
)
from oodkit.evaluation.compare import (
    disagreements,
    normalize_scores,
    rank_samples,
    score_correlation,
)
from oodkit.evaluation.performance import PerformanceCurve, score_vs_metric
from oodkit.evaluation.combine import (
    concatenate_embedding_results,
    ood_labels_from_blocks,
    ood_labels_from_counts,
)
from oodkit.evaluation.pooling import pool_image_scores
from oodkit.evaluation import plots

__all__ = [
    # Container
    "ScoreBank",
    # Metrics
    "auroc",
    "aupr",
    "detection_error",
    "evaluate",
    "evaluate_by_class",
    "fpr_at_tpr",
    "MetricsTable",
    "pr_curve",
    "roc_curve",
    # Compare
    "disagreements",
    "normalize_scores",
    "rank_samples",
    "score_correlation",
    # Performance
    "PerformanceCurve",
    "score_vs_metric",
    # Combine ID/OOD results
    "concatenate_embedding_results",
    "ood_labels_from_blocks",
    "ood_labels_from_counts",
    # Object-detection pooling
    "pool_image_scores",
    # Plots (namespace import)
    "plots",
]
