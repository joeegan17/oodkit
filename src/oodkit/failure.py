"""Failure-prediction baselines for object-detection OOD workflows."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONFIDENCE_FEATURES = [
    "mean_detection_confidence",
    "max_detection_confidence",
    "min_detection_confidence",
    "num_detections",
]
GLOBAL_OOD_FEATURES = ["global_ood_score"]
CHIP_OOD_FEATURES = [
    "chip_ood_mean",
    "chip_ood_max",
    "chip_ood_p90",
    "chip_ood_std",
    "num_chips",
]

DEFAULT_FEATURE_SETS: Mapping[str, Sequence[str]] = {
    "class prior": [],
    "confidence only": CONFIDENCE_FEATURES,
    "global OOD only": GLOBAL_OOD_FEATURES,
    "chip OOD only": CHIP_OOD_FEATURES,
    "global + chip": GLOBAL_OOD_FEATURES + CHIP_OOD_FEATURES,
    "global + chip + confidence": GLOBAL_OOD_FEATURES
    + CHIP_OOD_FEATURES
    + CONFIDENCE_FEATURES,
    "global + chip + confidence + class": GLOBAL_OOD_FEATURES
    + CHIP_OOD_FEATURES
    + CONFIDENCE_FEATURES
    + ["class_id"],
}


def grouped_train_test_split(
    table: pd.DataFrame,
    *,
    group_col: str = "image_id",
    test_size: float = 0.3,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positional train/test indices with no group overlap."""
    if group_col not in table:
        raise ValueError(f"table is missing group_col={group_col!r}")
    if len(table) == 0:
        raise ValueError("table must contain at least one row")

    groups = table[group_col].to_numpy()
    if len(pd.unique(groups)) < 2:
        raise ValueError("grouped split requires at least two unique groups")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(table, groups=groups))
    return train_idx, test_idx


def evaluate_failure_baselines(
    table: pd.DataFrame,
    *,
    target_col: str = "has_failure",
    feature_sets: Optional[Mapping[str, Sequence[str]]] = None,
    group_col: str = "image_id",
    categorical_columns: Sequence[str] = ("class_id",),
    test_size: float = 0.3,
    random_state: int = 0,
    class_weight: Optional[str] = "balanced",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train grouped logistic baselines for detection-failure prediction.

    Numeric NaNs are imputed using train-split medians through sklearn's
    pipeline, missingness indicators are added by the imputer, numeric features
    are standardized, and categorical features such as ``class_id`` are one-hot
    encoded. The target defaults to ``has_failure``.

    Returns:
        ``(metrics, predictions)`` where metrics has one row per feature set and
        predictions has one row per held-out table row per feature set.
    """
    if target_col not in table:
        raise ValueError(f"table is missing target_col={target_col!r}")
    feature_sets = DEFAULT_FEATURE_SETS if feature_sets is None else feature_sets

    work = table.copy()
    target = work[target_col]
    valid = target.notna()
    work = work.loc[valid].copy()
    if work.empty:
        raise ValueError("no rows have a non-missing target")
    y = work[target_col].astype(bool).astype(int).to_numpy()

    train_idx, test_idx = grouped_train_test_split(
        work,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
    )
    y_train = y[train_idx]
    y_test = y[test_idx]

    rows = []
    pred_frames = []
    for name, features in feature_sets.items():
        features = list(features)
        missing = [col for col in features if col not in work.columns]
        if missing:
            raise ValueError(f"feature set {name!r} references missing columns: {missing}")

        if not features or np.unique(y_train).size < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(train_idx), 1)), y_train)
            proba = model.predict_proba(np.zeros((len(test_idx), 1)))[:, 1]
        else:
            categorical = [c for c in features if c in set(categorical_columns)]
            numeric = [c for c in features if c not in categorical]
            model = _make_logistic_pipeline(
                numeric,
                categorical,
                class_weight=class_weight,
                random_state=random_state,
            )
            model.fit(work.iloc[train_idx][features], y_train)
            proba = model.predict_proba(work.iloc[test_idx][features])[:, 1]

        metric = _classification_metrics(y_test, proba)
        metric.update(
            {
                "model": name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "positive_rate_test": float(np.mean(y_test)),
            }
        )
        rows.append(metric)

        cols = [c for c in ["image_id", "class_id", "class_name"] if c in work]
        pred = work.iloc[test_idx][cols].copy()
        pred["target"] = y_test
        pred["predicted_failure_probability"] = proba
        pred["model"] = name
        pred_frames.append(pred)

    metrics = pd.DataFrame(rows)
    ordered = [
        "model",
        "AUROC",
        "AUPR",
        "Brier",
        "n_train",
        "n_test",
        "positive_rate_test",
    ]
    metrics = metrics[[c for c in ordered if c in metrics.columns]]
    predictions = (
        pd.concat(pred_frames, ignore_index=True)
        if pred_frames
        else pd.DataFrame(
            columns=["target", "predicted_failure_probability", "model"]
        )
    )
    return metrics, predictions


def calibration_bins(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin predicted probabilities for calibration plotting."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for idx in range(n_bins):
        mask = bins == idx
        rows.append(
            {
                "bin": idx,
                "bin_left": float(edges[idx]),
                "bin_right": float(edges[idx + 1]),
                "count": int(np.sum(mask)),
                "mean_predicted": float(np.mean(p[mask])) if np.any(mask) else np.nan,
                "observed_rate": float(np.mean(y[mask])) if np.any(mask) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _make_logistic_pipeline(
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    class_weight: Optional[str],
    random_state: int,
) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                                keep_empty_features=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric),
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical),
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if np.unique(y_true).size < 2:
        out["AUROC"] = np.nan
        out["AUPR"] = np.nan
    else:
        out["AUROC"] = float(roc_auc_score(y_true, y_prob))
        out["AUPR"] = float(average_precision_score(y_true, y_prob))
    out["Brier"] = float(brier_score_loss(y_true, y_prob))
    return out


__all__ = [
    "CHIP_OOD_FEATURES",
    "CONFIDENCE_FEATURES",
    "DEFAULT_FEATURE_SETS",
    "GLOBAL_OOD_FEATURES",
    "calibration_bins",
    "evaluate_failure_baselines",
    "grouped_train_test_split",
]
