"""
ScoreBank — the primary interface between detectors and the evaluation module.

Once scores are added, every evaluation function (metrics, compare, plots)
accepts the bank directly. The user never has to worry about aligning arrays,
matching lengths, or remembering which scores came from which detector.
"""

from typing import Dict, List, Optional

import numpy as np

from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy


class ScoreBank:
    """Unified container for multi-detector OOD scores.

    ``ScoreBank`` is the single handoff point between detector outputs and
    analysis. It can be built incrementally (``add`` after creation) or
    populated from a dict at construction. Every evaluation function accepts
    a ``ScoreBank`` — no manual array alignment required.

    All arrays are converted to NumPy on ingest. Scores follow the library
    convention: **higher = more OOD**.

    Example::

        bank = ScoreBank(ood_labels=ood_gt, class_labels=pred_classes)
        bank.add("MSP", msp.score(features))
        bank.add("Energy", energy.score(features))
        bank.add("ViM", vim.score(features))
        bank.add_metric("accuracy", per_sample_acc)

    Args:
        scores: Optional initial dict mapping detector name to score array
            ``(n_samples,)``.
        ood_labels: Ground-truth ID/OOD labels, shape ``(n_samples,)``, 0=ID
            1=OOD. Required for supervised metrics; optional otherwise.
        class_labels: Per-sample class indices ``(n_samples,)`` (predicted or
            ground-truth — caller's choice). Required for ``by_class`` and
            ``evaluate_by_class``.
        sample_metrics: Optional dict mapping metric name to per-sample float
            array ``(n_samples,)``, e.g. per-sample accuracy or IoU.
    """

    def __init__(
        self,
        scores: Optional[Dict[str, ArrayLike]] = None,
        ood_labels: Optional[ArrayLike] = None,
        class_labels: Optional[ArrayLike] = None,
        sample_metrics: Optional[Dict[str, ArrayLike]] = None,
    ) -> None:
        self._scores: Dict[str, np.ndarray] = {}
        self._ood_labels: Optional[np.ndarray] = None
        self._class_labels: Optional[np.ndarray] = None
        self._sample_metrics: Dict[str, np.ndarray] = {}
        self._n_samples: Optional[int] = None

        if ood_labels is not None:
            self._ood_labels = to_numpy(ood_labels).astype(np.int64)
            self._set_n_samples(len(self._ood_labels), "ood_labels")

        if class_labels is not None:
            self._class_labels = to_numpy(class_labels).astype(np.int64)
            self._set_n_samples(len(self._class_labels), "class_labels")

        if scores is not None:
            for name, arr in scores.items():
                self.add(name, arr)

        if sample_metrics is not None:
            for name, arr in sample_metrics.items():
                self.add_metric(name, arr)

    # ------------------------------------------------------------------
    # Building the bank
    # ------------------------------------------------------------------

    def add(self, name: str, scores: ArrayLike) -> "ScoreBank":
        """Add per-sample OOD scores for one detector.

        Args:
            name: Detector identifier (used as the key in all downstream
                analysis functions).
            scores: Score array, shape ``(n_samples,)``. Higher = more OOD.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``len(scores)`` does not match the bank's
                established ``n_samples``.
        """
        arr = to_numpy(scores).astype(np.float64).ravel()
        self._set_n_samples(len(arr), f"scores[{name!r}]")
        self._scores[name] = arr
        return self

    def add_metric(self, name: str, values: ArrayLike) -> "ScoreBank":
        """Add per-sample user-provided metric values (e.g. accuracy, IoU).

        Args:
            name: Metric identifier.
            values: Float array, shape ``(n_samples,)``.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``len(values)`` does not match the bank's
                established ``n_samples``.
        """
        arr = to_numpy(values).astype(np.float64).ravel()
        self._set_n_samples(len(arr), f"sample_metrics[{name!r}]")
        self._sample_metrics[name] = arr
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def detectors(self) -> List[str]:
        """Names of all detectors currently in the bank."""
        return list(self._scores.keys())

    @property
    def n_samples(self) -> int:
        """Number of samples.

        Raises:
            ValueError: If the bank is empty (no arrays added yet).
        """
        if self._n_samples is None:
            raise ValueError("ScoreBank is empty — add scores before querying n_samples.")
        return self._n_samples

    @property
    def has_ood_labels(self) -> bool:
        """``True`` when OOD ground-truth labels are present."""
        return self._ood_labels is not None

    @property
    def has_class_labels(self) -> bool:
        """``True`` when per-sample class labels are present."""
        return self._class_labels is not None

    @property
    def ood_labels(self) -> Optional[np.ndarray]:
        """OOD labels array, shape ``(n_samples,)``, or ``None``."""
        return self._ood_labels

    @property
    def class_labels(self) -> Optional[np.ndarray]:
        """Class label array, shape ``(n_samples,)``, or ``None``."""
        return self._class_labels

    @property
    def classes(self) -> Optional[np.ndarray]:
        """Sorted unique class values, or ``None`` if no class labels."""
        if self._class_labels is None:
            return None
        return np.unique(self._class_labels)

    @property
    def metric_names(self) -> List[str]:
        """Names of all user-provided sample metrics."""
        return list(self._sample_metrics.keys())

    # ------------------------------------------------------------------
    # Score access
    # ------------------------------------------------------------------

    def scores_for(self, detector: str) -> np.ndarray:
        """Return the raw score array for a single detector.

        Args:
            detector: Detector name as passed to ``add()``.

        Returns:
            Score array, shape ``(n_samples,)``.

        Raises:
            KeyError: If the detector is not in the bank.
        """
        if detector not in self._scores:
            raise KeyError(
                f"Detector {detector!r} not in ScoreBank. "
                f"Available: {self.detectors}"
            )
        return self._scores[detector]

    def metric_for(self, name: str) -> np.ndarray:
        """Return the per-sample array for a user metric.

        Args:
            name: Metric name as passed to ``add_metric()``.

        Returns:
            Float array, shape ``(n_samples,)``.

        Raises:
            KeyError: If the metric is not in the bank.
        """
        if name not in self._sample_metrics:
            raise KeyError(
                f"Metric {name!r} not in ScoreBank. "
                f"Available: {self.metric_names}"
            )
        return self._sample_metrics[name]

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    def by_class(self, class_label: int) -> "ScoreBank":
        """Return a new ``ScoreBank`` restricted to one class.

        Args:
            class_label: The class value to select (must be present in
                ``class_labels``).

        Returns:
            A new ``ScoreBank`` containing only samples where
            ``class_labels == class_label``.

        Raises:
            ValueError: If no class labels are in the bank.
        """
        if self._class_labels is None:
            raise ValueError(
                "Cannot slice by class: ScoreBank has no class_labels. "
                "Pass class_labels at construction or add them before slicing."
            )
        mask = self._class_labels == class_label
        return self.subset(np.where(mask)[0])

    def subset(self, indices: ArrayLike) -> "ScoreBank":
        """Return a new ``ScoreBank`` restricted to the given sample indices.

        Args:
            indices: Integer index array; may be any array-like.

        Returns:
            A new ``ScoreBank`` with all arrays sliced to ``indices``.
        """
        idx = to_numpy(indices).astype(np.intp)

        sliced_scores = {name: arr[idx] for name, arr in self._scores.items()}
        sliced_ood = self._ood_labels[idx] if self._ood_labels is not None else None
        sliced_cls = self._class_labels[idx] if self._class_labels is not None else None
        sliced_metrics = {name: arr[idx] for name, arr in self._sample_metrics.items()}

        return ScoreBank(
            scores=sliced_scores,
            ood_labels=sliced_ood,
            class_labels=sliced_cls,
            sample_metrics=sliced_metrics if sliced_metrics else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_n_samples(self, length: int, source: str) -> None:
        if self._n_samples is None:
            self._n_samples = length
        elif self._n_samples != length:
            raise ValueError(
                f"Length mismatch: {source} has {length} samples but the bank "
                f"already has {self._n_samples}. All arrays must share the same "
                "number of samples."
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self._n_samples if self._n_samples is not None else 0
        parts = [f"ScoreBank(n_samples={n}"]
        if self._scores:
            parts.append(f"detectors={self.detectors}")
        if self.has_ood_labels:
            parts.append("ood_labels=True")
        if self.has_class_labels:
            parts.append(f"classes={list(self.classes)}")
        if self._sample_metrics:
            parts.append(f"metrics={self.metric_names}")
        return ", ".join(parts) + ")"
