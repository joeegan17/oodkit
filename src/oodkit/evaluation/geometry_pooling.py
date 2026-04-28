"""Geometry-aware image-level pooling for object-detection OOD scores.

The pooler treats each image as a small scene graph: chips/objects are nodes,
and selected object pairs are edges. It fits simple ID statistics, then scores
new images by combining object-level OOD scores with size, layout,
co-occurrence, and interaction surprises.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from oodkit.evaluation.pooling import pool_image_scores

_COMPONENTS = ("node", "size", "layout", "cooccurrence", "interaction")


class GeometryAwarePooler:
    """Non-learned scene-level OOD pooling from ID geometry statistics.

    Args:
        node_pool_method: Baseline object-score pooling method. One of
            ``"mean"``, ``"max"``, or ``"topk_mean"``.
        k: Top-k value used when ``node_pool_method="topk_mean"``.
        edge_k: For crowded images, keep up to this many nearest-neighbor edges
            per object before applying ``max_pairs``.
        max_pairs: Maximum number of object pairs scored per image.
        min_class_count: Minimum number of ID objects needed for per-class size
            statistics. Rarer classes use global size statistics.
        min_pair_count: Minimum number of ID pairs needed for per-class-pair
            layout statistics. Rarer pairs use global pair statistics.
        smoothing: Additive smoothing for class-pair co-occurrence surprise.
        weights: Optional component weights. Missing components default to 1.
        eps: Small positive constant for numeric stability.
    """

    def __init__(
        self,
        *,
        node_pool_method: str = "topk_mean",
        k: int = 3,
        edge_k: int = 8,
        max_pairs: int = 256,
        min_class_count: int = 2,
        min_pair_count: int = 2,
        smoothing: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        eps: float = 1e-6,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if edge_k < 1:
            raise ValueError(f"edge_k must be >= 1, got {edge_k}")
        if max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
        if min_class_count < 1:
            raise ValueError("min_class_count must be >= 1")
        if min_pair_count < 1:
            raise ValueError("min_pair_count must be >= 1")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.node_pool_method = node_pool_method
        self.k = int(k)
        self.edge_k = int(edge_k)
        self.max_pairs = int(max_pairs)
        self.min_class_count = int(min_class_count)
        self.min_pair_count = int(min_pair_count)
        self.smoothing = float(smoothing)
        self.eps = float(eps)

        self.weights = {name: 1.0 for name in _COMPONENTS}
        if weights is not None:
            unknown = set(weights) - set(_COMPONENTS)
            if unknown:
                raise ValueError(f"unknown component weights: {sorted(unknown)}")
            for name, value in weights.items():
                if value < 0:
                    raise ValueError(f"weight {name!r} must be non-negative")
                self.weights[name] = float(value)

        self._fitted = False

    def fit(
        self,
        chip_scores: np.ndarray,
        chip_to_image: np.ndarray,
        boxes: np.ndarray,
        class_labels: np.ndarray,
        image_sizes: np.ndarray,
        *,
        n_images: Optional[int] = None,
    ) -> "GeometryAwarePooler":
        """Fit ID-only size, layout, and co-occurrence statistics."""
        data = self._validate_inputs(
            chip_scores, chip_to_image, boxes, class_labels, image_sizes,
            n_images=n_images,
        )
        if data["scores"].size == 0:
            raise ValueError("GeometryAwarePooler.fit requires at least one chip")

        self._chip_score_mean = float(np.mean(data["scores"]))
        self._chip_score_std = self._safe_scalar_std(data["scores"])

        geom = self._geometry_features(data["boxes"], data["image_sizes"])
        self._fit_size_stats(data["classes"], geom["log_area"], geom["log_aspect"])
        self._fit_pair_stats(data, geom)

        self._fitted = True
        raw = self._raw_components(data, geom)
        self._component_mean: Dict[str, float] = {}
        self._component_std: Dict[str, float] = {}
        for name in _COMPONENTS:
            vals = raw[name]
            vals = vals[np.isfinite(vals)]
            self._component_mean[name] = float(np.mean(vals)) if vals.size else 0.0
            self._component_std[name] = self._safe_scalar_std(vals)
        return self

    def score(
        self,
        chip_scores: np.ndarray,
        chip_to_image: np.ndarray,
        boxes: np.ndarray,
        class_labels: np.ndarray,
        image_sizes: np.ndarray,
        *,
        n_images: Optional[int] = None,
        return_components: bool = False,
    ):
        """Score images, optionally returning calibrated component breakdowns."""
        if not self._fitted:
            raise RuntimeError("GeometryAwarePooler must be fit before score().")
        data = self._validate_inputs(
            chip_scores, chip_to_image, boxes, class_labels, image_sizes,
            n_images=n_images,
        )
        geom = self._geometry_features(data["boxes"], data["image_sizes"])
        raw = self._raw_components(data, geom)
        components = self._calibrate_components(raw)
        final = np.full(data["n_images"], np.nan, dtype=np.float64)
        has_chip = np.isfinite(raw["node"])
        final[has_chip] = 0.0
        for name in _COMPONENTS:
            final[has_chip] += self.weights[name] * components[name][has_chip]
        components["final"] = final
        if return_components:
            return final, components
        return final

    def _validate_inputs(
        self,
        chip_scores,
        chip_to_image,
        boxes,
        class_labels,
        image_sizes,
        *,
        n_images: Optional[int],
    ) -> Dict[str, np.ndarray]:
        scores = np.asarray(chip_scores, dtype=np.float64).ravel()
        c2i = np.asarray(chip_to_image, dtype=np.int64).ravel()
        b = np.asarray(boxes, dtype=np.float64)
        cls = np.asarray(class_labels, dtype=np.int64).ravel()
        sizes = np.asarray(image_sizes, dtype=np.float64)

        n = scores.shape[0]
        if c2i.shape[0] != n or cls.shape[0] != n:
            raise ValueError("chip_scores, chip_to_image, and class_labels length mismatch")
        if b.ndim != 2 or b.shape[1] != 4 or b.shape[0] != n:
            raise ValueError(f"boxes must have shape (N, 4), got {b.shape}")
        if sizes.ndim != 2 or sizes.shape[1] != 2 or sizes.shape[0] != n:
            raise ValueError(f"image_sizes must have shape (N, 2), got {sizes.shape}")
        if np.any(c2i < 0):
            raise ValueError("chip_to_image must be non-negative")
        if np.any(sizes <= 0):
            raise ValueError("image_sizes values must be positive")
        widths = b[:, 2] - b[:, 0]
        heights = b[:, 3] - b[:, 1]
        if np.any(widths <= 0) or np.any(heights <= 0):
            raise ValueError("boxes must have positive width and height")

        if n_images is None:
            n_out = int(c2i.max()) + 1 if c2i.size else 0
        else:
            if n_images < 0:
                raise ValueError("n_images must be non-negative")
            if c2i.size and int(c2i.max()) >= n_images:
                raise ValueError("n_images is smaller than max(chip_to_image) + 1")
            n_out = int(n_images)

        return {
            "scores": scores,
            "chip_to_image": c2i,
            "boxes": b,
            "classes": cls,
            "image_sizes": sizes,
            "n_images": n_out,
        }

    def _geometry_features(self, boxes: np.ndarray, image_sizes: np.ndarray) -> Dict[str, np.ndarray]:
        w = np.maximum(boxes[:, 2] - boxes[:, 0], self.eps)
        h = np.maximum(boxes[:, 3] - boxes[:, 1], self.eps)
        iw = image_sizes[:, 0]
        ih = image_sizes[:, 1]
        area_frac = np.maximum((w * h) / np.maximum(iw * ih, self.eps), self.eps)
        centers = np.column_stack(
            [
                ((boxes[:, 0] + boxes[:, 2]) * 0.5) / iw,
                ((boxes[:, 1] + boxes[:, 3]) * 0.5) / ih,
            ]
        )
        return {
            "centers": centers,
            "log_area": np.log(area_frac),
            "log_aspect": np.log(np.maximum(w / h, self.eps)),
        }

    def _fit_size_stats(
        self, classes: np.ndarray, log_area: np.ndarray, log_aspect: np.ndarray
    ) -> None:
        feats = np.column_stack([log_area, log_aspect])
        self._size_global_mean = np.mean(feats, axis=0)
        self._size_global_std = self._safe_vector_std(feats)
        self._size_stats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for cls in np.unique(classes):
            vals = feats[classes == cls]
            if vals.shape[0] >= self.min_class_count:
                self._size_stats[int(cls)] = (
                    np.mean(vals, axis=0),
                    self._safe_vector_std(vals),
                )

    def _fit_pair_stats(self, data: Dict[str, np.ndarray], geom: Dict[str, np.ndarray]) -> None:
        by_pair: Dict[Tuple[int, int], List[np.ndarray]] = {}
        pair_counts: Dict[Tuple[int, int], int] = {}
        all_feats: List[np.ndarray] = []
        for indices in self._image_index_groups(data["chip_to_image"], data["n_images"]):
            for i, j in self._select_pairs(indices, geom["centers"]):
                key = self._pair_key(data["classes"][i], data["classes"][j])
                feat = self._pair_feature(i, j, data, geom)
                by_pair.setdefault(key, []).append(feat)
                pair_counts[key] = pair_counts.get(key, 0) + 1
                all_feats.append(feat)

        if all_feats:
            all_arr = np.vstack(all_feats)
            self._pair_global_mean = np.mean(all_arr, axis=0)
            self._pair_global_std = self._safe_vector_std(all_arr)
        else:
            self._pair_global_mean = np.zeros(5, dtype=np.float64)
            self._pair_global_std = np.ones(5, dtype=np.float64)

        self._pair_stats: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for key, vals in by_pair.items():
            arr = np.vstack(vals)
            if arr.shape[0] >= self.min_pair_count:
                self._pair_stats[key] = (np.mean(arr, axis=0), self._safe_vector_std(arr))
        self._pair_counts = pair_counts
        self._pair_total = int(sum(pair_counts.values()))
        self._pair_vocab = int(len(pair_counts))

    def _raw_components(self, data: Dict[str, np.ndarray], geom: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n_images = data["n_images"]
        raw = {
            "node": pool_image_scores(
                data["scores"],
                data["chip_to_image"],
                method=self.node_pool_method,
                k=self.k,
                n_images=n_images,
            ),
            "size": np.full(n_images, np.nan, dtype=np.float64),
            "layout": np.full(n_images, np.nan, dtype=np.float64),
            "cooccurrence": np.full(n_images, np.nan, dtype=np.float64),
            "interaction": np.full(n_images, np.nan, dtype=np.float64),
        }
        chip_z = np.maximum(0.0, (data["scores"] - self._chip_score_mean) / self._chip_score_std)

        for indices in self._image_index_groups(data["chip_to_image"], n_images):
            if indices.size == 0:
                continue
            image_id = int(data["chip_to_image"][indices[0]])
            size_vals = [
                self._size_surprise(
                    int(data["classes"][idx]), geom["log_area"][idx], geom["log_aspect"][idx]
                )
                for idx in indices
            ]
            raw["size"][image_id] = float(np.mean(size_vals))

            pairs = self._select_pairs(indices, geom["centers"])
            if not pairs:
                raw["layout"][image_id] = 0.0
                raw["cooccurrence"][image_id] = 0.0
                raw["interaction"][image_id] = 0.0
                continue

            layout_vals: List[float] = []
            co_vals: List[float] = []
            interaction_vals: List[float] = []
            for i, j in pairs:
                key = self._pair_key(data["classes"][i], data["classes"][j])
                layout = self._layout_surprise(key, self._pair_feature(i, j, data, geom))
                co = self._cooccurrence_surprise(key)
                layout_vals.append(layout)
                co_vals.append(co)
                interaction_vals.append(max(chip_z[i], chip_z[j]) * (layout + co))

            raw["layout"][image_id] = float(np.mean(layout_vals))
            raw["cooccurrence"][image_id] = float(np.mean(co_vals))
            raw["interaction"][image_id] = float(np.mean(interaction_vals))

        return raw

    def _calibrate_components(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name in _COMPONENTS:
            vals = raw[name]
            comp = np.full(vals.shape, np.nan, dtype=np.float64)
            mask = np.isfinite(vals)
            comp[mask] = np.maximum(
                0.0,
                (vals[mask] - self._component_mean[name]) / self._component_std[name],
            )
            out[name] = comp
        return out

    def _size_surprise(self, cls: int, log_area: float, log_aspect: float) -> float:
        mean, std = self._size_stats.get(
            int(cls), (self._size_global_mean, self._size_global_std)
        )
        z = np.abs((np.array([log_area, log_aspect]) - mean) / std)
        return float(np.mean(z))

    def _layout_surprise(self, key: Tuple[int, int], feat: np.ndarray) -> float:
        mean, std = self._pair_stats.get(
            key, (self._pair_global_mean, self._pair_global_std)
        )
        return float(np.mean(np.abs((feat - mean) / std)))

    def _cooccurrence_surprise(self, key: Tuple[int, int]) -> float:
        if self._pair_total == 0:
            return 0.0
        denom = self._pair_total + self.smoothing * (self._pair_vocab + 1)
        count = self._pair_counts.get(key, 0)
        prob = (count + self.smoothing) / denom
        return float(-np.log(prob))

    def _select_pairs(self, indices: np.ndarray, centers: np.ndarray) -> List[Tuple[int, int]]:
        idx = np.asarray(indices, dtype=np.int64).ravel()
        n = idx.shape[0]
        if n < 2:
            return []
        total_pairs = n * (n - 1) // 2
        if total_pairs <= self.max_pairs:
            return [(int(i), int(j)) for i, j in combinations(idx.tolist(), 2)]

        local_centers = centers[idx]
        d = np.linalg.norm(
            local_centers[:, None, :] - local_centers[None, :, :], axis=2
        )
        np.fill_diagonal(d, np.inf)
        pairs = set()
        k = min(self.edge_k, n - 1)
        for row in range(n):
            nbrs = np.argpartition(d[row], kth=k - 1)[:k]
            for col in nbrs:
                a, b = sorted((row, int(col)))
                pairs.add((a, b))
        ordered = sorted(pairs, key=lambda p: d[p[0], p[1]])[: self.max_pairs]
        return [(int(idx[a]), int(idx[b])) for a, b in ordered]

    def _pair_feature(
        self, i: int, j: int, data: Dict[str, np.ndarray], geom: Dict[str, np.ndarray]
    ) -> np.ndarray:
        ci, cj = int(data["classes"][i]), int(data["classes"][j])
        if (ci, cj) > (cj, ci):
            i, j = j, i
        dx = geom["centers"][j, 0] - geom["centers"][i, 0]
        dy = geom["centers"][j, 1] - geom["centers"][i, 1]
        dist = float(np.hypot(dx, dy))
        scale = geom["log_area"][i] - geom["log_area"][j]
        iou = self._iou(data["boxes"][i], data["boxes"][j])
        return np.array([dist, dx, dy, scale, iou], dtype=np.float64)

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        ia, ib = int(a), int(b)
        return (ia, ib) if ia <= ib else (ib, ia)

    @staticmethod
    def _image_index_groups(chip_to_image: np.ndarray, n_images: int) -> Iterable[np.ndarray]:
        for image_id in range(n_images):
            yield np.nonzero(chip_to_image == image_id)[0]

    def _safe_vector_std(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] < 2:
            return np.ones(arr.shape[1], dtype=np.float64)
        std = np.std(arr, axis=0)
        return np.where(std > self.eps, std, 1.0)

    def _safe_scalar_std(self, arr: np.ndarray) -> float:
        if arr.size < 2:
            return 1.0
        std = float(np.std(arr))
        return std if std > self.eps else 1.0

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        denom = area_a + area_b - inter
        return 0.0 if denom <= 0 else inter / denom


__all__ = ["GeometryAwarePooler"]
