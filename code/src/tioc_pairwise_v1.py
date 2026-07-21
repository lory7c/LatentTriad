"""Low-capacity matched-pair directions for TIOC hidden-state monitoring."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _paired_indices(metadata: Sequence[dict]) -> list[tuple[int, int, str]]:
    by_pair: dict[str, dict[str, int | str]] = defaultdict(dict)
    for index, row in enumerate(metadata):
        pair_id = str(row["pair_id"])
        role = str(row["role"])
        if role not in {"benign", "malicious"}:
            raise ValueError(f"Unknown pair role: {role}")
        if role in by_pair[pair_id]:
            raise ValueError(f"Duplicate {role} record for pair: {pair_id}")
        by_pair[pair_id][role] = index
        group = str(row["base_group_id"])
        previous_group = by_pair[pair_id].get("base_group_id")
        if previous_group is not None and previous_group != group:
            raise ValueError(f"Pair spans multiple base groups: {pair_id}")
        by_pair[pair_id]["base_group_id"] = group

    pairs = []
    for pair_id, values in sorted(by_pair.items()):
        if not {"benign", "malicious"}.issubset(values):
            raise ValueError(f"Incomplete benign/malicious pair: {pair_id}")
        pairs.append(
            (
                int(values["benign"]),
                int(values["malicious"]),
                str(values["base_group_id"]),
            )
        )
    if not pairs:
        raise ValueError("At least one matched pair is required")
    return pairs


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[:, None], axis=0) / np.sum(weights)


@dataclass
class PairedContrastiveDirection:
    """Estimate a CAA-style direction after removing pair-midpoint nuisance scale.

    The detector uses role labels only through within-pair differences. Pair midpoints
    estimate task/template variation, and diagonal shrinkage keeps the estimator stable
    when hidden width is much larger than the number of training pairs.
    """

    shrinkage: float = 0.9
    normalize_pair_deltas: bool = True
    epsilon: float = 1e-6

    def fit(
        self, features: np.ndarray, metadata: Sequence[dict]
    ) -> "PairedContrastiveDirection":
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != len(metadata):
            raise ValueError("Features must have shape [sample, hidden]")
        if not np.isfinite(features).all():
            raise ValueError("Features contain non-finite values")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")

        pairs = _paired_indices(metadata)
        group_counts = Counter(group for _, _, group in pairs)
        pair_weights = np.asarray(
            [1.0 / group_counts[group] for _, _, group in pairs], dtype=np.float64
        )
        pair_weights /= pair_weights.sum()

        benign = np.stack([features[benign_index] for benign_index, _, _ in pairs])
        malicious = np.stack(
            [features[malicious_index] for _, malicious_index, _ in pairs]
        )
        midpoints = 0.5 * (benign + malicious)
        location = _weighted_mean(midpoints, pair_weights)
        midpoint_variance = _weighted_mean((midpoints - location) ** 2, pair_weights)
        positive = midpoint_variance[midpoint_variance > 0]
        global_variance = float(np.median(positive)) if len(positive) else 1.0
        shrunk_variance = (
            (1.0 - self.shrinkage) * midpoint_variance
            + self.shrinkage * global_variance
        )
        variance_floor = max(global_variance * self.epsilon, self.epsilon)
        scale = np.sqrt(np.maximum(shrunk_variance, variance_floor))

        deltas = (malicious - benign) / scale
        if self.normalize_pair_deltas:
            delta_norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            deltas = np.divide(
                deltas,
                delta_norms,
                out=np.zeros_like(deltas),
                where=delta_norms > self.epsilon,
            )
        direction = _weighted_mean(deltas, pair_weights)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= self.epsilon:
            raise ValueError("Matched pairs do not define a non-zero direction")
        direction /= direction_norm

        raw_scores = ((features - location) / scale) @ direction
        sample_weights = np.zeros(len(features), dtype=np.float64)
        for weight, (benign_index, malicious_index, _) in zip(pair_weights, pairs):
            sample_weights[benign_index] = 0.5 * weight
            sample_weights[malicious_index] = 0.5 * weight
        score_location = float(np.sum(sample_weights * raw_scores))
        score_variance = float(
            np.sum(sample_weights * (raw_scores - score_location) ** 2)
        )

        self.location_ = location
        self.scale_ = scale
        self.direction_ = direction
        self.score_location_ = score_location
        self.score_scale_ = max(np.sqrt(score_variance), self.epsilon)
        self.pair_count_ = len(pairs)
        self.base_group_count_ = len(group_counts)
        self.global_midpoint_variance_ = global_variance
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "direction_"):
            raise ValueError("Detector must be fit before scoring")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.direction_):
            raise ValueError("Features have the wrong shape")
        raw_scores = ((features - self.location_) / self.scale_) @ self.direction_
        return (raw_scores - self.score_location_) / self.score_scale_

