"""Architecture-neutral feature construction for TIOC pre-action traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


REQUIRED_VIEWS = ("task", "task_declared", "task_actual", "full")


def _validated_views(states: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = sorted(set(REQUIRED_VIEWS) - set(states))
    if missing:
        raise ValueError(f"Missing TIOC views: {missing}")
    arrays = {name: np.asarray(states[name], dtype=np.float64) for name in REQUIRED_VIEWS}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"TIOC view shapes differ: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 3:
        raise ValueError("Each TIOC state must have shape [layer, suffix_token, hidden]")
    if not np.isfinite(np.stack(list(arrays.values()))).all():
        raise ValueError("TIOC states contain non-finite values")
    return arrays


def factorial_interaction(states: Mapping[str, np.ndarray]) -> np.ndarray:
    """Second-order declaration-by-actual interaction conditioned on the task."""
    arrays = _validated_views(states)
    return (
        arrays["full"]
        - arrays["task_declared"]
        - arrays["task_actual"]
        + arrays["task"]
    )


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        np.sum(left * right, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )


def suffix_statistics(values: np.ndarray) -> np.ndarray:
    """Pool over fixed suffix positions without selecting a single fragile token."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Suffix statistics expect [layer, suffix_token]")
    positions = np.arange(values.shape[1], dtype=np.float64)
    centered = positions - positions.mean()
    denominator = float(np.sum(centered**2))
    slopes = (
        np.sum((values - values.mean(axis=1, keepdims=True)) * centered, axis=1)
        / denominator
        if denominator > 0
        else np.zeros(values.shape[0], dtype=np.float64)
    )
    return np.stack(
        (
            values.min(axis=1),
            values.mean(axis=1),
            values.max(axis=1),
            values.std(axis=1),
            slopes,
        ),
        axis=1,
    )


def hidden_interaction_features(states: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return per-layer scalar features derived from four matched prompt views."""
    arrays = _validated_views(states)
    interaction = factorial_interaction(arrays)
    actual_effect = arrays["full"] - arrays["task_declared"]
    declared_effect = arrays["full"] - arrays["task_actual"]
    skill_effect = arrays["full"] - arrays["task"]
    matrices = (
        suffix_statistics(np.linalg.norm(interaction, axis=-1)),
        suffix_statistics(np.linalg.norm(actual_effect, axis=-1)),
        suffix_statistics(np.linalg.norm(declared_effect, axis=-1)),
        suffix_statistics(np.linalg.norm(skill_effect, axis=-1)),
        suffix_statistics(cosine_rows(actual_effect, declared_effect)),
        suffix_statistics(cosine_rows(arrays["full"], arrays["task"])),
    )
    return np.concatenate(matrices, axis=1).astype(np.float32)


def normalized_entropy(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[axis]
    if count <= 1:
        return np.zeros(np.sum(values, axis=axis).shape, dtype=np.float64)
    total = values.sum(axis=axis, keepdims=True)
    probabilities = np.divide(
        values,
        total,
        out=np.full_like(values, 1.0 / count),
        where=total > 1e-12,
    )
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(probabilities * np.log(probabilities), axis=axis) / np.log(count)


def attention_region_features(
    masses: np.ndarray,
    region_names: Sequence[str] = ("task", "declaration", "actual"),
) -> np.ndarray:
    """Summarize suffix-to-region attention and task-control capture per layer."""
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim != 3:
        raise ValueError("Attention masses must have shape [layer, suffix_token, region]")
    if masses.shape[-1] != len(region_names):
        raise ValueError("Region name count does not match attention masses")
    if (masses < 0).any() or not np.isfinite(masses).all():
        raise ValueError("Attention masses must be finite and non-negative")
    summaries = [suffix_statistics(masses[:, :, index]) for index in range(len(region_names))]
    entropy = suffix_statistics(normalized_entropy(masses, axis=-1))
    by_name = {name: masses[:, :, index] for index, name in enumerate(region_names)}
    if {"task", "declaration", "actual"}.issubset(by_name):
        trusted = by_name["task"] + by_name["declaration"]
        capture = by_name["actual"] / np.maximum(trusted, 1e-8)
        summaries.append(suffix_statistics(capture))
    summaries.append(entropy)
    return np.concatenate(summaries, axis=1).astype(np.float32)


def headwise_attention_region_features(
    masses: np.ndarray,
    region_names: Sequence[str] = ("task", "declaration", "actual"),
) -> np.ndarray:
    """Pool headwise region masses without assuming equal head counts across models."""
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim != 4:
        raise ValueError(
            "Headwise attention masses must have shape [layer, head, suffix_token, region]"
        )
    if masses.shape[-1] != len(region_names):
        raise ValueError("Region name count does not match attention masses")
    pooled = (
        masses.mean(axis=1),
        masses.max(axis=1),
        masses.std(axis=1),
    )
    return np.concatenate(
        [attention_region_features(value, region_names) for value in pooled], axis=1
    ).astype(np.float32)
