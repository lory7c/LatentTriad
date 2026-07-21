"""Task-conditioned code/policy relations for pre-action skill monitoring."""

from __future__ import annotations

from typing import Mapping

import numpy as np


RELATIONAL_MODULES = ("attention", "mlp")
RELATIONAL_CANDIDATE_ORDER = (
    "unit_code_minus_full_policy",
    "unit_code_minus_declared_policy",
    "raw_code_minus_full_policy",
    "raw_code_minus_declared_policy",
    "cosine_code_full_policy",
    "cosine_code_declared_policy",
    "cosine_alignment_change",
)


def masked_probe_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pool [sample, probe, layer, token, hidden] into [sample, layer, probe, hidden]."""
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 5 or mask.shape != (
        values.shape[0],
        values.shape[1],
        values.shape[3],
    ):
        raise ValueError("Probe values or mask have the wrong shape")
    counts = mask.sum(axis=2)
    if np.any(counts == 0) or not np.isfinite(values).all():
        raise ValueError("Every probe requires finite active tokens")
    pooled = np.sum(values * mask[:, :, None, :, None], axis=3)
    pooled /= counts[:, :, None, None]
    return pooled.transpose(0, 2, 1, 3)


def masked_segment_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pool [sample, layer, segment, hidden] into [sample, layer, hidden]."""
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 4 or mask.shape != (values.shape[0], values.shape[2]):
        raise ValueError("Segment values or mask have the wrong shape")
    counts = mask.sum(axis=1)
    if np.any(counts == 0) or not np.isfinite(values).all():
        raise ValueError("Every sample requires finite active segments")
    return np.sum(values * mask[:, None, :, None], axis=2) / counts[:, None, None]


def centered_unit(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=-1, keepdims=True))
    return np.divide(
        centered,
        np.maximum(norm, epsilon),
        out=np.zeros_like(centered),
    )


def relational_candidate_sequences(
    segments: Mapping[str, np.ndarray],
    segment_mask: np.ndarray,
    full_suffix: Mapping[str, np.ndarray],
    declared_suffix: Mapping[str, np.ndarray],
    suffix_mask: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Build fixed code-to-policy relations without label-dependent feature choices."""
    missing = set(RELATIONAL_MODULES) - set(segments)
    missing |= set(RELATIONAL_MODULES) - set(full_suffix)
    missing |= set(RELATIONAL_MODULES) - set(declared_suffix)
    if missing:
        raise ValueError(f"Missing relational modules: {sorted(missing)}")
    candidates = {name: {} for name in RELATIONAL_CANDIDATE_ORDER}
    for module in RELATIONAL_MODULES:
        code = masked_segment_mean(segments[module], segment_mask)[:, :, None, :]
        full = masked_probe_mean(full_suffix[module], suffix_mask)
        declared = masked_probe_mean(declared_suffix[module], suffix_mask)
        if code.shape[:2] != full.shape[:2] or full.shape != declared.shape:
            raise ValueError("Code and policy representations are not layer-aligned")
        unit_code = centered_unit(code)
        unit_full = centered_unit(full)
        unit_declared = centered_unit(declared)
        candidates["unit_code_minus_full_policy"][module] = unit_code - unit_full
        candidates["unit_code_minus_declared_policy"][module] = (
            unit_code - unit_declared
        )
        candidates["raw_code_minus_full_policy"][module] = code - full
        candidates["raw_code_minus_declared_policy"][module] = code - declared
        cosine_full = np.sum(unit_code * unit_full, axis=-1, keepdims=True)
        cosine_declared = np.sum(
            unit_code * unit_declared, axis=-1, keepdims=True
        )
        candidates["cosine_code_full_policy"][module] = cosine_full
        candidates["cosine_code_declared_policy"][module] = cosine_declared
        candidates["cosine_alignment_change"][module] = (
            cosine_full - cosine_declared
        )
    return candidates

