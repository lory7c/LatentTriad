"""Fixed-contract pre-action policy forks and low-capacity sequence probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.tioc_pairwise_v1 import PairedContrastiveDirection


MODULE_ORDER = ("residual", "attention", "mlp")


@dataclass(frozen=True)
class ProbeSuffix:
    name: str
    text: str


PROBE_SUFFIXES = (
    ProbeSuffix(
        "operation",
        "The single next operation I am about to perform is:",
    ),
    ProbeSuffix(
        "resource",
        "The tool and target resource for that operation are:",
    ),
    ProbeSuffix(
        "authority",
        "The user-task authorization relevant to that operation is:",
    ),
)


def selected_probe_suffixes(names: Sequence[str] | None = None) -> tuple[ProbeSuffix, ...]:
    if names is None:
        return PROBE_SUFFIXES
    requested = list(names)
    if len(requested) != len(set(requested)):
        raise ValueError("Probe suffix names must be unique")
    by_name = {probe.name: probe for probe in PROBE_SUFFIXES}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown policy-fork probes: {unknown}")
    if not requested:
        raise ValueError("At least one policy-fork probe is required")
    return tuple(by_name[name] for name in requested)


def adaptive_segment_mean(values: np.ndarray, segment_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool an ordered token sequence into a fixed number of contiguous segments."""
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite [token, hidden] matrix")
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    if len(values) == 0:
        raise ValueError("Cannot segment an empty token sequence")
    active = min(segment_count, len(values))
    boundaries = np.linspace(0, len(values), active + 1, dtype=np.int64)
    pooled = np.zeros((segment_count, values.shape[1]), dtype=values.dtype)
    mask = np.zeros(segment_count, dtype=bool)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        pooled[index] = values[start:end].mean(axis=0)
        mask[index] = True
    return pooled, mask


def masked_sequence_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("Expected values [sample, token, hidden] and mask [sample, token]")
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every sample must contain at least one active token")
    return np.sum(values * mask[:, :, None], axis=1) / counts[:, None]


def flatten_probe_tokens(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [sample, probe, layer, token, hidden] to [sample, layer, sequence, hidden]."""
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 5:
        raise ValueError("Fork values must have shape [sample, probe, layer, token, hidden]")
    if mask.shape != (values.shape[0], values.shape[1], values.shape[3]):
        raise ValueError("Fork mask shape does not match probe/token axes")
    flattened = values.transpose(0, 2, 1, 3, 4).reshape(
        values.shape[0], values.shape[2], values.shape[1] * values.shape[3], values.shape[4]
    )
    return flattened, mask.reshape(mask.shape[0], -1)


def _extract_tensor(output):
    value = output
    while isinstance(value, (tuple, list)):
        if not value:
            raise TypeError("Hook output tuple is empty")
        value = value[0]
    if not hasattr(value, "ndim") or value.ndim != 3:
        raise TypeError("Hook output must resolve to [batch, token, hidden]")
    return value


class ModuleSequenceCollector:
    """Collect selected decoder/module outputs without materializing full attention maps."""

    def __init__(
        self,
        layers,
        selected_layers: Sequence[int],
        *,
        module_names: Sequence[str] = MODULE_ORDER,
        token_positions: Sequence[int] | None = None,
        segment_count: int | None = None,
        defer_cpu_copy: bool = False,
    ) -> None:
        self.layers = layers
        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.module_names = tuple(module_names)
        if not self.module_names or len(set(self.module_names)) != len(self.module_names):
            raise ValueError("module_names must be a non-empty unique sequence")
        unknown = sorted(set(self.module_names) - set(MODULE_ORDER))
        if unknown:
            raise ValueError(f"Unknown module names: {unknown}")
        self.token_positions = None if token_positions is None else tuple(int(x) for x in token_positions)
        self.segment_count = segment_count
        self.defer_cpu_copy = bool(defer_cpu_copy)
        self.handles = []
        self.values: dict[tuple[str, int], np.ndarray] = {}
        self.mask: np.ndarray | None = None

    def _hook(self, module_name: str, layer_index: int):
        def capture(_module, _inputs, output):
            tensor = _extract_tensor(output).detach()[0]
            if self.token_positions is not None:
                if not self.token_positions:
                    raise ValueError("Token position selection is empty")
                tensor = tensor[list(self.token_positions)]
            if self.segment_count is not None:
                active = min(self.segment_count, len(tensor))
                boundaries = np.linspace(0, len(tensor), active + 1, dtype=np.int64)
                pooled = tensor.new_zeros((self.segment_count, tensor.shape[-1]))
                for index, (start, end) in enumerate(
                    zip(boundaries[:-1], boundaries[1:])
                ):
                    pooled[index] = tensor[int(start) : int(end)].mean(dim=0)
                tensor = pooled
                mask = np.zeros(self.segment_count, dtype=bool)
                mask[:active] = True
                if self.mask is None:
                    self.mask = mask
                elif not np.array_equal(self.mask, mask):
                    raise RuntimeError("Collector segment masks differ across modules")
            else:
                self.mask = np.ones(len(tensor), dtype=bool)
            self.values[(module_name, layer_index)] = (
                tensor if self.defer_cpu_copy else tensor.to(device="cpu").float().numpy()
            )

        return capture

    def __enter__(self) -> "ModuleSequenceCollector":
        for layer_index in self.selected_layers:
            layer = self.layers[layer_index]
            targets = {
                "residual": layer,
                "attention": getattr(layer, "self_attn", None),
                "mlp": getattr(layer, "mlp", None),
            }
            targets = {name: targets[name] for name in self.module_names}
            missing = [name for name, module in targets.items() if module is None]
            if missing:
                raise TypeError(f"Layer {layer_index} lacks modules: {missing}")
            for name, module in targets.items():
                self.handles.append(module.register_forward_hook(self._hook(name, layer_index)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def stacked(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        expected = {
            (module, layer)
            for module in self.module_names
            for layer in self.selected_layers
        }
        missing = sorted(expected - set(self.values))
        if missing:
            raise RuntimeError(f"Collector is incomplete: {missing[:5]}")
        if self.mask is None:
            raise RuntimeError("Collector captured no token sequence")
        arrays = {}
        for module in self.module_names:
            values = [self.values[(module, layer)] for layer in self.selected_layers]
            if self.defer_cpu_copy:
                import torch

                arrays[module] = torch.stack(values, dim=0).to(device="cpu").float().numpy()
            else:
                arrays[module] = np.stack(values, axis=0)
        return arrays, self.mask.copy()


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    location = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - location)))
    scale = mad if mad > 1e-8 else float(np.std(values))
    return location, max(scale, 1e-8)


class PairedSequenceEnsemble:
    """Fixed layer/module matched-pair direction with coherence-aware pooling."""

    def __init__(
        self,
        *,
        shrinkage: float = 0.9,
        top_k_fraction: float = 0.25,
        top_k_weight: float = 0.6,
        spike_penalty: float = 0.1,
    ) -> None:
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if not 0.0 <= top_k_weight <= 1.0:
            raise ValueError("top_k_weight must be in [0, 1]")
        if spike_penalty < 0.0:
            raise ValueError("spike_penalty must be non-negative")
        self.shrinkage = shrinkage
        self.top_k_fraction = top_k_fraction
        self.top_k_weight = top_k_weight
        self.spike_penalty = spike_penalty

    @staticmethod
    def _validate(
        sequences: Mapping[str, np.ndarray], mask: np.ndarray, metadata: Sequence[dict]
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in sequences.items()}
        if not arrays:
            raise ValueError("At least one sequence module is required")
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("Sequence module shapes differ")
        shape = next(iter(shapes))
        if len(shape) != 4 or shape[0] != len(metadata):
            raise ValueError("Sequences must have shape [sample, layer, token, hidden]")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (shape[0], shape[2]) or np.any(mask.sum(axis=1) == 0):
            raise ValueError("Sequence mask is invalid")
        if not np.isfinite(np.stack(list(arrays.values()))).all():
            raise ValueError("Sequences contain non-finite values")
        return arrays, mask

    def fit(
        self,
        sequences: Mapping[str, np.ndarray],
        mask: np.ndarray,
        metadata: Sequence[dict],
    ) -> "PairedSequenceEnsemble":
        arrays, mask = self._validate(sequences, mask, metadata)
        benign = np.asarray([row["role"] == "benign" for row in metadata])
        if not benign.any():
            raise ValueError("Benign samples are required for coherence calibration")
        self.module_names_ = tuple(sorted(arrays))
        self.layer_count_ = next(iter(arrays.values())).shape[1]
        self.models_: dict[tuple[str, int], PairedContrastiveDirection] = {}
        self.variance_calibration_: dict[tuple[str, int], tuple[float, float]] = {}
        self.active_components_: list[tuple[str, int]] = []
        for module_name in self.module_names_:
            for layer_position in range(self.layer_count_):
                vectors = masked_sequence_mean(arrays[module_name][:, layer_position], mask)
                try:
                    model = PairedContrastiveDirection(shrinkage=self.shrinkage).fit(
                        vectors, metadata
                    )
                except ValueError as error:
                    if str(error) == "Matched pairs do not define a non-zero direction":
                        continue
                    raise
                token_scores = self._token_scores(
                    model, arrays[module_name][:, layer_position], mask
                )
                variances = np.asarray(
                    [np.var(scores) for scores in token_scores], dtype=np.float64
                )
                self.models_[(module_name, layer_position)] = model
                self.variance_calibration_[(module_name, layer_position)] = (
                    _robust_location_scale(variances[benign])
                )
                self.active_components_.append((module_name, layer_position))
        if not self.active_components_:
            raise ValueError("No layer/module defines a non-zero matched-pair direction")
        return self

    @staticmethod
    def _token_scores(
        model: PairedContrastiveDirection, values: np.ndarray, mask: np.ndarray
    ) -> list[np.ndarray]:
        scores = []
        for sample, active in zip(values, mask):
            scores.append(model.decision_function(sample[active]))
        return scores

    def decision_function(
        self, sequences: Mapping[str, np.ndarray], mask: np.ndarray
    ) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("PairedSequenceEnsemble must be fitted before scoring")
        arrays, mask = self._validate(
            sequences,
            mask,
            [{"role": "benign"}] * next(iter(sequences.values())).shape[0],
        )
        if tuple(sorted(arrays)) != self.module_names_ or next(iter(arrays.values())).shape[1] != self.layer_count_:
            raise ValueError("Scoring sequence contract differs from fit contract")
        components = []
        for module_name, layer_position in self.active_components_:
            model = self.models_[(module_name, layer_position)]
            token_scores = self._token_scores(
                model, arrays[module_name][:, layer_position], mask
            )
            location, scale = self.variance_calibration_[(module_name, layer_position)]
            pooled = []
            for scores in token_scores:
                count = max(1, int(np.ceil(len(scores) * self.top_k_fraction)))
                top_k = np.partition(scores, len(scores) - count)[-count:]
                variance_excess = max(0.0, (float(np.var(scores)) - location) / scale)
                pooled.append(
                    self.top_k_weight * float(np.mean(top_k))
                    + (1.0 - self.top_k_weight) * float(np.mean(scores))
                    - self.spike_penalty * variance_excess
                )
            components.append(np.asarray(pooled, dtype=np.float64))
        return np.median(np.stack(components, axis=1), axis=1)


def topk_total_variation_lower_bound(
    first_ids: np.ndarray,
    first_log_probs: np.ndarray,
    second_ids: np.ndarray,
    second_log_probs: np.ndarray,
) -> float:
    """Lower bound on total variation using only the union of stored top-k tokens."""
    first = {int(token): float(np.exp(logp)) for token, logp in zip(first_ids, first_log_probs)}
    second = {
        int(token): float(np.exp(logp)) for token, logp in zip(second_ids, second_log_probs)
    }
    union = set(first) | set(second)
    return 0.5 * sum(abs(first.get(token, 0.0) - second.get(token, 0.0)) for token in union)
