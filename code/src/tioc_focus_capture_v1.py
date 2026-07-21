"""Linear-memory capture of pre-action suffix attention to TIOC regions."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional, Sequence

import numpy as np


def compute_headwise_region_masses(
    query,
    key,
    query_positions: Sequence[int],
    region_positions: Mapping[str, Sequence[int]],
    scale: Optional[float] = None,
):
    """Compute exact causal region mass only for selected query positions."""
    import torch

    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("query and key must have shape [1, head, sequence, head_dim]")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions differ")
    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("query head count is not divisible by key head count")
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    positions = torch.as_tensor(query_positions, device=query.device, dtype=torch.long)
    if positions.numel() == 0 or int(positions.min()) < 0 or int(positions.max()) >= query.shape[2]:
        raise ValueError("query positions are empty or outside the sequence")
    selected_query = query.index_select(2, positions)
    scores = torch.matmul(selected_query, key.transpose(-2, -1)).float()
    scores *= float(scale) if scale is not None else query.shape[-1] ** -0.5
    key_positions = torch.arange(key.shape[2], device=query.device)
    causal = key_positions.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)
    masses = []
    for name, values in region_positions.items():
        region = torch.as_tensor(values, device=query.device, dtype=torch.long)
        if region.numel() == 0 or int(region.min()) < 0 or int(region.max()) >= key.shape[2]:
            raise ValueError(f"Region {name!r} is empty or outside the sequence")
        masses.append(probabilities.index_select(-1, region).sum(dim=-1))
    return torch.stack(masses, dim=-1)[0]


class SdpaRegionCollector:
    """Intercept selected SDPA layers while preserving the model's original output."""

    def __init__(
        self,
        layers,
        selected_layers: Sequence[int],
        query_positions: Sequence[int],
        region_positions: Mapping[str, Sequence[int]],
    ) -> None:
        self.layers = layers
        self.selected_layers = list(selected_layers)
        self.query_positions = list(query_positions)
        self.region_positions = OrderedDict(
            (name, list(values)) for name, values in region_positions.items()
        )
        self.current_layer: Optional[int] = None
        self.captured = {}
        self.hooks = []
        self.functional = None
        self.original_sdpa = None

    def _before(self, layer_index: int) -> None:
        if self.current_layer is not None:
            raise RuntimeError("Nested selected attention calls are unsupported")
        self.current_layer = layer_index

    def _after(self, layer_index: int) -> None:
        if self.current_layer != layer_index:
            raise RuntimeError("Selected attention layer context was lost")
        self.current_layer = None

    def _wrapped_sdpa(self, query, key, value, *args, **kwargs):
        if self.current_layer is not None:
            if self.current_layer in self.captured:
                raise RuntimeError("Selected attention layer invoked SDPA more than once")
            masses = compute_headwise_region_masses(
                query,
                key,
                self.query_positions,
                self.region_positions,
                scale=kwargs.get("scale"),
            )
            self.captured[self.current_layer] = (
                masses.detach().float().cpu().numpy().astype(np.float32)
            )
        assert self.original_sdpa is not None
        return self.original_sdpa(query, key, value, *args, **kwargs)

    def __enter__(self):
        import torch.nn.functional as functional

        self.functional = functional
        self.original_sdpa = functional.scaled_dot_product_attention
        functional.scaled_dot_product_attention = self._wrapped_sdpa
        for layer_index in self.selected_layers:
            attention = getattr(self.layers[layer_index], "self_attn", None)
            if attention is None:
                raise TypeError("Selected decoder block does not expose self_attn")

            def before(_module, _inputs, index=layer_index):
                self._before(index)

            def after(_module, _inputs, _output, index=layer_index):
                self._after(index)

            self.hooks.append(attention.register_forward_pre_hook(before))
            self.hooks.append(attention.register_forward_hook(after))
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if self.functional is not None and self.original_sdpa is not None:
            self.functional.scaled_dot_product_attention = self.original_sdpa
        self.current_layer = None

    def stacked(self) -> np.ndarray:
        if set(self.captured) != set(self.selected_layers):
            missing = sorted(set(self.selected_layers) - set(self.captured))
            raise RuntimeError(f"SDPA did not expose selected attention layers: {missing}")
        shapes = {value.shape for value in self.captured.values()}
        if len(shapes) != 1:
            raise ValueError(f"Selected attention mass shapes differ: {sorted(shapes)}")
        return np.stack([self.captured[index] for index in self.selected_layers], axis=0)
