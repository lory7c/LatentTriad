#!/usr/bin/env python3
"""Extract boundary hidden states for decoy skills on GPU.

Minimal extraction: forward pass through Llama-3.1-8B, capture hidden states
at every normalized layer depth for the LAST token (boundary).
"""

import json, os, sys, time, argparse
from pathlib import Path
import numpy as np

NORMALIZED_DEPTHS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)

def normalized_layer_indices(layer_count):  # type: (int) -> list
    return sorted({
        min(layer_count - 1, max(0, int(round(depth * layer_count)) - 1))
        for depth in NORMALIZED_DEPTHS
    })


class BoundaryCollector:
    """Captures last-token hidden states at specified layers."""
    def __init__(self, layers, layer_indices, torch):  # type: (..., list, ...) -> None
        self.layer_indices = layer_indices
        self.boundary: dict = {}
        self.handles = [
            layers[index].register_forward_hook(self._hook(index))
            for index in layer_indices
        ]

    def _hook(self, layer_index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            last_token = hidden[0, -1, :].float()
            self.boundary[layer_index] = last_token.detach().cpu().numpy().astype(np.float16)
        return hook

    def result(self) -> np.ndarray:
        return np.stack([self.boundary[idx] for idx in self.layer_indices])

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL file with decoy skills (field: skill_md)")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    parser.add_argument("--model-name", type=Path, default="/work/yz/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=24576)
    args = parser.parse_args()

    decoys = [json.loads(l) for l in open(args.input) if l.strip()]
    texts = [d.get("full_prompt", d.get("skill_md", "")) for d in decoys]
    print(f"Loaded {len(texts)} decoy texts")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if backbone is None or layers is None:
        raise ValueError("Cannot access model.layers")
    layer_indices = normalized_layer_indices(len(layers))
    print(f"Model loaded. Layers: {len(layers)}, sampled: {layer_indices}")

    collector = BoundaryCollector(layers, layer_indices, torch)
    all_boundary = []

    try:
        for i, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=args.max_length)
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded["attention_mask"].to(args.device)

            with torch.inference_mode():
                backbone(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True)

            bnd = collector.result()  # [8, 4096]
            all_boundary.append(bnd)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(texts)}] done", flush=True)
    finally:
        collector.close()

    boundary_array = np.stack(all_boundary, axis=0)  # [N, 8, 4096]
    print(f"Boundary array: {boundary_array.shape}, dtype={boundary_array.dtype}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, boundary=boundary_array)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
