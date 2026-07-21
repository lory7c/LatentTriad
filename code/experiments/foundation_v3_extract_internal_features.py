#!/usr/bin/env python3
"""Extract shared pre-action features for Ours, RouteGuard, and AgentLens."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from experiments.routeguard_full_v3 import (  # noqa: E402
    ATTENTION_FEATURES,
    HIDDEN_FEATURES,
    LinearMemoryCollector,
    structural_token_windows,
    token_positions_for_char_span,
)
from src.foundation_loaded_prompt_contract_v1 import token_ids_sha256  # noqa: E402
from src.foundation_v3_experiment import (  # noqa: E402
    load_contract,
    read_hashed_utf8_text,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from scripts.validate_local_model_snapshot_v3 import validate_snapshot  # noqa: E402


SCHEMA_VERSION = "skillprobe-foundation-internal-features-v1.0"
NORMALIZED_DEPTHS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
CONTRACT_READER_PATH = CODE_DIR / "src/foundation_v3_experiment.py"


def positions_for_spans(
    offsets: list[tuple[int, int]], spans: list[list[int]]
) -> list[int]:
    positions = {
        index
        for span in spans
        for index in token_positions_for_char_span(offsets, (span[0], span[1]))
    }
    return sorted(positions)


def normalized_layer_indices(layer_count: int) -> list[int]:
    if layer_count < 8:
        raise ValueError("Internal feature extraction requires at least eight layers")
    return sorted(
        {
            min(layer_count - 1, max(0, int(round(depth * layer_count)) - 1))
            for depth in NORMALIZED_DEPTHS
        }
    )


class RegionCollector:
    def __init__(self, layers, layer_indices: list[int], torch):
        self.layer_indices = layer_indices
        self.torch = torch
        self.declaration_positions: list[int] = []
        self.operation_positions: list[int] = []
        self.declaration: dict[int, np.ndarray] = {}
        self.operation: dict[int, np.ndarray] = {}
        self.handles = [
            layers[index].register_forward_hook(self._hook(index))
            for index in layer_indices
        ]

    def set_positions(
        self, declaration_positions: list[int], operation_positions: list[int]
    ) -> None:
        if not declaration_positions or not operation_positions:
            raise ValueError("Alignment regions must both contain tokens")
        self.declaration_positions = declaration_positions
        self.operation_positions = operation_positions
        self.declaration = {}
        self.operation = {}

    def _hook(self, layer_index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            declaration = hidden[0, self.declaration_positions, :].float().mean(dim=0)
            operation = hidden[0, self.operation_positions, :].float().mean(dim=0)
            self.declaration[layer_index] = (
                declaration.detach().cpu().numpy().astype(np.float16)
            )
            self.operation[layer_index] = (
                operation.detach().cpu().numpy().astype(np.float16)
            )

        return hook

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        missing = [
            index
            for index in self.layer_indices
            if index not in self.declaration or index not in self.operation
        ]
        if missing:
            raise RuntimeError(f"Region hooks did not fire for layers {missing}")
        return (
            np.stack([self.declaration[index] for index in self.layer_indices]),
            np.stack([self.operation[index] for index in self.layer_indices]),
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def write_feature_archive(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_model_identity(
    model_path: Path,
    model_label: str,
    audit_path: Path | None,
    validation_path: Path | None,
    *,
    diagnostic: bool,
) -> dict[str, Any]:
    if audit_path is None or validation_path is None:
        if diagnostic:
            return {"status": "not_required_for_limited_diagnostic"}
        raise ValueError(
            "Complete extraction requires --model-snapshot-audit and "
            "--model-snapshot-validation"
        )
    audit_path = audit_path.resolve()
    validation_path = validation_path.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    errors, verified = validate_snapshot(
        audit,
        expected_model_label=model_label,
        verify_file_hashes=True,
    )
    validator_path = CODE_DIR / "scripts/validate_local_model_snapshot_v3.py"
    auditor_path = CODE_DIR / "scripts/audit_local_model_snapshot_v3.py"
    if errors:
        raise ValueError(f"Model snapshot validation failed: {errors[:3]}")
    if (
        validation.get("status") != "valid"
        or validation.get("file_hashes_verified") is not True
        or validation.get("aggregate_sha256") != audit.get("aggregate_sha256")
        or validation.get("provenance", {}).get("audit_sha256")
        != sha256_file(audit_path)
        or validation.get("provenance", {}).get("validator_sha256")
        != sha256_file(validator_path)
        or audit.get("provenance", {}).get("script_sha256")
        != sha256_file(auditor_path)
        or Path(audit.get("model_dir", "")).resolve() != model_path
    ):
        raise ValueError("Model snapshot validation is stale or mismatched")
    return {
        "status": "validated",
        "aggregate_sha256": audit["aggregate_sha256"],
        "audit_sha256": sha256_file(audit_path),
        "validation_sha256": sha256_file(validation_path),
        "verified_file_count": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "sealed_test"), required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-name", type=Path, required=True)
    parser.add_argument("--model-snapshot-audit", type=Path)
    parser.add_argument("--model-snapshot-validation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-window-tokens", type=int, default=64)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Accept a class-balanced non-paired benchmark contract such as MASB/MASW.",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Internal feature output already exists: {output_dir}")
    contract_root = args.contract_dir.resolve()
    rows, contract = load_contract(
        contract_root,
        args.phase,
        model_label=args.model_label,
        require_pairs=not args.allow_unpaired,
    )
    selected_indices = [
        index
        for index in range(len(rows))
        if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]
    if not selected_indices:
        raise ValueError("Internal feature shard contains no samples")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_name.resolve()
    model_identity = validate_model_identity(
        model_path,
        args.model_label,
        args.model_snapshot_audit,
        args.model_snapshot_validation,
        diagnostic=args.limit is not None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Internal feature spans require a fast tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if backbone is None or layers is None:
        raise ValueError("Backbone must expose model.layers")
    layer_indices = normalized_layer_indices(len(layers))
    routeguard = LinearMemoryCollector(layers, layer_indices, torch)
    regions = RegionCollector(layers, layer_indices, torch)
    attention_rows = []
    route_hidden_rows = []
    response_rows = []
    declaration_rows = []
    operation_rows = []
    metadata_rows = []
    elapsed_ms = []
    tokenization_ms = []
    token_counts = []
    window_counts = []
    peak_memory = []
    try:
        for completed, index in enumerate(selected_indices, start=1):
            row = rows[index]
            view = row["model_views"][args.model_label]
            prompt_path = contract_root / view["prompt_file"]
            prompt = read_hashed_utf8_text(prompt_path, view["prompt_sha256"])
            tokenization_start = time.perf_counter_ns()
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            tokenization_elapsed = (
                time.perf_counter_ns() - tokenization_start
            ) / 1_000_000.0
            offsets = [
                tuple(values)
                for values in encoded.pop("offset_mapping")[0].tolist()
            ]
            token_ids = encoded["input_ids"][0].tolist()
            if (
                len(token_ids) != view["token_count"]
                or token_ids_sha256(token_ids) != view["input_token_ids_sha256"]
            ):
                raise ValueError(f"Token contract mismatch: {row['sample_id']}")
            package_span = tuple(view["package_char_span"])
            declaration_positions = positions_for_spans(
                offsets, view["declaration_char_spans"]
            )
            operation_positions = positions_for_spans(
                offsets, view["operation_char_spans"]
            )
            trusted_positions = token_positions_for_char_span(
                offsets, tuple(view["task_char_span"])
            )
            response_positions = [len(token_ids) - 1]
            windows = structural_token_windows(
                prompt, offsets, package_span, args.max_window_tokens
            )
            routeguard.set_spans(response_positions, trusted_positions, windows)
            regions.set_positions(declaration_positions, operation_positions)
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded["attention_mask"].to(args.device)
            context_limit = int(getattr(model.config, "max_position_embeddings", 0))
            if context_limit and input_ids.shape[1] > context_limit:
                raise ValueError("Prompt exceeds model context; truncation is forbidden")
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
                torch.cuda.synchronize(args.device)
            start = time.perf_counter_ns()
            with torch.inference_mode():
                backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            attention, route_hidden, response = routeguard.result()
            declaration, operation = regions.result()
            attention_rows.append(attention)
            route_hidden_rows.append(route_hidden)
            response_rows.append(response)
            declaration_rows.append(declaration)
            operation_rows.append(operation)
            elapsed_ms.append(elapsed)
            tokenization_ms.append(tokenization_elapsed)
            token_counts.append(len(token_ids))
            window_counts.append(len(windows))
            peak_memory.append(
                int(torch.cuda.max_memory_allocated(args.device))
                if torch.cuda.is_available() and str(args.device).startswith("cuda")
                else 0
            )
            metadata_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "role": row["role"],
                    "label": row["label"],
                    "split": row["split"],
                    "leakage_cluster_id": row["leakage_cluster_id"],
                    "package_sha256": row["package_sha256"],
                    "prompt_sha256": view["prompt_sha256"],
                    "token_count": len(token_ids),
                    "structural_window_count": len(windows),
                    "forward_with_hooks_ms": elapsed,
                    "tokenization_ms": tokenization_elapsed,
                    "standalone_tokenization_and_hooks_ms": (
                        tokenization_elapsed + elapsed
                    ),
                    "peak_memory_allocated_bytes": peak_memory[-1],
                }
            )
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(selected_indices),
                        "sample_id": row["sample_id"],
                        "token_count": len(token_ids),
                        "elapsed_ms": elapsed,
                    }
                ),
                flush=True,
            )
    finally:
        regions.close()
        routeguard.close()

    output_dir.mkdir(parents=True)
    feature_path = output_dir / "features.npz"
    write_feature_archive(
        feature_path,
        attention=np.asarray(attention_rows, dtype=np.float32),
        route_hidden=np.asarray(route_hidden_rows, dtype=np.float32),
        response=np.asarray(response_rows, dtype=np.float16),
        declaration=np.asarray(declaration_rows, dtype=np.float16),
        operation=np.asarray(operation_rows, dtype=np.float16),
    )
    metadata_path = output_dir / "metadata.jsonl"
    write_jsonl_atomic(metadata_path, metadata_rows)
    elapsed_array = np.asarray(elapsed_ms, dtype=np.float64)
    tokenization_array = np.asarray(tokenization_ms, dtype=np.float64)
    standalone_array = elapsed_array + tokenization_array
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "limited_diagnostic"
            if args.limit is not None
            else "complete"
            if args.shard_count == 1
            else "complete_shard"
        ),
        "phase": args.phase,
        "model_label": args.model_label,
        "model_path": str(model_path),
        "attention_implementation_requested": args.attn_implementation,
        "attention_implementation_resolved": getattr(
            model.config, "_attn_implementation", None
        ),
        "sample_count": len(metadata_rows),
        "expected_phase_sample_count": len(rows),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "layer_indices": layer_indices,
        "normalized_depths": list(NORMALIZED_DEPTHS),
        "attention_feature_names": list(ATTENTION_FEATURES),
        "route_hidden_feature_names": list(HIDDEN_FEATURES),
        "max_window_tokens": args.max_window_tokens,
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "window_count_min": min(window_counts),
        "window_count_max": max(window_counts),
        "forward_with_hooks_latency_ms": {
            "mean": float(elapsed_array.mean()),
            "p50": float(np.quantile(elapsed_array, 0.50)),
            "p95": float(np.quantile(elapsed_array, 0.95)),
        },
        "tokenization_latency_ms": {
            "mean": float(tokenization_array.mean()),
            "p50": float(np.quantile(tokenization_array, 0.50)),
            "p95": float(np.quantile(tokenization_array, 0.95)),
        },
        "standalone_tokenization_and_hooks_latency_ms": {
            "mean": float(standalone_array.mean()),
            "p50": float(np.quantile(standalone_array, 0.50)),
            "p95": float(np.quantile(standalone_array, 0.95)),
        },
        "peak_memory_allocated_bytes_max": max(peak_memory),
        "generation_performed": False,
        "tool_dispatch_performed": False,
        "provenance": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "contract_reader_sha256": sha256_file(CONTRACT_READER_PATH),
            "routeguard_collector_sha256": sha256_file(
                Path(__file__).with_name("routeguard_full_v3.py")
            ),
            "contract_status_sha256": sha256_file(contract_root / "status.json"),
            "contract_manifest_sha256": contract["manifest_sha256"],
            "features_sha256": sha256_file(feature_path),
            "metadata_sha256": sha256_file(metadata_path),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "test_labels_used_for_selection": False,
            "model_snapshot": model_identity,
        },
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
