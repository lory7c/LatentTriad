#!/usr/bin/env python3
"""Extract exact description-operation-boundary states from external skills.

Unlike the legacy MSB/MASB scripts, this extractor never substitutes the first
N prompt tokens for the description and never pools the whole prompt as the
operation.  It serializes package bytes, binds semantic character spans, maps
them jointly through a fast tokenizer, and performs one pre-action prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from src.external_triad_contract_v1 import (  # noqa: E402
    DEFAULT_TOOLS,
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    canonical_json_sha256,
    locate_package_regions,
    partition_token_positions,
    render_agent_prompt,
    render_loaded_package,
    sha256_bytes,
    sha256_text,
    token_ids_sha256,
)


SCHEMA_VERSION = "skillprobe-external-triad-features-v2.0"
NORMALIZED_DEPTHS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
DEFAULT_TASK = "Use the loaded skill to complete the assigned task."


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    label: int
    source: str
    group_id: str
    relative_path: str
    source_path: Path
    virtual_markdown: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_layer_indices(layer_count: int) -> list[int]:
    if layer_count < len(NORMALIZED_DEPTHS):
        raise ValueError("TRIAD extraction requires at least eight model layers")
    return sorted(
        {
            min(layer_count - 1, max(0, int(round(depth * layer_count)) - 1))
            for depth in NORMALIZED_DEPTHS
        }
    )


def discover_malskillbench(root: Path, source: str) -> list[SourceSample]:
    samples: list[SourceSample] = []
    for subdir, label in (("benign", 0), ("malware", 1)):
        class_root = root / subdir
        if not class_root.is_dir():
            raise ValueError(f"Missing MalSkillBench class directory: {class_root}")
        for package_root in sorted(class_root.iterdir(), key=lambda path: path.name):
            if not package_root.is_dir() or not (package_root / "SKILL.md").is_file():
                continue
            relative = package_root.relative_to(root).as_posix()
            samples.append(
                SourceSample(
                    sample_id=f"{source}:{subdir}:{package_root.name}",
                    label=label,
                    source=source,
                    group_id=f"{source}:{package_root.name}",
                    relative_path=relative,
                    source_path=package_root,
                    virtual_markdown=False,
                )
            )
    if not samples:
        raise ValueError(f"No MalSkillBench packages found below {root}")
    return samples


def discover_markdown_tree(
    root: Path, source: str, label: int
) -> list[SourceSample]:
    samples: list[SourceSample] = []
    for path in sorted(root.rglob("*.md"), key=lambda value: value.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        identifier = sha256_text(f"{source}\0{relative}")[:20]
        parent = path.parent.relative_to(root).as_posix() or "."
        samples.append(
            SourceSample(
                sample_id=f"{source}:{identifier}",
                label=label,
                source=source,
                group_id=f"{source}:{parent}",
                relative_path=relative,
                source_path=path,
                virtual_markdown=True,
            )
        )
    if not samples:
        raise ValueError(f"No Markdown files found below {root}")
    return samples


def read_package_files(sample: SourceSample) -> list[tuple[str, bytes]]:
    if sample.virtual_markdown:
        return [("SKILL.md", sample.source_path.read_bytes())]
    files: list[tuple[str, bytes]] = []
    for path in sorted(sample.source_path.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"Package contains a symbolic link: {path}")
        if path.is_file():
            files.append((path.relative_to(sample.source_path).as_posix(), path.read_bytes()))
    return files


def select_shard(
    samples: list[SourceSample], shard_count: int, shard_index: int, limit: int | None
) -> tuple[list[SourceSample], tuple[int, int]]:
    chunk = math.ceil(len(samples) / shard_count)
    start = shard_index * chunk
    end = min(start + chunk, len(samples))
    selected = samples[start:end]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Selected shard contains no samples")
    return selected, (start, end)


def load_tools(path: Path | None) -> list[dict]:
    if path is None:
        return DEFAULT_TOOLS
    value = json.loads(path.read_text(encoding="utf-8"))
    tools = value.get("tools") if isinstance(value, dict) else value
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tool schema must be a non-empty list or {'tools': [...]} object")
    return tools


def content_tree_sha256(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, content in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_bytes(content).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def span_text_sha256(text: str, spans: list[list[int]]) -> str:
    return sha256_text("\n".join(text[start:end] for start, end in spans))


class TriadCollector:
    def __init__(self, layers, layer_indices: list[int], torch_module):
        self.layer_indices = layer_indices
        self.torch = torch_module
        self.description_positions: list[int] = []
        self.operation_positions: list[int] = []
        self.boundary_position = -1
        self.description: dict[int, np.ndarray] = {}
        self.operation: dict[int, np.ndarray] = {}
        self.boundary: dict[int, np.ndarray] = {}
        self.handles = [
            layers[index].register_forward_hook(self._hook(index))
            for index in layer_indices
        ]

    def set_positions(
        self,
        description_positions: list[int],
        operation_positions: list[int],
        boundary_position: int,
    ) -> None:
        if not description_positions or not operation_positions:
            raise ValueError("Description and operation token regions must be non-empty")
        if set(description_positions) & set(operation_positions):
            raise ValueError("Description and operation token regions overlap")
        if boundary_position in description_positions or boundary_position in operation_positions:
            raise ValueError("Pre-action boundary overlaps a semantic package region")
        self.description_positions = description_positions
        self.operation_positions = operation_positions
        self.boundary_position = boundary_position
        self.description = {}
        self.operation = {}
        self.boundary = {}

    def _hook(self, layer_index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            description = hidden[0, self.description_positions, :].float().mean(dim=0)
            operation = hidden[0, self.operation_positions, :].float().mean(dim=0)
            boundary = hidden[0, self.boundary_position, :].float()
            self.description[layer_index] = (
                description.detach().cpu().numpy().astype(np.float16)
            )
            self.operation[layer_index] = (
                operation.detach().cpu().numpy().astype(np.float16)
            )
            self.boundary[layer_index] = (
                boundary.detach().cpu().numpy().astype(np.float16)
            )

        return hook

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        missing = [
            index
            for index in self.layer_indices
            if index not in self.description
            or index not in self.operation
            or index not in self.boundary
        ]
        if missing:
            raise RuntimeError(f"TRIAD hooks did not fire for layers {missing}")
        return (
            np.stack([self.description[index] for index in self.layer_indices]),
            np.stack([self.operation[index] for index in self.layer_indices]),
            np.stack([self.boundary[index] for index in self.layer_indices]),
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_features(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-kind", choices=("malskillbench", "markdown_tree"), required=True
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--label", type=int, choices=(0, 1))
    parser.add_argument("--model-name", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tools-json", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--on-invalid", choices=("error", "skip"), default="error")
    parser.add_argument("--allow-operation-fallback", action="store_true")
    args = parser.parse_args()

    if args.input_kind == "markdown_tree" and args.label is None:
        parser.error("--label is required for markdown_tree input")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid --shard-count/--shard-index")
    if args.max_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Token budget and limit must be positive")

    input_root = args.input_root.resolve()
    model_path = args.model_name.resolve()
    output_dir = args.output_dir.resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"Output or partial output already exists: {output_dir}")
    if args.input_kind == "malskillbench":
        samples = discover_malskillbench(input_root, args.source)
    else:
        samples = discover_markdown_tree(input_root, args.source, int(args.label))
    selected, shard_range = select_shard(
        samples, args.shard_count, args.shard_index, args.limit
    )
    tools = load_tools(args.tools_json.resolve() if args.tools_json else None)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Exact TRIAD spans require a fast tokenizer")
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
        raise ValueError("Model backbone must expose model.layers")
    layer_indices = normalized_layer_indices(len(layers))
    model_context = int(getattr(model.config, "max_position_embeddings", 0))
    effective_limit = min(args.max_tokens, model_context) if model_context else args.max_tokens

    description_rows: list[np.ndarray] = []
    operation_rows: list[np.ndarray] = []
    boundary_rows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    elapsed_ms: list[float] = []
    collector = TriadCollector(layers, layer_indices, torch)
    started = time.time()
    try:
        for completed, sample in enumerate(selected, start=1):
            try:
                source_files = read_package_files(sample)
                package_text, files, structure = render_loaded_package(source_files)
                local_description, local_operation, region_status, file_metadata = (
                    locate_package_regions(files, package_text)
                )
                if (
                    not args.allow_operation_fallback
                    and (
                        region_status["fallback"] != "none"
                        or region_status["description_operation_overlap"]
                    )
                ):
                    raise ValueError(
                        "Package lacks disjoint description and operation regions"
                    )
                prompt, prompt_record = render_agent_prompt(
                    tokenizer, args.task, package_text, tools
                )
                package_start = prompt_record["package_char_span"][0]
                description_spans = [
                    [package_start + start, package_start + end]
                    for start, end in local_description
                ]
                operation_spans = [
                    [package_start + start, package_start + end]
                    for start, end in local_operation
                ]
                tokenization_started = time.perf_counter_ns()
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                tokenization_ms = (
                    time.perf_counter_ns() - tokenization_started
                ) / 1_000_000.0
                offsets = encoded.pop("offset_mapping")[0].tolist()
                token_ids = encoded["input_ids"][0].tolist()
                if len(token_ids) > effective_limit:
                    raise ValueError(
                        f"Prompt has {len(token_ids)} tokens, above limit {effective_limit}; "
                        "truncation is forbidden"
                    )
                description_positions, operation_positions, token_audit = (
                    partition_token_positions(
                        offsets, description_spans, operation_spans
                    )
                )
                boundary_position = len(token_ids) - 1
                if boundary_position < 0:
                    raise ValueError("Tokenizer returned an empty prompt")
                collector.set_positions(
                    description_positions, operation_positions, boundary_position
                )
            except (OSError, ValueError, UnicodeError) as error:
                if args.on_invalid == "error":
                    raise ValueError(f"{sample.sample_id}: {error}") from error
                exclusions.append(
                    {
                        "sample_id": sample.sample_id,
                        "relative_path": sample.relative_path,
                        "reason": type(error).__name__,
                        "detail": str(error),
                    }
                )
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(selected),
                            "sample_id": sample.sample_id,
                            "status": "excluded",
                            "reason": str(error),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded["attention_mask"].to(args.device)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            forward_started = time.perf_counter_ns()
            with torch.inference_mode():
                backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            forward_ms = (time.perf_counter_ns() - forward_started) / 1_000_000.0
            description, operation, boundary = collector.result()
            description_rows.append(description)
            operation_rows.append(operation)
            boundary_rows.append(boundary)
            labels.append(sample.label)
            elapsed_ms.append(forward_ms)
            metadata.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "source": sample.source,
                    "group_id": sample.group_id,
                    "source_relative_path": sample.relative_path,
                    "source_virtual_markdown": sample.virtual_markdown,
                    "source_content_tree_sha256": content_tree_sha256(files),
                    "package_sha256": sha256_text(package_text),
                    "package_structure": structure,
                    "package_files": file_metadata,
                    "prompt_sha256": prompt_record["prompt_sha256"],
                    "input_token_ids_sha256": token_ids_sha256(token_ids),
                    "token_count": len(token_ids),
                    "task_sha256": sha256_text(args.task),
                    "description_char_spans": description_spans,
                    "operation_char_spans": operation_spans,
                    "description_text_sha256": span_text_sha256(
                        prompt, description_spans
                    ),
                    "operation_text_sha256": span_text_sha256(prompt, operation_spans),
                    "description_token_count": token_audit["description_token_count"],
                    "operation_token_count": token_audit["operation_token_count"],
                    "boundary_straddling_token_count": token_audit[
                        "boundary_straddling_token_count"
                    ],
                    "boundary_token_position": boundary_position,
                    "assistant_boundary_kind": prompt_record[
                        "assistant_boundary_kind"
                    ],
                    "assistant_boundary_char_span": prompt_record[
                        "assistant_boundary_char_span"
                    ],
                    "region_status": region_status,
                    "tokenization_ms": tokenization_ms,
                    "forward_with_hooks_ms": forward_ms,
                }
            )
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(selected),
                        "sample_id": sample.sample_id,
                        "status": "extracted",
                        "tokens": len(token_ids),
                        "description_tokens": len(description_positions),
                        "operation_tokens": len(operation_positions),
                        "boundary_position": boundary_position,
                        "forward_ms": round(forward_ms, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        collector.close()

    if not metadata:
        raise ValueError("No valid samples were extracted")
    partial_dir.mkdir(parents=True)
    feature_path = partial_dir / "features.npz"
    metadata_path = partial_dir / "metadata.jsonl"
    exclusions_path = partial_dir / "exclusions.jsonl"
    write_features(
        feature_path,
        declaration=np.asarray(description_rows, dtype=np.float16),
        operation=np.asarray(operation_rows, dtype=np.float16),
        response=np.asarray(boundary_rows, dtype=np.float16),
        label=np.asarray(labels, dtype=np.int64),
        layer_indices=np.asarray(layer_indices, dtype=np.int64),
    )
    write_jsonl(metadata_path, metadata)
    write_jsonl(exclusions_path, exclusions)
    latency = np.asarray(elapsed_ms, dtype=np.float64)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "complete" if not exclusions else "complete_with_exclusions",
        "input_kind": args.input_kind,
        "input_root": str(input_root),
        "source": args.source,
        "discovered_sample_count": len(samples),
        "selected_shard_range": list(shard_range),
        "selected_sample_count": len(selected),
        "extracted_sample_count": len(metadata),
        "excluded_sample_count": len(exclusions),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "model_path": str(model_path),
        "model_context_limit": model_context,
        "configured_token_limit": args.max_tokens,
        "effective_token_limit": effective_limit,
        "layer_indices": layer_indices,
        "normalized_depths": list(NORMALIZED_DEPTHS),
        "task": args.task,
        "task_sha256": sha256_text(args.task),
        "tool_schema_sha256": canonical_json_sha256(tools),
        "region_semantics": {
            "declaration": "root SKILL.md frontmatter, title, and first prose section",
            "operation": "remaining root SKILL.md plus all textual auxiliary files",
            "response": "last native assistant-generation-boundary token",
            "token_partition": "maximum character overlap with deterministic tie break",
            "truncation_allowed": False,
            "legacy_first_100_proxy": False,
            "legacy_whole_prompt_operation_proxy": False,
        },
        "forward_with_hooks_latency_ms": {
            "mean": float(latency.mean()),
            "p50": float(np.quantile(latency, 0.50)),
            "p95": float(np.quantile(latency, 0.95)),
        },
        "generation_performed": False,
        "tool_dispatch_performed": False,
        "wall_time_seconds": time.time() - started,
        "provenance": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "contract_module_sha256": sha256_file(
                CODE_DIR / "src/external_triad_contract_v1.py"
            ),
            "features_sha256": sha256_file(feature_path),
            "metadata_sha256": sha256_file(metadata_path),
            "exclusions_sha256": sha256_file(exclusions_path),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
    }
    write_json(partial_dir / "summary.json", summary)
    shutil.move(str(partial_dir), str(output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
