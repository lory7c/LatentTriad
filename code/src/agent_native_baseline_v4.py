"""Shared contract and evaluation helpers for the agent-native v4 baselines.

The module deliberately keeps development selection on the training partition.
The 17-pair development partition is loaded only for a single frozen replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from experiments.tioc_hidden_pilot_v1 import (
    calibrated_threshold,
    dependency_group,
    group_label_weights,
    largest_cluster_excluded,
    metric_report,
    operating_point,
)


MODEL_LABEL = "qwen25_instruct"
SYSTEM_VARIANT = "standard_agent"
TARGET_FPR = 0.02

OPERATIONAL_BLOCK = re.compile(
    r"<operational_instructions>\n?(.*?)\n?</operational_instructions>", re.DOTALL
)
NATIVE_SYSTEM_BLOCK = re.compile(
    r"\A<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, value: object) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(),
    )


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    write_bytes_atomic(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows
        ).encode(),
    )


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def split_pair_ids(fold: dict) -> dict[str, set[str]]:
    if fold.get("status") != "ready":
        raise ValueError("Agent-native development fold is not ready")
    if fold.get("selection_uses_model_scores") is not False:
        raise ValueError("Agent-native development fold must be score-blind")
    if fold.get("train_dev_dependency_overlap_count") != 0:
        raise ValueError("Agent-native development fold leaks dependency clusters")
    values = fold.get("split_pair_ids")
    if not isinstance(values, dict):
        raise ValueError("Agent-native fold lacks split_pair_ids")
    result = {name: {str(value) for value in values[name]} for name in ("train", "dev")}
    if result["train"] & result["dev"]:
        raise ValueError("Agent-native train/dev pair IDs overlap")
    if len(result["train"]) != int(fold["train_pair_count"]):
        raise ValueError("Agent-native train pair count differs from fold")
    if len(result["dev"]) != int(fold["dev_pair_count"]):
        raise ValueError("Agent-native dev pair count differs from fold")
    return result


def _checked_span(text: str, span: object, name: str) -> str:
    if (
        not isinstance(span, list)
        or len(span) != 2
        or not all(isinstance(value, int) for value in span)
    ):
        raise ValueError(f"Invalid {name} character span")
    start, end = span
    if not 0 <= start < end <= len(text):
        raise ValueError(f"Out-of-range {name} character span")
    return text[start:end]


def extract_operational_instructions(actual_text: str) -> str:
    matches = OPERATIONAL_BLOCK.findall(actual_text)
    if len(matches) != 1 or not matches[0].strip():
        raise ValueError("Actual region does not contain exactly one operational block")
    return matches[0]


def extract_native_system(full_prompt: str) -> str:
    match = NATIVE_SYSTEM_BLOCK.search(full_prompt)
    if match is None or not match.group(1).strip():
        raise ValueError("Native prompt does not contain a trusted system block")
    return match.group(1)


def load_contract_samples(
    prompt_contract_dir: Path,
    development_fold_path: Path,
    *,
    model_label: str = MODEL_LABEL,
) -> tuple[dict[str, list[dict]], dict]:
    """Load and independently re-hash the fixed 54/17 pair contract."""

    prompt_dir = prompt_contract_dir.resolve()
    fold_path = development_fold_path.resolve()
    manifest_path = prompt_dir / "manifest.jsonl"
    status_path = prompt_dir / "status.json"
    status = json.loads(status_path.read_text())
    if (
        status.get("status") != "ready"
        or status.get("system_variant") != SYSTEM_VARIANT
        or status.get("pair_count") != 71
        or status.get("record_count") != 142
        or status.get("selection_uses_model_scores") is not False
    ):
        raise ValueError("Prompt contract is not the frozen standard-agent 71-pair view")
    manifest_sha256 = sha256_file(manifest_path)
    if status.get("provenance", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError("Prompt manifest hash differs from status")
    fold = json.loads(fold_path.read_text())
    splits = split_pair_ids(fold)
    split_by_pair = {
        pair_id: split_name
        for split_name, pair_ids in splits.items()
        for pair_id in pair_ids
    }
    rows = read_jsonl(manifest_path)
    observed = Counter((row.get("pair_id"), row.get("role")) for row in rows)
    expected = {
        (pair_id, role)
        for pair_id in split_by_pair
        for role in ("benign", "malicious")
    }
    if set(observed) != expected or any(count != 1 for count in observed.values()):
        raise ValueError("Prompt manifest does not exactly cover every pair-role sample")

    samples: dict[str, list[dict]] = {"train": [], "dev": []}
    for row in sorted(rows, key=lambda value: (value["pair_id"], value["role"])):
        if row.get("system_variant") != SYSTEM_VARIANT:
            raise ValueError("Prompt manifest mixes system variants")
        if row.get("boundary") != "native_assistant_generation_suffix_pre_first_action":
            raise ValueError("Prompt manifest mixes trace boundaries")
        view = row.get("model_views", {}).get(model_label, {}).get("full")
        if not isinstance(view, dict):
            raise ValueError(f"Prompt record lacks {model_label} full view")
        prompt_path = prompt_dir / view["prompt_file"]
        if sha256_file(prompt_path) != view["prompt_sha256"]:
            raise ValueError(f"Prompt hash mismatch: {row['trace_id']}")
        full_prompt = prompt_path.read_text()
        if len(full_prompt.encode()) == 0:
            raise ValueError("Prompt file is empty")
        spans = view.get("char_spans", {})
        actual_text = _checked_span(full_prompt, spans.get("actual"), "actual")
        task_text = _checked_span(full_prompt, spans.get("task"), "task")
        declaration_text = _checked_span(
            full_prompt, spans.get("declaration"), "declaration"
        )
        if sha256_text(actual_text) != row.get("actual_visible_sha256"):
            raise ValueError(f"Actual-region hash mismatch: {row['trace_id']}")
        if sha256_text(task_text) != row.get("task_sha256"):
            raise ValueError(f"Task-region hash mismatch: {row['trace_id']}")
        if sha256_text(declaration_text) != row.get("declaration_sha256"):
            raise ValueError(f"Declaration-region hash mismatch: {row['trace_id']}")
        role = row["role"]
        metadata = {
            key: row.get(key)
            for key in (
                "trace_id",
                "pair_id",
                "role",
                "base_group_id",
                "leakage_cluster_id",
                "source_dataset",
                "attack_channel",
                "visibility",
                "actual_truncated",
            )
        }
        samples[split_by_pair[row["pair_id"]]].append(
            {
                "label": int(role == "malicious"),
                "metadata": metadata,
                "full_prompt": full_prompt,
                "actual_text": actual_text,
                "task_text": task_text,
                "declaration_text": declaration_text,
                "operational_instructions": extract_operational_instructions(actual_text),
                "system_text": extract_native_system(full_prompt),
                "prompt_sha256": view["prompt_sha256"],
                "actual_sha256": row["actual_visible_sha256"],
            }
        )

    for split_name, expected_pairs in (("train", 54), ("dev", 17)):
        values = samples[split_name]
        if len(values) != 2 * expected_pairs:
            raise ValueError(f"Unexpected {split_name} sample count")
        labels = Counter(row["label"] for row in values)
        if labels != Counter({0: expected_pairs, 1: expected_pairs}):
            raise ValueError(f"Unexpected {split_name} label balance")
    train_clusters = {
        dependency_group(row["metadata"]) for row in samples["train"]
    }
    dev_clusters = {dependency_group(row["metadata"]) for row in samples["dev"]}
    if train_clusters & dev_clusters:
        raise ValueError("Loaded samples violate train/dev cluster isolation")
    provenance = {
        "prompt_status_sha256": sha256_file(status_path),
        "prompt_manifest_sha256": manifest_sha256,
        "development_fold_sha256": sha256_file(fold_path),
        "model_label": model_label,
        "system_variant": SYSTEM_VARIANT,
        "train_pair_count": 54,
        "dev_pair_count": 17,
        "train_leakage_cluster_count": len(train_clusters),
        "dev_leakage_cluster_count": len(dev_clusters),
        "train_dev_leakage_cluster_overlap_count": 0,
    }
    return samples, provenance


def materialize(samples: list[dict], text_key: str | None = None):
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    metadata = [row["metadata"] for row in samples]
    texts = [row[text_key] for row in samples] if text_key else None
    return labels, metadata, texts


def fixed_score_report(
    train_metadata: list[dict],
    train_labels: np.ndarray,
    train_scores: np.ndarray,
    dev_metadata: list[dict],
    dev_labels: np.ndarray,
    dev_scores: np.ndarray,
    *,
    target_fpr: float = TARGET_FPR,
) -> dict:
    arrays = (train_labels, train_scores, dev_labels, dev_scores)
    if any(np.asarray(values).ndim != 1 for values in arrays):
        raise ValueError("Labels and scores must be one-dimensional")
    if len(train_labels) != len(train_scores) or len(dev_labels) != len(dev_scores):
        raise ValueError("Labels and scores differ in length")
    if not np.isfinite(train_scores).all() or not np.isfinite(dev_scores).all():
        raise ValueError("Scores contain non-finite values")
    threshold = calibrated_threshold(
        train_labels,
        train_scores,
        group_label_weights(train_metadata, train_labels),
        target_fpr,
    )
    return {
        "selection_uses_dev": False,
        "threshold_calibration": f"train_oof_or_zero_shot_target_fpr_{target_fpr:g}",
        "threshold": threshold,
        "train_metrics": metric_report(train_metadata, train_labels, train_scores),
        "train_operating_point": operating_point(
            train_metadata, train_labels, train_scores, threshold
        ),
        "dev_metrics": metric_report(dev_metadata, dev_labels, dev_scores),
        "dev_operating_point": operating_point(
            dev_metadata, dev_labels, dev_scores, threshold
        ),
        "dev_largest_cluster_excluded": largest_cluster_excluded(
            dev_metadata, dev_labels, dev_scores
        ),
    }


def prediction_rows(
    samples: list[dict], scores: np.ndarray, threshold: float
) -> list[dict]:
    return [
        {
            **row["metadata"],
            "label": int(row["label"]),
            "score": float(scores[index]),
            "prediction": int(scores[index] >= threshold),
            "prompt_sha256": row["prompt_sha256"],
            "actual_sha256": row["actual_sha256"],
        }
        for index, row in enumerate(samples)
    ]


def benchmark_callable(
    functions: list[Callable[[], object]], *, warmup_rounds: int = 1, rounds: int = 5
) -> dict:
    if not functions or warmup_rounds < 0 or rounds <= 0:
        raise ValueError("Invalid benchmark configuration")
    for _ in range(warmup_rounds):
        for function in functions:
            function()
    elapsed_ms = []
    for _ in range(rounds):
        for function in functions:
            start = time.perf_counter_ns()
            function()
            elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    values = np.asarray(elapsed_ms, dtype=np.float64)
    return {
        "status": "complete",
        "scope": "detector_scoring_only",
        "sample_measurement_count": len(values),
        "warmup_rounds": warmup_rounds,
        "measurement_rounds": rounds,
        "mean_ms_per_sample": float(values.mean()),
        "p50_ms_per_sample": float(np.quantile(values, 0.50)),
        "p95_ms_per_sample": float(np.quantile(values, 0.95)),
        "min_ms_per_sample": float(values.min()),
        "max_ms_per_sample": float(values.max()),
        "model_loading_included": False,
        "prompt_construction_included": False,
        "artifact_serialization_included": False,
    }


def method_score_digest(
    train_trace_ids: list[str],
    train_scores: np.ndarray,
    dev_trace_ids: list[str],
    dev_scores: np.ndarray,
) -> str:
    rows = [
        {"split": "train", "trace_id": trace_id, "score": float(score)}
        for trace_id, score in zip(train_trace_ids, train_scores)
    ] + [
        {"split": "dev", "trace_id": trace_id, "score": float(score)}
        for trace_id, score in zip(dev_trace_ids, dev_scores)
    ]
    return canonical_sha256(rows)

