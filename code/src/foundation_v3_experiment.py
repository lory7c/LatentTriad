"""Shared loading, thresholding, and metrics for Foundation-v3 experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA_VERSION = "skillprobe-foundation-experiment-contract-v1.0"
TARGET_FPR = 0.02
ROLE_LABELS = {"clean": 0, "malicious": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_hashed_utf8_text(path: Path, expected_sha256: str) -> str:
    """Read contract text without universal-newline translation."""
    payload = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Contract text hash mismatch: {path}")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Contract text is not valid UTF-8: {path}") from error


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_bytes_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ).encode("utf-8"),
    )


def load_contract(
    contract_dir: Path,
    expected_phase: str,
    model_label: Optional[str] = None,
    *,
    require_pairs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = contract_dir.resolve()
    status_path = root / "status.json"
    manifest_path = root / "manifest.jsonl"
    status = read_json(status_path)
    if (
        status.get("schema_version") != SCHEMA_VERSION
        or status.get("status") != "ready"
        or status.get("phase") != expected_phase
        or status.get("selection_uses_model_or_detector_scores") is not False
        or status.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("Foundation experiment contract is incomplete or stale")
    rows = read_jsonl(manifest_path)
    if len(rows) != status.get("sample_count"):
        raise ValueError("Contract sample count differs from status")
    observed = Counter((row.get("pair_id"), row.get("role")) for row in rows)
    if any(count != 1 for count in observed.values()):
        raise ValueError("Contract contains duplicate pair-role samples")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed_splits = {"train", "dev"} if expected_phase == "development" else {"test"}
    for row in rows:
        by_pair[row["pair_id"]].append(row)
        if row.get("split") not in allowed_splits:
            raise ValueError("Contract phase exposes an unexpected split")
        if row.get("label") != ROLE_LABELS.get(row.get("role")):
            raise ValueError("Contract role and label disagree")
        package_path = root / row["package_file"]
        if sha256_file(package_path) != row.get("package_sha256"):
            raise ValueError(f"Package hash mismatch: {row.get('sample_id')}")
        if model_label is not None:
            model = row.get("model_views", {}).get(model_label)
            if not isinstance(model, dict):
                raise ValueError(f"Missing model view {model_label}")
            prompt_path = root / model["prompt_file"]
            if sha256_file(prompt_path) != model.get("prompt_sha256"):
                raise ValueError(f"Prompt hash mismatch: {row.get('sample_id')}")
    if require_pairs:
        for pair_id, pair_rows in by_pair.items():
            if {row["role"] for row in pair_rows} != set(ROLE_LABELS):
                raise ValueError(f"Pair does not contain both roles: {pair_id}")
            if len({row["split"] for row in pair_rows}) != 1:
                raise ValueError(f"Pair roles cross splits: {pair_id}")
            if len({row["leakage_cluster_id"] for row in pair_rows}) != 1:
                raise ValueError(f"Pair roles disagree on cluster: {pair_id}")
    if len(by_pair) != status.get("pair_count"):
        raise ValueError("Contract pair count differs from status")
    return rows, status


def group_label_weights(rows: list[dict[str, Any]], labels: np.ndarray) -> np.ndarray:
    if len(rows) != len(labels):
        raise ValueError("Rows and labels differ in length")
    cells = [
        (row["leakage_cluster_id"], int(label))
        for row, label in zip(rows, labels)
    ]
    counts = Counter(cells)
    weights = np.asarray([1.0 / counts[cell] for cell in cells], dtype=np.float64)
    return weights * (len(weights) / weights.sum())


def weighted_operating_point(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    weights: Optional[np.ndarray] = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    predictions = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tp = float(weights[predictions & positive].sum())
    fp = float(weights[predictions & negative].sum())
    tn = float(weights[~predictions & negative].sum())
    fn = float(weights[~predictions & positive].sum())
    return {
        "fpr": fp / max(fp + tn, 1e-12),
        "fnr": fn / max(fn + tp, 1e-12),
        "precision": tp / max(tp + fp, 1e-12),
        "recall": tp / max(tp + fn, 1e-12),
    }


def select_dev_threshold(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = TARGET_FPR,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid dev labels or scores")
    weights = group_label_weights(rows, labels)
    no_block_threshold = float(np.nextafter(scores.max(), np.inf))
    candidates = [no_block_threshold] + sorted(
        {float(value) for value in scores}, reverse=True
    )
    feasible = []
    for threshold in candidates:
        point = weighted_operating_point(labels, scores, threshold, weights)
        if point["fpr"] <= target_fpr + 1e-12:
            feasible.append((point["recall"], threshold, point))
    if not feasible:
        raise AssertionError("The no-block threshold must satisfy the FPR constraint")
    recall, threshold, point = max(feasible, key=lambda row: (row[0], row[1]))
    return {
        "selection_split": "dev",
        "target_group_weighted_fpr": target_fpr,
        "threshold": threshold,
        "group_weighted_operating_point": point,
        "feasible_candidate_count": len(feasible),
        "maximum_feasible_weighted_recall": recall,
    }


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "tp": int(((labels == 1) & (predictions == 1)).sum()),
        "tn": int(((labels == 0) & (predictions == 0)).sum()),
        "fp": int(((labels == 0) & (predictions == 1)).sum()),
        "fn": int(((labels == 1) & (predictions == 0)).sum()),
    }


def micro_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    weights: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    counts = confusion(labels, predictions)
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result: dict[str, Any] = {
        **counts,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    if len(np.unique(labels)) == 2:
        # Per-sample (unweighted) ranking metrics — retained for backward compat
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["auprc"] = float(average_precision_score(labels, scores))
        # Group-weighted ranking metrics — consistent with FPR/FNR operating-point
        # weighting.  Each (leakage_cluster_id, label) cell receives equal mass.
        if weights is not None:
            result["auroc_group_weighted"] = float(
                roc_auc_score(labels, scores, sample_weight=weights)
            )
            result["auprc_group_weighted"] = float(
                average_precision_score(labels, scores, sample_weight=weights)
            )
        else:
            result["auroc_group_weighted"] = None
            result["auprc_group_weighted"] = None
    else:
        result["auroc"] = None
        result["auprc"] = None
        result["auroc_group_weighted"] = None
        result["auprc_group_weighted"] = None
    return result


def pair_ranking_accuracy(rows: list[dict[str, Any]], scores: np.ndarray) -> float:
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for row, score in zip(rows, scores):
        by_pair[row["pair_id"]][row["role"]] = float(score)
    if any(set(value) != set(ROLE_LABELS) for value in by_pair.values()):
        raise ValueError("Pair ranking requires both roles")
    return float(
        np.mean(
            [
                value["malicious"] > value["clean"]
                for value in by_pair.values()
            ]
        )
    )


def cluster_bootstrap_intervals(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    repeats: int = 10000,
    seed: int = 20260718,
    weights: Optional[np.ndarray] = None,
) -> dict[str, list[float]]:
    clusters = sorted({row["leakage_cluster_id"] for row in rows})
    by_cluster = {
        cluster: np.asarray(
            [index for index, row in enumerate(rows) if row["leakage_cluster_id"] == cluster],
            dtype=np.int64,
        )
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(repeats):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([by_cluster[cluster] for cluster in sampled])
        sampled_weights = weights[indices] if weights is not None else None
        report = micro_metrics(labels[indices], scores[indices], threshold, weights=sampled_weights)
        for key in ("fpr", "fnr", "precision", "recall", "f1", "auroc", "auprc",
                     "auroc_group_weighted", "auprc_group_weighted"):
            if key in report and report[key] is not None:
                values[key].append(float(report[key]))
    return {
        key: [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ]
        for key, samples in sorted(values.items())
        if samples
    }


def stratified_metrics(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    strata: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[
            f"surface_explicitness={row.get('surface_explicitness', 'unknown')}"
        ].append(index)
        for name, enabled in sorted(row.get("slices", {}).items()):
            if enabled:
                strata[f"slice={name}"].append(index)
    return {
        name: {
            "sample_count": len(indices),
            "pair_count": len({rows[index]["pair_id"] for index in indices}),
            "metrics": micro_metrics(
                labels[np.asarray(indices)], scores[np.asarray(indices)], threshold
            ),
        }
        for name, indices in sorted(strata.items())
        if len({int(labels[index]) for index in indices}) == 2
    }


def evaluate_scores(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    bootstrap_repeats: int = 10000,
    allow_unpaired: bool = False,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(rows) != len(labels) or len(labels) != len(scores):
        raise ValueError("Evaluation rows, labels, and scores differ in length")
    pair_accuracy = None
    if not allow_unpaired:
        pair_accuracy = pair_ranking_accuracy(rows, scores)
    # Compute group weights once so that micro_metrics, bootstrap, and
    # stratified all use the same per-(cluster,label)-cell equal-mass weights.
    eval_weights = group_label_weights(rows, labels)
    return {
        "sample_count": len(rows),
        "pair_count": len({row["pair_id"] for row in rows}),
        "leakage_cluster_count": len({row["leakage_cluster_id"] for row in rows}),
        "threshold": threshold,
        "micro": micro_metrics(labels, scores, threshold, weights=eval_weights),
        "pair_ranking_accuracy": pair_accuracy,
        "pair_ranking_scope": "not_applicable_unpaired" if allow_unpaired else "all_pairs",
        "cluster_bootstrap_repeats": bootstrap_repeats,
        "cluster_bootstrap_95_ci": cluster_bootstrap_intervals(
            rows,
            labels,
            scores,
            threshold,
            repeats=bootstrap_repeats,
            weights=eval_weights,
        ),
        "stratified": stratified_metrics(rows, labels, scores, threshold),
    }


def prediction_rows(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "pair_id": row["pair_id"],
            "role": row["role"],
            "label": int(row["label"]),
            "split": row["split"],
            "leakage_cluster_id": row["leakage_cluster_id"],
            "score": float(score),
            "prediction": int(float(score) >= threshold),
            "package_sha256": row["package_sha256"],
        }
        for row, score in zip(rows, scores)
    ]
