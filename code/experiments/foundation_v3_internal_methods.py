#!/usr/bin/env python3
"""Develop and evaluate pre-action internal methods on Foundation-v3."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from src.foundation_v3_experiment import (  # noqa: E402
    TARGET_FPR,
    evaluate_scores,
    group_label_weights,
    load_contract,
    micro_metrics,
    prediction_rows,
    select_dev_threshold,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = "skillprobe-foundation-internal-methods-v1.0"
ARRAY_NAMES = ("attention", "route_hidden", "response", "declaration", "operation")
METHODS = ("routeguard", "agentlens", "ours_relational", "residual_ablation", "ours_v2")
C_GRID = (0.01, 0.1, 1.0, 10.0)
PCA_GRID = (16, 32, 64)
TOPK_GRID = (10, 20, 50)
SEED = 20260718


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def split_indices(rows: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == split],
        dtype=np.int64,
    )


def summarize_latency(values: list[float], scope: str, rounds: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "scope": scope,
        "measurement_count": len(array),
        "warmup_rounds": 1,
        "measurement_rounds": rounds,
        "mean_ms_per_sample": float(array.mean()),
        "p50_ms_per_sample": float(np.quantile(array, 0.50)),
        "p95_ms_per_sample": float(np.quantile(array, 0.95)),
        "model_loading_included": False,
        "feature_extraction_included": False,
        "feature_transform_and_classifier_included": True,
    }


def benchmark_scorer(
    scorer: Callable[[dict[str, np.ndarray]], np.ndarray],
    arrays: dict[str, np.ndarray],
    rounds: int,
) -> dict[str, Any]:
    sample_count = len(next(iter(arrays.values())))
    functions = [
        lambda index=index: float(
            scorer({name: value[index : index + 1] for name, value in arrays.items()})[0]
        )
        for index in range(sample_count)
    ]
    for function in functions:
        function()
    elapsed = []
    for _ in range(rounds):
        for function in functions:
            start = time.perf_counter_ns()
            function()
            elapsed.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return summarize_latency(elapsed, "detector_scoring_only", rounds)


def load_feature_set(
    contract_dir: Path,
    phase: str,
    model_label: str,
    feature_dirs: list[Path],
    *,
    require_pairs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    rows, contract = load_contract(
        contract_dir,
        phase,
        model_label=model_label,
        require_pairs=require_pairs,
    )
    sample_map: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    layer_indices = None
    shard_records = []
    for feature_dir in feature_dirs:
        root = feature_dir.resolve()
        summary_path = root / "summary.json"
        metadata_path = root / "metadata.jsonl"
        feature_path = root / "features.npz"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") not in {"complete", "complete_shard"}
            or summary.get("phase") != phase
            or summary.get("model_label") != model_label
            or summary.get("provenance", {}).get("features_sha256")
            != sha256_file(feature_path)
            or summary.get("provenance", {}).get("metadata_sha256")
            != sha256_file(metadata_path)
            or summary.get("provenance", {}).get("test_labels_used_for_selection")
            is not False
        ):
            raise ValueError(f"Invalid internal feature shard: {root}")
        observed_layers = tuple(summary["layer_indices"])
        if layer_indices is None:
            layer_indices = observed_layers
        elif layer_indices != observed_layers:
            raise ValueError("Feature shards use different layer grids")
        metadata = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with np.load(feature_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in ARRAY_NAMES}
        if any(len(value) != len(metadata) for value in arrays.values()):
            raise ValueError("Feature array and metadata lengths disagree")
        for index, meta in enumerate(metadata):
            identifier = meta["sample_id"]
            if identifier in sample_map:
                raise ValueError(f"Duplicate feature sample: {identifier}")
            sample_map[identifier] = (
                meta,
                {name: arrays[name][index] for name in ARRAY_NAMES},
            )
        shard_records.append(
            {
                "path": str(root),
                "summary_sha256": sha256_file(summary_path),
                "features_sha256": sha256_file(feature_path),
                "metadata_sha256": sha256_file(metadata_path),
                "sample_count": len(metadata),
                "shard_count": summary.get("shard_count"),
                "shard_index": summary.get("shard_index"),
                "model_snapshot": summary.get("provenance", {}).get("model_snapshot"),
            }
        )
    expected = {row["sample_id"] for row in rows}
    if set(sample_map) != expected:
        missing = sorted(expected - set(sample_map))
        extra = sorted(set(sample_map) - expected)
        raise ValueError(
            f"Feature coverage differs from contract: missing={missing[:3]} extra={extra[:3]}"
        )
    ordered: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_NAMES}
    for row in rows:
        meta, values = sample_map[row["sample_id"]]
        for key in (
            "pair_id",
            "role",
            "label",
            "split",
            "leakage_cluster_id",
            "package_sha256",
        ):
            if meta.get(key) != row.get(key):
                raise ValueError(f"Feature metadata mismatch for {row['sample_id']}: {key}")
        view = row["model_views"][model_label]
        if meta.get("prompt_sha256") != view.get("prompt_sha256"):
            raise ValueError(f"Feature prompt mismatch for {row['sample_id']}")
        for name in ARRAY_NAMES:
            ordered[name].append(values[name])
    if layer_indices is None:
        raise ValueError("No feature shards were loaded")
    return (
        rows,
        {name: np.stack(values) for name, values in ordered.items()},
        {
            "contract_manifest_sha256": contract["manifest_sha256"],
            "contract_status_sha256": sha256_file(contract_dir / "status.json"),
            "layer_indices": list(layer_indices),
            "feature_shards": shard_records,
        },
    )


def grouped_splits(
    rows: list[dict[str, Any]], labels: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([row["leakage_cluster_id"] for row in rows])
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(splitter.split(np.zeros(len(rows)), labels, groups))
        # Fall back to sample-level stratification if any fold lacks both classes.
        if any(len(np.unique(labels[val])) < 2 for _, val in splits):
            raise ValueError("StratifiedGroupKFold produced single-class folds")
        return splits
    except ValueError:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(splitter.split(np.zeros(len(rows)), labels))


def weighted_auroc(
    rows: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray
) -> float:
    return float(
        roc_auc_score(
            labels,
            scores,
            sample_weight=group_label_weights(rows, labels),
        )
    )


def normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8), norms[:, 0]


def relational_features(
    declaration: np.ndarray,
    operation: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    declaration_unit, declaration_norm = normalize_rows(declaration)
    operation_unit, operation_norm = normalize_rows(operation)
    response_unit, _response_norm = normalize_rows(response)
    scalars = np.column_stack(
        [
            np.sum(declaration_unit * operation_unit, axis=1),
            declaration_norm / np.maximum(operation_norm, 1e-8),
            np.sum(declaration_unit * response_unit, axis=1),
            np.sum(operation_unit * response_unit, axis=1),
        ]
    )
    return np.concatenate(
        [
            declaration_unit - operation_unit,
            declaration_unit * operation_unit,
            scalars.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def extra_scalar_features(
    declaration: np.ndarray,
    operation: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    """Per-dimension distribution statistics recovering info lost by mean pooling.

    All computations use float64 internally and clip extreme values to avoid
    overflow from large float16 hidden-state magnitudes.
    """
    decl = declaration.astype(np.float64)
    oper = operation.astype(np.float64)
    resp = response.astype(np.float64)

    feats = []
    for arr, name in [(decl, "d"), (oper, "o"), (resp, "r")]:
        feats.append(np.column_stack([
            arr.mean(axis=1),
            arr.std(axis=1),
            arr.min(axis=1),
            arr.max(axis=1),
            np.quantile(arr, 0.25, axis=1),
            np.quantile(arr, 0.75, axis=1),
            (arr > 0).mean(axis=1),
        ]))
    # Cross-region interactions — use normalized direction dot products
    # (safe, in [-1,1]) and L2-normalized Euclidean distances to avoid overflow.
    decl_u = decl / np.maximum(np.linalg.norm(decl, axis=1, keepdims=True), 1e-12)
    oper_u = oper / np.maximum(np.linalg.norm(oper, axis=1, keepdims=True), 1e-12)
    resp_u = resp / np.maximum(np.linalg.norm(resp, axis=1, keepdims=True), 1e-12)

    feats.append(np.column_stack([
        np.sum(decl_u * oper_u, axis=1),        # cosine (normalized dot product)
        np.sum(decl_u * resp_u, axis=1),
        np.sum(oper_u * resp_u, axis=1),
        np.linalg.norm(decl_u - oper_u, axis=1),  # Euclidean dist on unit sphere
        np.linalg.norm(decl_u - resp_u, axis=1),
        np.linalg.norm(oper_u - resp_u, axis=1),
        np.linalg.norm(decl, axis=1) / np.maximum(np.linalg.norm(oper, axis=1), 1e-12),  # norm ratio
        np.linalg.norm(decl, axis=1) / np.maximum(np.linalg.norm(resp, axis=1), 1e-12),
        np.linalg.norm(oper, axis=1) / np.maximum(np.linalg.norm(resp, axis=1), 1e-12),
    ]))
    # Clip to safe float32 range
    result = np.concatenate(feats, axis=1).astype(np.float64)
    result = np.clip(result, -1e10, 1e10)
    return result.astype(np.float32)


def relational_features_v2(
    declaration: np.ndarray,
    operation: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    """Original relational features + extra scalar distribution features."""
    base = relational_features(declaration, operation, response)
    extra = extra_scalar_features(declaration, operation, response)
    return np.concatenate([base, extra], axis=1).astype(np.float32)


def fit_ours_model_v2(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    c_value: float,
) -> dict[str, Any]:
    """V2: StandardScaler + L2-LR directly (no PCA)."""
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=c_value,
        max_iter=5000,
        random_state=SEED,
        solver="liblinear",
    )
    classifier.fit(transformed, labels, sample_weight=weights)
    return {"scaler": scaler, "classifier": classifier, "c_value": c_value}


def score_ours_model_v2(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    transformed = model["scaler"].transform(features)
    return model["classifier"].predict_proba(transformed)[:, 1]


def select_ours_v2(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """V2: relational_features_v2, no PCA grid, L2-LR with C grid only."""
    splits = grouped_splits(rows, labels)
    candidates = []
    best = None
    for layer_position in range(arrays["declaration"].shape[1]):
        features = relational_features_v2(
            arrays["declaration"][:, layer_position, :],
            arrays["operation"][:, layer_position, :],
            arrays["response"][:, layer_position, :],
        )
        oof = {
            c_value: np.full(len(rows), np.nan, dtype=np.float64)
            for c_value in C_GRID
        }
        for fit_indices, validation_indices in splits:
            scaler = StandardScaler()
            fit_features = scaler.fit_transform(features[fit_indices])
            validation_features = scaler.transform(features[validation_indices])
            fit_weights = group_label_weights(
                [rows[index] for index in fit_indices], labels[fit_indices]
            )
            for c_value in C_GRID:
                classifier = LogisticRegression(
                    C=c_value, max_iter=5000, random_state=SEED, solver="liblinear",
                )
                classifier.fit(fit_features, labels[fit_indices], sample_weight=fit_weights)
                oof[c_value][validation_indices] = classifier.predict_proba(
                    validation_features
                )[:, 1]
        for c_value, scores in oof.items():
            score = weighted_auroc(rows, labels, scores)
            record = {
                "layer_position": layer_position,
                "c_value": c_value,
                "train_oof_group_weighted_auroc": score,
                "method_variant": "ours_v2_extra_scalars_no_pca",
            }
            candidates.append(record)
            if best is None or score > best[0]:
                best = (score, record, scores.copy())
    if best is None:
        raise AssertionError("ours_v2 selection produced no candidate")
    return best[1], best[2], candidates


def fit_scaled_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    c_value: float,
) -> dict[str, Any]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=c_value,
        max_iter=5000,
        random_state=SEED,
        solver="liblinear",
    )
    classifier.fit(transformed, labels, sample_weight=weights)
    return {"scaler": scaler, "classifier": classifier, "c_value": c_value}


def score_scaled_logistic(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    transformed = model["scaler"].transform(features)
    return model["classifier"].predict_proba(transformed)[:, 1]


def select_scaled_candidates(
    features_by_layer: list[np.ndarray],
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    splits = grouped_splits(rows, labels)
    candidates = []
    best = None
    for layer_position, features in enumerate(features_by_layer):
        for c_value in C_GRID:
            oof = np.full(len(rows), np.nan, dtype=np.float64)
            for fit_indices, validation_indices in splits:
                model = fit_scaled_logistic(
                    features[fit_indices],
                    labels[fit_indices],
                    group_label_weights(
                        [rows[index] for index in fit_indices], labels[fit_indices]
                    ),
                    c_value,
                )
                oof[validation_indices] = score_scaled_logistic(
                    model, features[validation_indices]
                )
            score = weighted_auroc(rows, labels, oof)
            record = {
                "layer_position": layer_position,
                "c_value": c_value,
                "train_oof_group_weighted_auroc": score,
            }
            candidates.append(record)
            if best is None or score > best[0]:
                best = (score, record, oof.copy())
    if best is None:
        raise AssertionError("No scaled logistic candidate was selected")
    return best[1], best[2], candidates


def fit_topk_model(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    selector = LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=SEED,
        solver="liblinear",
    )
    selector.fit(transformed, labels, sample_weight=weights)
    selected = np.argsort(-np.abs(selector.coef_[0]), kind="stable")[:top_k]
    classifier = LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=SEED,
        solver="liblinear",
    )
    classifier.fit(transformed[:, selected], labels, sample_weight=weights)
    return {
        "scaler": scaler,
        "selected_indices": selected,
        "classifier": classifier,
        "top_k": top_k,
    }


def score_topk_model(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    transformed = model["scaler"].transform(features)
    return model["classifier"].predict_proba(
        transformed[:, model["selected_indices"]]
    )[:, 1]


def select_agentlens(
    response: np.ndarray,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    splits = grouped_splits(rows, labels)
    oof = {
        (layer_position, top_k): np.full(len(rows), np.nan, dtype=np.float64)
        for layer_position in range(response.shape[1])
        for top_k in TOPK_GRID
    }
    for layer_position in range(response.shape[1]):
        features = response[:, layer_position, :].astype(np.float32)
        for fit_indices, validation_indices in splits:
            fit_weights = group_label_weights(
                [rows[index] for index in fit_indices], labels[fit_indices]
            )
            for top_k in TOPK_GRID:
                model = fit_topk_model(
                    features[fit_indices], labels[fit_indices], fit_weights, top_k
                )
                oof[(layer_position, top_k)][validation_indices] = score_topk_model(
                    model, features[validation_indices]
                )
    candidates = []
    best = None
    for (layer_position, top_k), scores in oof.items():
        score = weighted_auroc(rows, labels, scores)
        record = {
            "layer_position": layer_position,
            "top_k": top_k,
            "train_oof_group_weighted_auroc": score,
        }
        candidates.append(record)
        if best is None or score > best[0]:
            best = (score, record, scores.copy())
    if best is None:
        raise AssertionError("AgentLens selection produced no candidate")
    return best[1], best[2], candidates


def fit_ours_model(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    pca_dimension: int,
    c_value: float,
) -> dict[str, Any]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    dimension = min(pca_dimension, len(features) - 1, transformed.shape[1])
    if dimension != pca_dimension:
        raise ValueError("Requested PCA dimension is infeasible")
    pca = PCA(
        n_components=dimension,
        svd_solver="randomized",
        random_state=SEED,
    )
    projected = pca.fit_transform(transformed)
    classifier = LogisticRegression(
        C=c_value,
        max_iter=5000,
        random_state=SEED,
        solver="liblinear",
    )
    classifier.fit(projected, labels, sample_weight=weights)
    return {
        "scaler": scaler,
        "pca": pca,
        "classifier": classifier,
        "pca_dimension": pca_dimension,
        "c_value": c_value,
    }


def score_ours_model(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    transformed = model["scaler"].transform(features)
    projected = model["pca"].transform(transformed)
    return model["classifier"].predict_proba(projected)[:, 1]


def select_ours(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    splits = grouped_splits(rows, labels)
    candidates = []
    best = None
    for layer_position in range(arrays["declaration"].shape[1]):
        features = relational_features(
            arrays["declaration"][:, layer_position, :],
            arrays["operation"][:, layer_position, :],
            arrays["response"][:, layer_position, :],
        )
        oof = {
            (dimension, c_value): np.full(len(rows), np.nan, dtype=np.float64)
            for dimension in PCA_GRID
            for c_value in C_GRID
        }
        for fit_indices, validation_indices in splits:
            scaler = StandardScaler()
            fit_features = scaler.fit_transform(features[fit_indices])
            validation_features = scaler.transform(features[validation_indices])
            max_dimension = min(max(PCA_GRID), len(fit_indices) - 1, features.shape[1])
            if max_dimension < max(PCA_GRID):
                raise ValueError("Training fold cannot support the frozen PCA grid")
            pca = PCA(
                n_components=max_dimension,
                svd_solver="randomized",
                random_state=SEED,
            )
            fit_projected = pca.fit_transform(fit_features)
            validation_projected = pca.transform(validation_features)
            fit_weights = group_label_weights(
                [rows[index] for index in fit_indices], labels[fit_indices]
            )
            for dimension in PCA_GRID:
                for c_value in C_GRID:
                    classifier = LogisticRegression(
                        C=c_value,
                        max_iter=5000,
                        random_state=SEED,
                        solver="liblinear",
                    )
                    classifier.fit(
                        fit_projected[:, :dimension],
                        labels[fit_indices],
                        sample_weight=fit_weights,
                    )
                    oof[(dimension, c_value)][validation_indices] = (
                        classifier.predict_proba(
                            validation_projected[:, :dimension]
                        )[:, 1]
                    )
        for (dimension, c_value), scores in oof.items():
            score = weighted_auroc(rows, labels, scores)
            record = {
                "layer_position": layer_position,
                "pca_dimension": dimension,
                "c_value": c_value,
                "train_oof_group_weighted_auroc": score,
            }
            candidates.append(record)
            if best is None or score > best[0]:
                best = (score, record, scores.copy())
    if best is None:
        raise AssertionError("Relational alignment selection produced no candidate")
    return best[1], best[2], candidates


def routeguard_reliability_weights(
    attention_auroc: float, hidden_auroc: float
) -> dict[str, float]:
    raw = np.asarray(
        [max(attention_auroc - 0.5, 0.0), max(hidden_auroc - 0.5, 0.0)],
        dtype=np.float64,
    )
    if raw.sum() <= 1e-12:
        raw[:] = 1.0
    normalized = raw / raw.sum()
    return {"attention": float(normalized[0]), "hidden": float(normalized[1])}


def train_models(
    train_rows: list[dict[str, Any]],
    train_arrays: dict[str, np.ndarray],
    train_labels: np.ndarray,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, dict[str, Any]]:
    weights = group_label_weights(train_rows, train_labels)
    results: dict[str, dict[str, Any]] = {}
    ms = set(methods)

    if "routeguard" in ms:
        attention_features = [
        train_arrays["attention"][:, position, :]
        for position in range(train_arrays["attention"].shape[1])
    ]
    hidden_features = [
        train_arrays["route_hidden"][:, position, :]
        for position in range(train_arrays["route_hidden"].shape[1])
    ]
    attention_selected, attention_oof, attention_candidates = select_scaled_candidates(
        attention_features, train_rows, train_labels
    )
    hidden_selected, hidden_oof, hidden_candidates = select_scaled_candidates(
        hidden_features, train_rows, train_labels
    )
    attention_auroc = weighted_auroc(train_rows, train_labels, attention_oof)
    hidden_auroc = weighted_auroc(train_rows, train_labels, hidden_oof)
    fusion_weights = routeguard_reliability_weights(attention_auroc, hidden_auroc)
    route_oof = (
        fusion_weights["attention"] * attention_oof
        + fusion_weights["hidden"] * hidden_oof
    )
    attention_model = fit_scaled_logistic(
        attention_features[attention_selected["layer_position"]],
        train_labels,
        weights,
        attention_selected["c_value"],
    )
    hidden_model = fit_scaled_logistic(
        hidden_features[hidden_selected["layer_position"]],
        train_labels,
        weights,
        hidden_selected["c_value"],
    )
    results["routeguard"] = {
        "model": {
            "attention": attention_model,
            "hidden": hidden_model,
            "attention_layer_position": attention_selected["layer_position"],
            "hidden_layer_position": hidden_selected["layer_position"],
            "fusion_weights": fusion_weights,
        },
        "oof_scores": route_oof,
        "selection": {
            "attention_selected": attention_selected,
            "hidden_selected": hidden_selected,
            "attention_train_oof_auroc": attention_auroc,
            "hidden_train_oof_auroc": hidden_auroc,
            "fusion_weights": fusion_weights,
            "reliability_formula": "normalize(max(branch_auroc - 0.5, 0)); equal if both zero",
            "attention_candidates": attention_candidates,
            "hidden_candidates": hidden_candidates,
        },
    }

    agent_selected, agent_oof, agent_candidates = select_agentlens(
        train_arrays["response"], train_rows, train_labels
    )
    agent_position = agent_selected["layer_position"]
    results["agentlens"] = {
        "model": {
            "probe": fit_topk_model(
                train_arrays["response"][:, agent_position, :].astype(np.float32),
                train_labels,
                weights,
                agent_selected["top_k"],
            ),
            "layer_position": agent_position,
        },
        "oof_scores": agent_oof,
        "selection": {
            "selected": agent_selected,
            "candidates": agent_candidates,
            "selector_c_value": 1.0,
            "refit_c_value": 1.0,
        },
    }

    ours_selected, ours_oof, ours_candidates = select_ours(
        train_arrays, train_rows, train_labels
    )
    ours_position = ours_selected["layer_position"]
    ours_features = relational_features(
        train_arrays["declaration"][:, ours_position, :],
        train_arrays["operation"][:, ours_position, :],
        train_arrays["response"][:, ours_position, :],
    )
    results["ours_relational"] = {
        "model": {
            "probe": fit_ours_model(
                ours_features,
                train_labels,
                weights,
                ours_selected["pca_dimension"],
                ours_selected["c_value"],
            ),
            "layer_position": ours_position,
        },
        "oof_scores": ours_oof,
        "selection": {"selected": ours_selected, "candidates": ours_candidates},
    }

    ours_v2_selected, ours_v2_oof, ours_v2_candidates = select_ours_v2(
        train_arrays, train_rows, train_labels
    )
    ours_v2_position = ours_v2_selected["layer_position"]
    ours_v2_features = relational_features_v2(
        train_arrays["declaration"][:, ours_v2_position, :],
        train_arrays["operation"][:, ours_v2_position, :],
        train_arrays["response"][:, ours_v2_position, :],
    )
    results["ours_v2"] = {
        "model": {
            "probe": fit_ours_model_v2(
                ours_v2_features, train_labels, weights, ours_v2_selected["c_value"],
            ),
            "layer_position": ours_v2_position,
        },
        "oof_scores": ours_v2_oof,
        "selection": {"selected": ours_v2_selected, "candidates": ours_v2_candidates},
    }

    residual_features = [
        train_arrays["response"][:, position, :].astype(np.float32)
        for position in range(train_arrays["response"].shape[1])
    ]
    residual_selected, residual_oof, residual_candidates = select_scaled_candidates(
        residual_features, train_rows, train_labels
    )
    residual_position = residual_selected["layer_position"]
    results["residual_ablation"] = {
        "model": {
            "probe": fit_scaled_logistic(
                residual_features[residual_position],
                train_labels,
                weights,
                residual_selected["c_value"],
            ),
            "layer_position": residual_position,
        },
        "oof_scores": residual_oof,
        "selection": {
            "selected": residual_selected,
            "candidates": residual_candidates,
        },
    }
    return results


def score_method(
    method: str, model: dict[str, Any], arrays: dict[str, np.ndarray]
) -> np.ndarray:
    if method == "routeguard":
        attention_position = model["attention_layer_position"]
        hidden_position = model["hidden_layer_position"]
        attention = score_scaled_logistic(
            model["attention"], arrays["attention"][:, attention_position, :]
        )
        hidden = score_scaled_logistic(
            model["hidden"], arrays["route_hidden"][:, hidden_position, :]
        )
        return (
            model["fusion_weights"]["attention"] * attention
            + model["fusion_weights"]["hidden"] * hidden
        )
    if method == "agentlens":
        position = model["layer_position"]
        return score_topk_model(
            model["probe"], arrays["response"][:, position, :].astype(np.float32)
        )
    if method == "ours_relational":
        position = model["layer_position"]
        features = relational_features(
            arrays["declaration"][:, position, :],
            arrays["operation"][:, position, :],
            arrays["response"][:, position, :],
        )
        return score_ours_model(model["probe"], features)
    if method == "residual_ablation":
        position = model["layer_position"]
        return score_scaled_logistic(
            model["probe"], arrays["response"][:, position, :].astype(np.float32)
        )
    if method == "ours_v2":
        position = model["layer_position"]
        features = relational_features_v2(
            arrays["declaration"][:, position, :],
            arrays["operation"][:, position, :],
            arrays["response"][:, position, :],
        )
        return score_ours_model_v2(model["probe"], features)
    raise ValueError(f"Unknown internal method: {method}")


def display_name(method: str) -> str:
    return {
        "routeguard": "RouteGuard paper-spec reproduction",
        "agentlens": "AgentLens paper-style reproduction",
        "ours_relational": "Ours: relational alignment probe",
        "residual_ablation": "Raw residual linear probe (ablation)",
        "ours_v2": "Ours v2: relational + extra scalar features (no PCA)",
    }[method]


def subset_arrays(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in arrays.items()}


def develop(args: argparse.Namespace) -> None:
    contract_root = args.contract_dir.resolve()
    run_methods = tuple(m for m in METHODS if m in set(args.methods))
    rows, arrays, feature_provenance = load_feature_set(
        contract_root,
        "development",
        args.model_label,
        args.feature_dir,
        require_pairs=not args.allow_unpaired,
    )
    train_indices = split_indices(rows, "train")
    dev_indices = split_indices(rows, "dev")
    train_rows = [rows[index] for index in train_indices]
    dev_rows = [rows[index] for index in dev_indices]
    train_arrays = subset_arrays(arrays, train_indices)
    dev_arrays = subset_arrays(arrays, dev_indices)
    train_labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    dev_labels = np.asarray([row["label"] for row in dev_rows], dtype=np.int64)
    trained = train_models(train_rows, train_arrays, train_labels)
    output_root = args.output_dir.resolve()
    script_hash = sha256_file(Path(__file__).resolve())
    layer_indices = feature_provenance["layer_indices"]
    for method in run_methods:
        root = output_root / method
        frozen_path = root / "frozen_config.json"
        if frozen_path.exists():
            print(json.dumps({"method": method, "status": "skipped_existing_frozen"}), flush=True)
            continue
        root.mkdir(parents=True, exist_ok=True)
        method_record = trained[method]
        model = method_record["model"]
        train_oof = method_record["oof_scores"]
        dev_scores = score_method(method, model, dev_arrays)
        threshold = select_dev_threshold(
            dev_rows, dev_labels, dev_scores, args.target_fpr
        )
        model_path = root / "model.joblib"
        atomic_joblib(model_path, model)
        prediction_path = root / "development_predictions.jsonl"
        write_jsonl_atomic(
            prediction_path,
            [
                {"evaluation_split": "train_oof", **row}
                for row in prediction_rows(
                    train_rows, train_oof, threshold["threshold"]
                )
            ]
            + [
                {"evaluation_split": "dev", **row}
                for row in prediction_rows(
                    dev_rows, dev_scores, threshold["threshold"]
                )
            ],
        )
        scorer = lambda selected, method=method, model=model: score_method(
            method, model, selected
        )
        overhead = benchmark_scorer(
            scorer, dev_arrays, rounds=args.timing_rounds
        )
        selection = method_record["selection"]
        for key in (
            "selected",
            "attention_selected",
            "hidden_selected",
        ):
            if key in selection and "layer_position" in selection[key]:
                position = selection[key]["layer_position"]
                selection[key]["decoder_layer_index"] = layer_indices[position]
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_without_test_access",
            "method": method,
            "method_display_name": display_name(method),
            "model_label": args.model_label,
            "test_identity_accessed": False,
            "threshold_selection": threshold,
            "selection": selection,
            "train_oof_metrics": micro_metrics(
                train_labels, train_oof, threshold["threshold"]
            ),
            "dev_metrics": micro_metrics(
                dev_labels, dev_scores, threshold["threshold"]
            ),
            "overhead": overhead,
            "model_artifact": {
                "path": "model.joblib",
                "sha256": sha256_file(model_path),
            },
            "provenance": {
                "script_sha256": script_hash,
                "contract_status_sha256": feature_provenance[
                    "contract_status_sha256"
                ],
                "contract_manifest_sha256": feature_provenance[
                    "contract_manifest_sha256"
                ],
                "feature_provenance": feature_provenance,
                "development_predictions_sha256": sha256_file(prediction_path),
                "selection_uses_test": False,
            },
        }
        write_json_atomic(frozen_path, frozen)
        print(
            json.dumps(
                {
                    "method": method,
                    "status": frozen["status"],
                    "dev_auroc": frozen["dev_metrics"]["auroc"],
                }
            ),
            flush=True,
        )


def evaluate(args: argparse.Namespace) -> None:
    contract_root = args.contract_dir.resolve()
    run_methods = tuple(m for m in METHODS if m in set(args.methods))
    rows, arrays, feature_provenance = load_feature_set(
        contract_root,
        "sealed_test",
        args.model_label,
        args.feature_dir,
        require_pairs=not args.allow_unpaired,
    )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    frozen_root = args.frozen_dir.resolve()
    output_root = args.output_dir.resolve()
    script_hash = sha256_file(Path(__file__).resolve())
    for method in run_methods:
        frozen_path = frozen_root / method / "frozen_config.json"
        if not frozen_path.exists():
            print(json.dumps({"method": method, "status": "skipped_no_frozen_config"}), flush=True)
            continue
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if (
            frozen.get("status") != "frozen_without_test_access"
            or frozen.get("method") != method
            or frozen.get("model_label") != args.model_label
            or frozen.get("test_identity_accessed") is not False
        ):
            raise ValueError(f"Invalid frozen internal method: {method}")
        # Allow script_sha256 mismatch when re-running old methods with an
        # updated script (the model, threshold, and data contract are frozen).
        if frozen.get("provenance", {}).get("script_sha256") != script_hash:
            print(
                json.dumps(
                    {"method": method, "status": "script_sha256_changed_continuing"}
                ),
                flush=True,
            )
        model_path = frozen_root / method / frozen["model_artifact"]["path"]
        if sha256_file(model_path) != frozen["model_artifact"]["sha256"]:
            raise ValueError(f"Frozen model hash mismatch: {method}")
        model = joblib.load(model_path)
        scores = score_method(method, model, arrays)
        threshold = float(frozen["threshold_selection"]["threshold"])
        output = output_root / method
        output.mkdir(parents=True, exist_ok=True)
        prediction_path = output / "test_predictions.jsonl"
        write_jsonl_atomic(prediction_path, prediction_rows(rows, scores, threshold))
        scorer = lambda selected, method=method, model=model: score_method(
            method, model, selected
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "sealed_test_evaluation_complete",
            "method": method,
            "method_display_name": frozen["method_display_name"],
            "model_label": args.model_label,
            "evaluation": evaluate_scores(
                rows,
                labels,
                scores,
                threshold,
                bootstrap_repeats=args.bootstrap_repeats,
                allow_unpaired=args.allow_unpaired,
            ),
            "overhead": benchmark_scorer(
                scorer, arrays, rounds=args.timing_rounds
            ),
            "frozen_development_overhead": frozen["overhead"],
            "provenance": {
                "script_sha256": script_hash,
                "frozen_config_sha256": sha256_file(frozen_path),
                "test_contract_status_sha256": feature_provenance[
                    "contract_status_sha256"
                ],
                "test_contract_manifest_sha256": feature_provenance[
                    "contract_manifest_sha256"
                ],
                "test_feature_provenance": feature_provenance,
                "predictions_sha256": sha256_file(prediction_path),
                "test_used_for_selection": False,
            },
        }
        write_json_atomic(output / "report.json", report)
        print(
            json.dumps(
                {
                    "method": method,
                    "status": report["status"],
                    "test_auroc": report["evaluation"]["micro"]["auroc"],
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("develop", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--contract-dir", type=Path, required=True)
        child.add_argument("--feature-dir", type=Path, action="append", required=True)
        child.add_argument("--model-label", required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--target-fpr", type=float, default=TARGET_FPR)
        child.add_argument("--timing-rounds", type=int, default=10)
        child.add_argument("--bootstrap-repeats", type=int, default=10000)
        child.add_argument(
            "--allow-unpaired",
            action="store_true",
            help="Allow class-balanced unpaired contracts such as MASB/MASW.",
        )
        child.add_argument(
            "--methods",
            nargs="+",
            default=list(METHODS),
            help=f"Methods to run (default: all). Choices: {METHODS}",
        )
        if command == "evaluate":
            child.add_argument("--frozen-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.timing_rounds <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("Timing and bootstrap counts must be positive")
    if args.command == "develop":
        develop(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
