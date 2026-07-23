#!/usr/bin/env python3
"""Develop and evaluate static Foundation-v3 baselines under one contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from experiments.baseline_suite import STATIC_PATTERNS  # noqa: E402
from src.foundation_v3_experiment import (  # noqa: E402
    TARGET_FPR,
    evaluate_scores,
    group_label_weights,
    load_contract,
    micro_metrics,
    prediction_rows,
    select_dev_threshold,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = "skillprobe-foundation-static-baselines-v1.0"
METHODS = ("static_regex", "tfidf_lr", "skillsieve_l1")
COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), category, weight)
    for pattern, category, weight in STATIC_PATTERNS
]
SKILLSIEVE_EXTRA = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), category, weight)
    for pattern, category, weight in (
        (r"docker\s+run.*--privileged", "privileged_container", 1.0),
        (r"mount\s+--bind\s+/", "container_escape", 1.0),
        (r"pip\s+install\s+-e\s+git\+", "untrusted_dependency", 0.8),
        (r"(?:eval|exec)\s*\(.*(?:base64|decode|decompress)", "dynamic_obfuscation", 1.0),
        (r"(?:ignore|override|bypass).{0,60}(?:user|system|safety|policy)", "instruction_override", 0.9),
    )
]
FILE_BLOCK = re.compile(
    r'<skill_file path="([^"]+)">\n(.*?)\n</skill_file>', re.DOTALL
)
SUSPICIOUS_CALLS = {
    "eval",
    "exec",
    "compile",
    "system",
    "popen",
    "run",
    "call",
    "remove",
    "unlink",
    "rmtree",
    "chmod",
    "setuid",
    "connect",
    "urlopen",
    "post",
    "put",
}
NETWORK_NAMES = {"requests", "urllib", "socket", "httpx", "aiohttp", "ftplib"}
ENV_NAMES = {"environ", "getenv", "keyring", "credential", "token", "secret"}


def load_texts(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    return [(root / row["package_file"]).read_text(encoding="utf-8") for row in rows]


def regex_score(text: str) -> float:
    weights = []
    categories = set()
    for pattern, category, weight in COMPILED_PATTERNS:
        if pattern.search(text):
            weights.append(float(weight))
            categories.add(category)
    return float(sum(weights) + 0.25 * len(categories))


def string_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    return -sum(
        (count / len(value)) * math.log2(count / len(value))
        for count in counts.values()
    )


def python_ast_features(text: str) -> dict[str, int]:
    result = {
        "dynamic_calls": 0,
        "network_names": 0,
        "environment_names": 0,
        "high_entropy_strings": 0,
        "parse_errors": 0,
    }
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        result["parse_errors"] = 1
        return result
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            result["dynamic_calls"] += name.lower() in SUSPICIOUS_CALLS
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = (
                node.id if isinstance(node, ast.Name) else node.attr
            ).lower()
            result["network_names"] += name in NETWORK_NAMES
            result["environment_names"] += name in ENV_NAMES
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            result["high_entropy_strings"] += (
                len(value) >= 32 and string_entropy(value) >= 4.3
            )
    return result


def skillsieve_l1_score(text: str, row: dict[str, Any]) -> float:
    score = regex_score(text)
    categories = set()
    for pattern, category, weight in SKILLSIEVE_EXTRA:
        if pattern.search(text):
            score += float(weight)
            categories.add(category)
    text_files = FILE_BLOCK.findall(text)
    ast_totals = Counter()
    script_count = 0
    for relative, content in text_files:
        suffix = Path(relative).suffix.lower()
        if suffix in {".py", ".pyw"}:
            script_count += 1
            ast_totals.update(python_ast_features(content))
        elif suffix in {".js", ".ts", ".sh", ".bash", ".zsh"}:
            script_count += 1
    score += 0.8 * ast_totals["dynamic_calls"]
    score += 0.5 * ast_totals["network_names"]
    score += 0.4 * ast_totals["environment_names"]
    score += 0.25 * ast_totals["high_entropy_strings"]
    score += 0.1 * ast_totals["parse_errors"]
    score += 0.15 * max(script_count - 1, 0)
    score += 0.2 * max(len(text_files) - 3, 0)
    score += 0.3 * int(row.get("package_structure", {}).get("binary_file_count", 0) > 0)
    score += 0.2 * len(categories)
    return float(score)


def vectorizer(view: str):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        lowercase=True,
        min_df=2,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
        lowercase=True,
        min_df=2,
    )
    if view == "word":
        return word
    if view == "char":
        return char
    if view == "hybrid":
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(view)


def tfidf_pipeline(view: str, c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", vectorizer(view)),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=5000,
                    random_state=20260718,
                    solver="liblinear",
                ),
            ),
        ]
    )


def fit_tfidf(
    train_rows: list[dict[str, Any]],
    train_texts: list[str],
    train_labels: np.ndarray,
) -> tuple[Pipeline, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    groups = np.asarray([row["leakage_cluster_id"] for row in train_rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260718)
    weights = group_label_weights(train_rows, train_labels)
    selection = []
    best = None
    for view in ("word", "char", "hybrid"):
        for c_value in (0.1, 1.0, 10.0):
            oof = np.full(len(train_rows), np.nan, dtype=np.float64)
            for fit_indices, validation_indices in splitter.split(
                train_texts, train_labels, groups
            ):
                model = tfidf_pipeline(view, c_value)
                fit_weights = group_label_weights(
                    [train_rows[index] for index in fit_indices],
                    train_labels[fit_indices],
                )
                model.fit(
                    [train_texts[index] for index in fit_indices],
                    train_labels[fit_indices],
                    classifier__sample_weight=fit_weights,
                )
                oof[validation_indices] = model.predict_proba(
                    [train_texts[index] for index in validation_indices]
                )[:, 1]
            if not np.isfinite(oof).all():
                raise RuntimeError("TF-IDF grouped OOF predictions are incomplete")
            auroc = float(roc_auc_score(train_labels, oof, sample_weight=weights))
            record = {
                "view": view,
                "c_value": c_value,
                "train_oof_group_weighted_auroc": auroc,
            }
            selection.append(record)
            key = (auroc, -len(view), -c_value)
            if best is None or key > best[0]:
                best = (key, view, c_value, oof)
    if best is None:
        raise AssertionError("TF-IDF selection produced no candidate")
    _, view, c_value, oof = best
    model = tfidf_pipeline(view, c_value)
    model.fit(
        train_texts,
        train_labels,
        classifier__sample_weight=weights,
    )
    return model, oof, selection, {"view": view, "c_value": c_value}


def benchmark_functions(functions: list[Callable[[], float]], rounds: int = 10) -> dict[str, Any]:
    for function in functions:
        function()
    elapsed = []
    for _ in range(rounds):
        for function in functions:
            start = time.perf_counter_ns()
            function()
            elapsed.append((time.perf_counter_ns() - start) / 1_000_000.0)
    values = np.asarray(elapsed, dtype=np.float64)
    cpu_model = ""
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    cpu_model = cpu_model or platform.processor().strip() or platform.machine()
    return {
        "status": "complete",
        "scope": "detector_scoring_only",
        "measurement_count": len(values),
        "warmup_rounds": 1,
        "measurement_rounds": rounds,
        "mean_ms_per_sample": float(values.mean()),
        "p50_ms_per_sample": float(np.quantile(values, 0.50)),
        "p95_ms_per_sample": float(np.quantile(values, 0.95)),
        "model_loading_included": False,
        "prompt_construction_included": False,
        "tokenization_included": True,
        "serialization_included": False,
        "execution_device": "cpu",
        "cpu_model": cpu_model,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def method_paths(output_root: Path, method: str) -> tuple[Path, Path]:
    root = output_root / method
    return root, root / "frozen_config.json"


def develop(args: argparse.Namespace) -> None:
    contract_root = args.contract_dir.resolve()
    rows, contract = load_contract(contract_root, "development")
    train_rows = split_rows(rows, "train")
    dev_rows = split_rows(rows, "dev")
    train_texts = load_texts(contract_root, train_rows)
    dev_texts = load_texts(contract_root, dev_rows)
    train_labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    dev_labels = np.asarray([row["label"] for row in dev_rows], dtype=np.int64)
    output_root = args.output_dir.resolve()
    script_hash = sha256_file(Path(__file__).resolve())

    method_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "static_regex": (
            np.asarray([regex_score(text) for text in train_texts]),
            np.asarray([regex_score(text) for text in dev_texts]),
        ),
        "skillsieve_l1": (
            np.asarray(
                [skillsieve_l1_score(text, row) for text, row in zip(train_texts, train_rows)]
            ),
            np.asarray(
                [skillsieve_l1_score(text, row) for text, row in zip(dev_texts, dev_rows)]
            ),
        ),
    }
    tfidf_model, tfidf_oof, selection, selected = fit_tfidf(
        train_rows, train_texts, train_labels
    )
    method_scores["tfidf_lr"] = (
        tfidf_oof,
        tfidf_model.predict_proba(dev_texts)[:, 1],
    )

    for method in METHODS:
        root, frozen_path = method_paths(output_root, method)
        if frozen_path.exists():
            raise FileExistsError(f"Development method already frozen: {method}")
        root.mkdir(parents=True, exist_ok=True)
        train_scores, dev_scores = method_scores[method]
        threshold = select_dev_threshold(
            dev_rows, dev_labels, dev_scores, args.target_fpr
        )
        model_artifact = None
        if method == "tfidf_lr":
            model_path = root / "model.joblib"
            atomic_joblib(model_path, tfidf_model)
            model_artifact = {
                "path": "model.joblib",
                "sha256": sha256_file(model_path),
                "selected": selected,
                "selection": selection,
            }
            functions = [
                (lambda text=text: float(tfidf_model.predict_proba([text])[0, 1]))
                for text in dev_texts
            ]
        elif method == "static_regex":
            functions = [(lambda text=text: regex_score(text)) for text in dev_texts]
        else:
            functions = [
                (lambda text=text, row=row: skillsieve_l1_score(text, row))
                for text, row in zip(dev_texts, dev_rows)
            ]
        overhead = benchmark_functions(functions, rounds=args.timing_rounds)
        predictions_path = root / "development_predictions.jsonl"
        write_jsonl_atomic(
            predictions_path,
            [
                {"evaluation_split": "train_oof_or_fixed", **row}
                for row in prediction_rows(train_rows, train_scores, threshold["threshold"])
            ]
            + [
                {"evaluation_split": "dev", **row}
                for row in prediction_rows(dev_rows, dev_scores, threshold["threshold"])
            ],
        )
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_without_test_access",
            "method": method,
            "method_display_name": {
                "static_regex": "Static regex",
                "tfidf_lr": "TF-IDF + LR",
                "skillsieve_l1": "SkillSieve-L1 paper-spec reproduction",
            }[method],
            "test_identity_accessed": False,
            "threshold_selection": threshold,
            "train_metrics": micro_metrics(
                train_labels, train_scores, threshold["threshold"]
            ),
            "dev_metrics": micro_metrics(dev_labels, dev_scores, threshold["threshold"]),
            "overhead": overhead,
            "model_artifact": model_artifact,
            "provenance": {
                "script_sha256": script_hash,
                "contract_status_sha256": sha256_file(contract_root / "status.json"),
                "contract_manifest_sha256": contract["manifest_sha256"],
                "development_predictions_sha256": sha256_file(predictions_path),
                "selection_uses_test": False,
            },
        }
        write_json_atomic(frozen_path, frozen)
        print(json.dumps({"method": method, "status": frozen["status"]}), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    contract_root = args.contract_dir.resolve()
    rows, contract = load_contract(contract_root, "sealed_test")
    texts = load_texts(contract_root, rows)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    frozen_root = args.frozen_dir.resolve()
    output_root = args.output_dir.resolve()
    script_hash = sha256_file(Path(__file__).resolve())
    for method in METHODS:
        frozen_path = frozen_root / method / "frozen_config.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if (
            frozen.get("status") != "frozen_without_test_access"
            or frozen.get("method") != method
            or frozen.get("test_identity_accessed") is not False
            or frozen.get("provenance", {}).get("script_sha256") != script_hash
        ):
            raise ValueError(f"Invalid frozen static method: {method}")
        threshold = float(frozen["threshold_selection"]["threshold"])
        if method == "static_regex":
            scores = np.asarray([regex_score(text) for text in texts])
            functions = [(lambda text=text: regex_score(text)) for text in texts]
        elif method == "skillsieve_l1":
            scores = np.asarray(
                [skillsieve_l1_score(text, row) for text, row in zip(texts, rows)]
            )
            functions = [
                (lambda text=text, row=row: skillsieve_l1_score(text, row))
                for text, row in zip(texts, rows)
            ]
        else:
            model_path = frozen_root / method / frozen["model_artifact"]["path"]
            if sha256_file(model_path) != frozen["model_artifact"]["sha256"]:
                raise ValueError("Frozen TF-IDF model hash mismatch")
            model = joblib.load(model_path)
            scores = model.predict_proba(texts)[:, 1]
            functions = [
                (lambda text=text: float(model.predict_proba([text])[0, 1]))
                for text in texts
            ]
        output = output_root / method
        output.mkdir(parents=True, exist_ok=True)
        predictions_path = output / "test_predictions.jsonl"
        write_jsonl_atomic(predictions_path, prediction_rows(rows, scores, threshold))
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "sealed_test_evaluation_complete",
            "method": method,
            "method_display_name": frozen["method_display_name"],
            "evaluation": evaluate_scores(
                rows,
                labels,
                scores,
                threshold,
                bootstrap_repeats=args.bootstrap_repeats,
            ),
            "overhead": benchmark_functions(functions, rounds=args.timing_rounds),
            "frozen_development_overhead": frozen["overhead"],
            "provenance": {
                "script_sha256": script_hash,
                "frozen_config_sha256": sha256_file(frozen_path),
                "test_contract_status_sha256": sha256_file(contract_root / "status.json"),
                "test_contract_manifest_sha256": contract["manifest_sha256"],
                "predictions_sha256": sha256_file(predictions_path),
                "test_used_for_selection": False,
            },
        }
        write_json_atomic(output / "report.json", report)
        print(json.dumps({"method": method, "status": report["status"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("develop", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--contract-dir", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--target-fpr", type=float, default=TARGET_FPR)
        child.add_argument("--timing-rounds", type=int, default=10)
        child.add_argument("--bootstrap-repeats", type=int, default=10000)
        if command == "evaluate":
            child.add_argument("--frozen-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "develop":
        develop(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
