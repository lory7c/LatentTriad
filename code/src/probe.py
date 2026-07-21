"""
探针训练与评估.
支持线性探针 (LogisticRegression), MLP 探针, XGBoost.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional
import joblib
import os


class ProbeTrainer:
    """
    探针训练器. 默认用 L2-正则化的 LogisticRegression.

    Parameters
    ----------
    probe_type : str
        "linear" | "mlp" | "xgboost"
    cv_folds : int
        交叉验证折数
    random_state : int
    """

    def __init__(
        self,
        probe_type: str = "linear",
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.probe_type = probe_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.probes: Dict[int, object] = {}  # layer_idx → trained probe
        self.metrics: Dict[int, dict] = {}   # layer_idx → metrics dict

    def _make_probe(self) -> object:
        if self.probe_type == "linear":
            return LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.probe_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(256, 64),
                alpha=0.001,
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
            )
        else:
            raise ValueError(f"Unknown probe_type: {self.probe_type}")

    def train_layer(
        self,
        X: np.ndarray,  # [n_samples, hidden_dim]
        y: np.ndarray,  # [n_samples]
        layer_idx: int,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> dict:
        """
        在单层上训探针. 返回 CV 指标 + held-out test 指标.

        Parameters
        ----------
        X_test, y_test : optional
            如果提供, 在全部训练数据上训最终探针后在 test 上评估.
        """
        probe = self._make_probe()
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        aurocs = []
        accuracies = []
        f1s = []
        precisions = []
        recalls = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_val)
            y_prob = probe.predict_proba(X_val)[:, 1]

            aurocs.append(roc_auc_score(y_val, y_prob))
            accuracies.append(accuracy_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            recalls.append(recall_score(y_val, y_pred, zero_division=0))

        # 在全部训练数据上训最终探针, 在 held-out test 上评估
        probe.fit(X, y)
        self.probes[layer_idx] = probe

        metrics = {
            "auroc_mean": np.mean(aurocs),
            "auroc_std": np.std(aurocs),
            "accuracy_mean": np.mean(accuracies),
            "f1_mean": np.mean(f1s),
            "precision_mean": np.mean(precisions),
            "recall_mean": np.mean(recalls),
        }

        if X_test is not None and y_test is not None:
            y_test_pred = probe.predict(X_test)
            y_test_prob = probe.predict_proba(X_test)[:, 1]
            metrics["test_auroc"] = roc_auc_score(y_test, y_test_prob)
            metrics["test_f1"] = f1_score(y_test, y_test_pred, zero_division=0)
            metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)

        self.metrics[layer_idx] = metrics
        return metrics

    def train_all_layers(
        self,
        layer_states: Dict[int, np.ndarray],
        y: np.ndarray,
        layer_states_test: Optional[Dict[int, np.ndarray]] = None,
        y_test: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[int, dict]:
        """
        在所有层上分别训探针.

        Parameters
        ----------
        layer_states : dict
            训练激活. key = layer_idx, value = [n_samples, hidden_dim]
        y : np.ndarray
            训练标签
        layer_states_test, y_test : optional
            held-out test 数据
        """
        for layer_idx in sorted(layer_states.keys()):
            X_test = layer_states_test.get(layer_idx) if layer_states_test else None
            if verbose:
                print(f"  Training probe at layer {layer_idx}...", end=" ")
            metrics = self.train_layer(
                layer_states[layer_idx], y, layer_idx,
                X_test=X_test, y_test=y_test,
            )
            if verbose:
                parts = [f"CV AUROC = {metrics['auroc_mean']:.4f} ± {metrics['auroc_std']:.4f}"]
                if 'test_auroc' in metrics:
                    parts.append(f"Test AUROC = {metrics['test_auroc']:.4f}")
                print(" | ".join(parts))

        return dict(self.metrics)

    def get_best_layer(self) -> int:
        """返回 AUROC 最高的层."""
        return max(self.metrics, key=lambda k: self.metrics[k]["auroc_mean"])

    def predict(self, X: np.ndarray, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """在指定层做预测. 返回 (y_pred, y_prob)."""
        probe = self.probes[layer_idx]
        y_pred = probe.predict(X)
        y_prob = probe.predict_proba(X)[:, 1]
        return y_pred, y_prob

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {"probes": self.probes, "metrics": self.metrics, "probe_type": self.probe_type},
            path,
        )

    def load(self, path: str):
        data = joblib.load(path)
        self.probes = data["probes"]
        self.metrics = data["metrics"]
        self.probe_type = data["probe_type"]
