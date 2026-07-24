"""SkillProbe v4: Boundary-Gated Centered Geometric Detector.

v1: oper+bnd + PCA(8) + LR(C=0.01) + L7+ layer selection
v2: v1 + decl-probe dual-pass max() fusion + malice attribution
v3: v1 + centering (−PC1 per layer, removes prompt-template anisotropy)
v4: v3 + Boundary safety gate (Gate<0.15)

Architecture:
    1. Extract oper + bnd hidden states (8 normalized layers, L7+ restricted)
    2. Center: per-layer mean subtraction + PC1 removal
    3. Geometric features: cos(oper,bnd), norm_ratio, direction_diff
    4. PCA(8) → LogisticRegression(C=0.01) → score_v3
    5. Boundary safety gate:
       if score_boundary < 0.15: risk = score_boundary
       else: risk = score_v3
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────
NORMALIZED_DEPTHS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
LAYER_INDICES = [3, 7, 11, 15, 19, 23, 27, 31]  # Llama-3.1-8B
L7P_START = 1       # Skip L3 (reads prompt template, not skill content)
GATE_THRESHOLD = 0.15  # Universal boundary safety threshold
PCA_COMPONENTS = 8
C_VALUE = 0.01


def normalized_layer_indices(layer_count: int) -> list[int]:
    """Map normalized depths to absolute layer indices."""
    return sorted({
        min(layer_count - 1, max(0, int(round(d * layer_count)) - 1))
        for d in NORMALIZED_DEPTHS
    })


# ── Geometric Features ─────────────────────────────────────

def geometric_features(
    operation: np.ndarray,   # (N, 8, 4096) float64
    boundary: np.ndarray,    # (N, 8, 4096) float64
    layer_idx: int,
) -> np.ndarray:
    """Compute 5 geometric features at a single layer.

    Returns (N, 4098) = direction_diff(4096d) + cos(1d) + norm_ratio(1d)
    """
    o = operation[:, layer_idx, :].astype(np.float64)
    r = boundary[:, layer_idx, :].astype(np.float64)

    o_norm = np.linalg.norm(o, axis=1, keepdims=True)
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    o_norm = np.where(o_norm == 0, 1e-12, o_norm)
    r_norm = np.where(r_norm == 0, 1e-12, r_norm)

    o_unit = o / o_norm
    r_unit = r / r_norm

    direction_diff = o_unit - r_unit                    # (N, 4096)
    cos_or = np.sum(o_unit * r_unit, axis=1, keepdims=True)  # (N, 1)
    norm_ratio = o_norm / (r_norm + 1e-12)               # (N, 1)

    return np.concatenate([direction_diff, cos_or, norm_ratio], axis=1)


# ── Centering ──────────────────────────────────────────────

def center_remove_pc1(
    train_arr: np.ndarray,   # (N, 8, 4096)
    test_arr: np.ndarray,    # (M, 8, 4096)
) -> None:
    """In-place per-layer centering: subtract mean, then remove first PC.

    Fit on train, transform both train and test.
    Replaces prompt-template-dominated variance with skill-content signal.
    """
    for li in range(train_arr.shape[1]):
        # Mean centering
        mean_tr = train_arr[:, li, :].mean(axis=0, keepdims=True)
        train_arr[:, li, :] -= mean_tr
        test_arr[:, li, :] -= mean_tr

        # Remove first principal component
        pca = PCA(n_components=1, random_state=42)
        X_tr = train_arr[:, li, :]
        pc1_tr = pca.inverse_transform(pca.fit_transform(X_tr))
        train_arr[:, li, :] -= pc1_tr

        pc1_te = pca.inverse_transform(pca.transform(test_arr[:, li, :]))
        test_arr[:, li, :] -= pc1_te


# ── Layer Selection ────────────────────────────────────────

def select_best_layer(
    geo_train: np.ndarray,   # (N, 8, feature_dim)
    labels: np.ndarray,
    groups: np.ndarray,
    cv,
    l7p_start: int = L7P_START,
) -> tuple[int, float]:
    """Select best layer via grouped cross-validation (L7+ only)."""
    best_layer, best_auc = -1, 0.0
    for li in range(l7p_start, geo_train.shape[1]):
        X = geo_train[:, li, :]
        nc = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
        aucs = []
        for train_i, val_i in cv.split(X, labels, groups):
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=nc),
                LogisticRegression(C=C_VALUE, max_iter=5000),
            )
            model.fit(X[train_i], labels[train_i])
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(
                labels[val_i],
                model.predict_proba(X[val_i])[:, 1],
            ))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc, best_layer = mean_auc, li
    return best_layer, best_auc


# ── v4 Pipeline ────────────────────────────────────────────

class SkillProbeV4:
    """Boundary-gated centered geometric detector."""

    def __init__(
        self,
        gate_threshold: float = GATE_THRESHOLD,
        pca_components: int = PCA_COMPONENTS,
        c_value: float = C_VALUE,
    ):
        self.gate_threshold = gate_threshold
        self.pca_components = pca_components
        self.c_value = c_value
        self.boundary_model = None
        self.v3_model = None
        self.best_layer = None

    def fit(
        self,
        operation: np.ndarray,   # (N, 8, 4096)
        response: np.ndarray,    # (N, 8, 4096)
        labels: np.ndarray,
        groups: np.ndarray,
        cv,
    ) -> SkillProbeV4:
        """Train boundary probe, center features, train v3 probe."""
        from sklearn.metrics import roc_auc_score

        # 1. Train Boundary probe (L2-LR on L23 boundary token)
        bd_li = 6  # L23 = index 6
        self.boundary_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.c_value, max_iter=5000),
        )
        self.boundary_model.fit(response[:, bd_li, :], labels)

        # 2. Center features
        Xo = operation.copy()
        Xr = response.copy()
        center_remove_pc1(Xo, Xo)  # in-place on training data
        center_remove_pc1(Xr, Xr)

        # 3. Build geometric features
        geo_train = np.stack([
            geometric_features(Xo, Xr, li)
            for li in range(operation.shape[1])
        ], axis=1)

        # 4. Layer selection
        self.best_layer, _ = select_best_layer(geo_train, labels, groups, cv)

        # 5. Train v3 probe
        nc = min(self.pca_components, geo_train.shape[0] - 1, geo_train.shape[2])
        self.v3_model = make_pipeline(
            StandardScaler(),
            PCA(n_components=nc),
            LogisticRegression(C=self.c_value, max_iter=5000),
        )
        self.v3_model.fit(geo_train[:, self.best_layer, :], labels)

        return self

    def predict_proba(
        self,
        operation: np.ndarray,   # (M, 8, 4096) float64 or float16
        response: np.ndarray,
        operation_train: np.ndarray,    # needed for centering fit
        response_train: np.ndarray,
    ) -> np.ndarray:
        """Return risk scores with boundary gating applied."""
        # Center test features using training data
        Xo_test = operation.astype(np.float64).copy()
        Xr_test = response.astype(np.float64).copy()
        Xo_train = operation_train.astype(np.float64).copy()
        Xr_train = response_train.astype(np.float64).copy()
        center_remove_pc1(Xo_train, Xo_test)
        center_remove_pc1(Xr_train, Xr_test)

        # Boundary scores (L23)
        bd_scores = self.boundary_model.predict_proba(
            response.astype(np.float64)[:, 6, :]
        )[:, 1]

        # v3 scores
        geo_test = np.stack([
            geometric_features(Xo_test, Xr_test, li)
            for li in range(operation.shape[1])
        ], axis=1)
        v3_scores = self.v3_model.predict_proba(
            geo_test[:, self.best_layer, :]
        )[:, 1]

        # Boundary safety gate
        gated = np.where(bd_scores < self.gate_threshold, bd_scores, v3_scores)
        return gated
