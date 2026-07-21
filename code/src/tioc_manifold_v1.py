"""Benign-only conditional residual scoring for TIOC hidden states."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [sample, feature] matrix")
    return array


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
    standard_scale = float(np.std(values))
    scale = mad_scale if mad_scale > 1e-10 else standard_scale
    return median, max(scale, 1e-10)


class BenignConditionalResidual:
    """Model the actual-effect manifold conditioned on trusted/declaration state."""

    def __init__(self, components: int = 8, ridge_alpha: float = 1.0, seed: int = 20260715):
        if components <= 0 or ridge_alpha <= 0:
            raise ValueError("components and ridge_alpha must be positive")
        self.components = int(components)
        self.ridge_alpha = float(ridge_alpha)
        self.seed = int(seed)
        self.condition_pca: Optional[PCA] = None
        self.actual_pca: Optional[PCA] = None
        self.regressor: Optional[Ridge] = None
        self.residual_covariance: Optional[LedoitWolf] = None
        self.component_scales: Optional[list[tuple[float, float]]] = None

    def fit(self, condition: np.ndarray, actual_effect: np.ndarray):
        condition = _matrix(condition, "condition")
        actual_effect = _matrix(actual_effect, "actual_effect")
        if len(condition) != len(actual_effect):
            raise ValueError("condition and actual_effect sample counts differ")
        maximum = min(
            self.components,
            len(condition) - 1,
            condition.shape[1],
            actual_effect.shape[1],
        )
        if maximum < 1:
            raise ValueError("At least two benign samples are required")
        self.condition_pca = PCA(
            n_components=maximum, svd_solver="randomized", random_state=self.seed
        )
        self.actual_pca = PCA(
            n_components=maximum, svd_solver="randomized", random_state=self.seed
        )
        condition_latent = self.condition_pca.fit_transform(condition)
        actual_latent = self.actual_pca.fit_transform(actual_effect)
        self.regressor = Ridge(alpha=self.ridge_alpha)
        self.regressor.fit(condition_latent, actual_latent)
        predicted_latent = self.regressor.predict(condition_latent)
        residual_latent = actual_latent - predicted_latent
        self.residual_covariance = LedoitWolf().fit(residual_latent)
        raw = self._raw_components(condition, actual_effect)
        self.component_scales = [
            robust_location_scale(raw[:, index]) for index in range(raw.shape[1])
        ]
        return self

    def _require_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.condition_pca,
                self.actual_pca,
                self.regressor,
                self.residual_covariance,
            )
        ):
            raise RuntimeError("BenignConditionalResidual is not fitted")

    def _raw_components(
        self, condition: np.ndarray, actual_effect: np.ndarray
    ) -> np.ndarray:
        self._require_fitted()
        assert self.condition_pca is not None
        assert self.actual_pca is not None
        assert self.regressor is not None
        assert self.residual_covariance is not None
        condition_latent = self.condition_pca.transform(condition)
        actual_latent = self.actual_pca.transform(actual_effect)
        predicted_latent = self.regressor.predict(condition_latent)
        residual_latent = actual_latent - predicted_latent
        centered = residual_latent - self.residual_covariance.location_
        mahalanobis = np.einsum(
            "ni,ij,nj->n", centered, self.residual_covariance.precision_, centered
        )
        predicted_actual = self.actual_pca.inverse_transform(predicted_latent)
        reconstructed_actual = self.actual_pca.inverse_transform(actual_latent)
        denominator = np.sqrt(actual_effect.shape[1])
        prediction_rmse = np.linalg.norm(
            actual_effect - predicted_actual, axis=1
        ) / max(denominator, 1.0)
        reconstruction_rmse = np.linalg.norm(
            actual_effect - reconstructed_actual, axis=1
        ) / max(denominator, 1.0)
        return np.stack(
            (mahalanobis, prediction_rmse, reconstruction_rmse), axis=1
        )

    def score_components(
        self, condition: np.ndarray, actual_effect: np.ndarray
    ) -> np.ndarray:
        condition = _matrix(condition, "condition")
        actual_effect = _matrix(actual_effect, "actual_effect")
        if len(condition) != len(actual_effect):
            raise ValueError("condition and actual_effect sample counts differ")
        if self.component_scales is None:
            raise RuntimeError("BenignConditionalResidual is not fitted")
        raw = self._raw_components(condition, actual_effect)
        return np.stack(
            [
                (raw[:, index] - location) / scale
                for index, (location, scale) in enumerate(self.component_scales)
            ],
            axis=1,
        )

    def decision_function(
        self, condition: np.ndarray, actual_effect: np.ndarray
    ) -> np.ndarray:
        return self.score_components(condition, actual_effect).mean(axis=1)
