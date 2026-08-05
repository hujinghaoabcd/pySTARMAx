# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Seasonal exact diffuse STARIMA facade with posterior operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.exact_diffuse_disturbance_smoothing import (
    ExactDiffuseDisturbanceResult,
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_lag_one_covariance import (
    ExactDiffuseLagOneCovarianceResult,
    exact_diffuse_lag_one_covariance,
)
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.forecasting import ForecastInterval
from pystarmax.likelihood_inference import LikelihoodInferenceResult
from pystarmax.seasonal_exact_diffuse_forecasting import (
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)
from pystarmax.seasonal_exact_diffuse_inference import (
    infer_seasonal_exact_diffuse_kalman_starima,
)
from pystarmax.seasonal_exact_diffuse_mle import (
    SeasonalExactDiffuseKalmanSTARIMA as _SeasonalExactDiffuseKalmanSTARIMA,
)
from pystarmax.seasonal_exact_diffuse_mle import (
    SeasonalExactDiffuseKalmanSTARIMAResult,
)
from pystarmax.seasonal_exact_diffuse_simulation_smoothing import (
    SeasonalExactDiffuseSimulationSmootherResult,
    seasonal_exact_diffuse_simulation_smoother,
)


class SeasonalExactDiffuseKalmanSTARIMA(_SeasonalExactDiffuseKalmanSTARIMA):
    """Seasonal exact diffuse STARIMA with posterior operations."""

    def smooth(
        self,
        data: Any | None = None,
        *,
        tolerance: float | None = None,
    ) -> ExactDiffuseSmootherResult:
        """Smooth training or newly initialized original-level observations.

        Passing ``data`` starts a fresh exact diffuse filter with the fitted
        seasonal state-space parameters. It does not continue from the terminal
        training posterior.
        """
        return exact_diffuse_smoother(
            self.filter(data),
            tolerance=tolerance,
        )

    def smooth_lag_one_covariance(
        self,
        data: Any | None = None,
        *,
        rcond: float = 1e-10,
        tolerance: float | None = None,
    ) -> ExactDiffuseLagOneCovarianceResult:
        """Return adjacent-time covariance for the complete seasonal state."""
        return exact_diffuse_lag_one_covariance(
            self.filter(data),
            rcond=rcond,
            tolerance=tolerance,
        )

    def simulate_smoothing_paths(
        self,
        data: Any | None = None,
        *,
        n_simulations: int = 1000,
        random_state: int | np.random.Generator | None = None,
        rcond: float = 1e-10,
        tolerance: float | None = None,
    ) -> SeasonalExactDiffuseSimulationSmootherResult:
        """Draw conditional complete and transformed seasonal state paths.

        Passing ``data`` starts a fresh exact diffuse filter under the fitted
        seasonal parameters. Every observed original-level cell is conditioned
        exactly; the transformed paths are projections of those same draws.
        """
        _result, integrated, _filtered = self._require_fit()
        return seasonal_exact_diffuse_simulation_smoother(
            self.filter(data),
            integrated,
            n_simulations=n_simulations,
            random_state=random_state,
            rcond=rcond,
            tolerance=tolerance,
        )

    def smooth_innovation_disturbances(
        self,
        data: Any | None = None,
        *,
        tolerance: float | None = None,
    ) -> ExactDiffuseDisturbanceResult:
        """Smooth primitive innovations and their state-equation images."""
        return exact_diffuse_disturbance_smoother(
            self.smooth(data, tolerance=tolerance),
            tolerance=tolerance,
        )

    def likelihood_inference(
        self,
        *,
        relative_step: float = 1e-4,
        absolute_step: float = 1e-6,
        rcond: float = 1e-10,
        allow_singular: bool = False,
    ) -> LikelihoodInferenceResult:
        """Infer curvature from the seasonal original-level exact objective."""
        return infer_seasonal_exact_diffuse_kalman_starima(
            self,
            relative_step=relative_step,
            absolute_step=absolute_step,
            rcond=rcond,
            allow_singular=allow_singular,
        )

    def simulate_forecast_paths(
        self,
        *,
        steps: int,
        n_simulations: int,
        random_state: int | np.random.Generator | None = None,
    ) -> FloatArray:
        """Simulate future original-level paths from the fitted posterior."""
        _result, integrated, filtered = self._require_fit()
        return simulate_seasonal_exact_diffuse_forecast_paths(
            filtered,
            integrated,
            steps=steps,
            n_simulations=n_simulations,
            random_state=random_state,
        )

    def simulate_differenced_forecast_paths(
        self,
        *,
        steps: int,
        n_simulations: int,
        random_state: int | np.random.Generator | None = None,
    ) -> FloatArray:
        """Simulate paths on the combined ordinary-seasonal difference scale."""
        _result, integrated, filtered = self._require_fit()
        return simulate_seasonal_exact_diffuse_differenced_forecast_paths(
            filtered,
            integrated,
            steps=steps,
            n_simulations=n_simulations,
            random_state=random_state,
        )

    def predict_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_simulations: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return a pathwise original-level exact diffuse interval."""
        _result, integrated, filtered = self._require_fit()
        return seasonal_exact_diffuse_forecast_interval(
            filtered,
            integrated,
            steps=steps,
            level=level,
            n_simulations=n_simulations,
            random_state=random_state,
        )

    def predict_differenced_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_simulations: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return an interval on the combined transformed scale."""
        _result, integrated, filtered = self._require_fit()
        return seasonal_exact_diffuse_differenced_forecast_interval(
            filtered,
            integrated,
            steps=steps,
            level=level,
            n_simulations=n_simulations,
            random_state=random_state,
        )


__all__ = [
    "ExactDiffuseDisturbanceResult",
    "ExactDiffuseLagOneCovarianceResult",
    "ExactDiffuseSmootherResult",
    "ForecastInterval",
    "LikelihoodInferenceResult",
    "SeasonalExactDiffuseKalmanSTARIMA",
    "SeasonalExactDiffuseKalmanSTARIMAResult",
    "SeasonalExactDiffuseSimulationSmootherResult",
]
