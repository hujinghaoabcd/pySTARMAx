# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse STARIMA estimator facade with forecasting and inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from pystarmax.exact_diffuse_disturbance_smoothing import (
    ExactDiffuseDisturbanceResult,
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_forecasting import (
    _projected_forecast_interval,
    exact_diffuse_forecast_interval,
)
from pystarmax.exact_diffuse_inference import infer_exact_diffuse_kalman_starima
from pystarmax.exact_diffuse_mle import (
    ExactDiffuseKalmanSTARIMA as _ExactDiffuseKalmanSTARIMA,
)
from pystarmax.exact_diffuse_mle import (
    ExactDiffuseKalmanSTARIMAResult,
)
from pystarmax.exact_diffuse_simulation_smoothing import (
    ExactDiffuseSimulationSmootherResult,
    exact_diffuse_simulation_smoother,
)
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.forecasting import ForecastInterval
from pystarmax.likelihood_inference import LikelihoodInferenceResult


class ExactDiffuseKalmanSTARIMA(_ExactDiffuseKalmanSTARIMA):
    """Exact diffuse ordinary STARIMA with forecasting and inference."""

    def smooth(
        self,
        data: Any | None = None,
        *,
        tolerance: float | None = None,
    ) -> ExactDiffuseSmootherResult:
        """Smooth training or newly initialized original-level observations."""
        return exact_diffuse_smoother(
            self.filter(data),
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
    ) -> ExactDiffuseSimulationSmootherResult:
        """Draw complete state paths conditional on training or new observations."""
        return exact_diffuse_simulation_smoother(
            self.filter(data),
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
        """Smooth primitive process innovations and their state-equation images."""
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
        """Infer curvature from the fitted original-level exact diffuse objective."""
        return infer_exact_diffuse_kalman_starima(
            self,
            relative_step=relative_step,
            absolute_step=absolute_step,
            rcond=rcond,
            allow_singular=allow_singular,
        )

    def predict_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_simulations: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return an original-level interval from the exact terminal posterior."""
        return exact_diffuse_forecast_interval(
            self.filter(),
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
        """Return an interval on the highest ordinary-difference scale."""
        result, integrated, filtered = self._require_fit()
        n_locations = integrated.model.n_locations
        offset = self.integration_order * n_locations
        design = np.zeros(
            (n_locations, integrated.model.state_dim),
            dtype=float,
        )
        design[:, offset:] = result.transformed_state_space.design
        return _projected_forecast_interval(
            filtered,
            design,
            steps=steps,
            level=level,
            n_simulations=n_simulations,
            random_state=random_state,
            method=(
                "fixed-parameter exact diffuse terminal-posterior and innovation "
                "simulation on the highest ordinary-difference scale"
            ),
        )


__all__ = [
    "ExactDiffuseDisturbanceResult",
    "ExactDiffuseKalmanSTARIMA",
    "ExactDiffuseKalmanSTARIMAResult",
    "ExactDiffuseSimulationSmootherResult",
    "ExactDiffuseSmootherResult",
    "ForecastInterval",
    "LikelihoodInferenceResult",
]
