# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Gaussian maximum-likelihood estimation for stationary STARMA models."""

from __future__ import annotations

from typing import Any

import numpy as np

from pystarmax._maximum_likelihood_model import KalmanSTARMA as _KalmanSTARMA
from pystarmax._maximum_likelihood_result import KalmanSTARMAResult
from pystarmax._maximum_likelihood_utils import CovarianceType
from pystarmax.forecasting import ForecastInterval
from pystarmax.innovation_smoothing import (
    InnovationDisturbanceResult,
    innovation_disturbance_smoother,
)
from pystarmax.kalman_forecasting import kalman_forecast_interval
from pystarmax.likelihood_inference import (
    LikelihoodInferenceResult,
    infer_kalman_starma,
)
from pystarmax.smoothing import KalmanSmootherResult, kalman_smoother


class KalmanSTARMA(_KalmanSTARMA):
    """Kalman STARMA estimator with inference and disturbance smoothing."""

    def infer(
        self,
        *,
        relative_step: float = 1e-4,
        absolute_step: float = 1e-6,
        rcond: float = 1e-10,
        allow_singular: bool = False,
    ) -> LikelihoodInferenceResult:
        """Estimate observed-information covariance and coefficient uncertainty."""
        return infer_kalman_starma(
            self,
            relative_step=relative_step,
            absolute_step=absolute_step,
            rcond=rcond,
            allow_singular=allow_singular,
        )

    def smooth(
        self,
        data: Any | None = None,
        *,
        rcond: float = 1e-10,
    ) -> KalmanSmootherResult:
        """Smooth training data or a new incomplete observation matrix."""
        return kalman_smoother(self.filter(data), rcond=rcond)

    def smooth_innovation_disturbances(
        self,
        data: Any | None = None,
        *,
        rcond: float = 1e-10,
    ) -> InnovationDisturbanceResult:
        """Smooth original location-level process innovations."""
        return innovation_disturbance_smoother(
            self.smooth(data, rcond=rcond),
            rcond=rcond,
        )

    def predict_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_simulations: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return a fixed-parameter Gaussian forecast interval."""
        return kalman_forecast_interval(
            self.filter(),
            steps=steps,
            level=level,
            n_simulations=n_simulations,
            random_state=random_state,
        )


__all__ = [
    "CovarianceType",
    "ForecastInterval",
    "InnovationDisturbanceResult",
    "KalmanSTARMA",
    "KalmanSTARMAResult",
    "KalmanSmootherResult",
    "LikelihoodInferenceResult",
]
