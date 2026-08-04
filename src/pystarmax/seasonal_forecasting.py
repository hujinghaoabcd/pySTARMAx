# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Forecast-interval facade for multiplicative seasonal Kalman STARIMA."""

from __future__ import annotations

import numpy as np

from pystarmax.forecasting import ForecastInterval
from pystarmax.kalman_forecasting import (
    integrated_kalman_forecast_interval,
    kalman_forecast_interval,
)
from pystarmax.seasonal_likelihood_inference import (
    SeasonalKalmanSTARIMA as _SeasonalKalmanSTARIMA,
)
from pystarmax.seasonal_likelihood_inference import infer_seasonal_kalman_starima
from pystarmax.seasonal_maximum_likelihood import (
    SeasonalKalmanAdmissibility,
    SeasonalKalmanSTARIMAResult,
)


class SeasonalKalmanSTARIMA(_SeasonalKalmanSTARIMA):
    """Seasonal Kalman STARIMA with inference and Gaussian intervals."""

    def predict_differenced_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_simulations: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return an interval on the combined transformed scale."""
        self._require_fit()
        return kalman_forecast_interval(
            self.filter(),
            steps=steps,
            level=level,
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
        """Return a pathwise ordinary-seasonal original-scale interval."""
        self._require_fit()
        if self.differencing_state_ is None:
            raise RuntimeError(
                "original-scale interval prediction requires finite ordinary and "
                "seasonal terminal histories; use predict_differenced_interval() "
                "or refit with finite trailing data"
            )
        return integrated_kalman_forecast_interval(
            self.filter(),
            self.differencing_state_,
            steps=steps,
            level=level,
            n_simulations=n_simulations,
            random_state=random_state,
        )


__all__ = [
    "SeasonalKalmanAdmissibility",
    "SeasonalKalmanSTARIMA",
    "SeasonalKalmanSTARIMAResult",
    "infer_seasonal_kalman_starima",
]
