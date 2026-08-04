# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Forecast-interval facade for conditional ordinary Kalman STARIMA."""

from __future__ import annotations

import numpy as np

from pystarmax.forecasting import ForecastInterval
from pystarmax.integrated_maximum_likelihood import (
    KalmanSTARIMA as _KalmanSTARIMA,
)
from pystarmax.integrated_maximum_likelihood import KalmanSTARIMAResult
from pystarmax.kalman_forecasting import (
    integrated_kalman_forecast_interval,
    kalman_forecast_interval,
)


class KalmanSTARIMA(_KalmanSTARIMA):
    """Conditional ordinary Kalman STARIMA with Gaussian forecast intervals."""

    def predict_differenced_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_simulations: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return an interval on the highest ordinary-difference scale."""
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
        """Return a pathwise inverse-differenced original-scale interval."""
        self._require_fit()
        if self.differencing_state_ is None:
            raise RuntimeError(
                "original-scale interval prediction requires finite terminal "
                "differencing anchors; use predict_differenced_interval() or "
                "refit with finite trailing data"
            )
        return integrated_kalman_forecast_interval(
            self.filter(),
            self.differencing_state_,
            steps=steps,
            level=level,
            n_simulations=n_simulations,
            random_state=random_state,
        )


__all__ = ["KalmanSTARIMA", "KalmanSTARIMAResult"]
