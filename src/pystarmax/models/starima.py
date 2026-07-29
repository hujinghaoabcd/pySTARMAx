# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Ordinary integrated STARMA model."""

from __future__ import annotations

from typing import Any

import numpy as np

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)
from pystarmax.differencing import (
    DifferencingState,
    ordinary_difference,
    restore_fitted_values,
)
from pystarmax.forecasting import (
    ForecastInterval,
    interval_from_paths,
    validate_interval_arguments,
)
from pystarmax.models.starma import STARMA
from pystarmax.results import STARMAResult


class STARIMA:
    """Fit an ordinary ``STARIMA(p, d, q)`` model.

    The class applies ordinary temporal differencing, delegates estimation to
    :class:`pystarmax.STARMA`, and integrates recursive forecasts back to the
    original observation scale.

    Parameters
    ----------
    ar_order:
        Maximum temporal autoregressive order ``p``.
    integration_order:
        Ordinary temporal differencing order ``d``.
    ma_order:
        Maximum temporal moving-average order ``q``.
    include_intercept:
        Include a common intercept on the differenced scale. For ``d=1`` this
        acts as a drift term on the original scale.
    max_iter, tol, ridge:
        Passed unchanged to the conditional STARMA estimator.
    """

    def __init__(
        self,
        ar_order: int = 1,
        integration_order: int = 1,
        ma_order: int = 0,
        *,
        include_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-8,
        ridge: float = 0.0,
    ) -> None:
        self.integration_order = validate_nonnegative_int(
            integration_order, name="integration_order"
        )
        self.core_model = STARMA(
            ar_order=ar_order,
            ma_order=ma_order,
            include_intercept=include_intercept,
            max_iter=max_iter,
            tol=tol,
            ridge=ridge,
        )
        self.result_: STARMAResult | None = None
        self.data_: FloatArray | None = None
        self.differenced_data_: FloatArray | None = None
        self.differencing_state_: DifferencingState | None = None

    @property
    def ar_order(self) -> int:
        """Temporal autoregressive order ``p``."""
        return self.core_model.ar_order

    @property
    def ma_order(self) -> int:
        """Temporal moving-average order ``q``."""
        return self.core_model.ma_order

    @property
    def max_lag(self) -> int:
        """Maximum conditional lag of the underlying STARMA model."""
        return self.core_model.max_lag

    @property
    def order(self) -> tuple[int, int, int]:
        """Model order tuple ``(p, d, q)``."""
        return self.ar_order, self.integration_order, self.ma_order

    def fit(self, data: Any, weights: Any) -> STARMAResult:
        """Fit the model and return a result on the differenced scale."""
        observations = validate_time_space(data)
        differenced, state = ordinary_difference(
            observations, order=self.integration_order
        )
        if differenced.shape[0] <= self.max_lag + 1:
            raise ValueError(
                "data contain too few time observations after differencing "
                "for the model order"
            )
        result = self.core_model.fit(differenced, weights)
        self.result_ = result
        self.data_ = observations.copy()
        self.differenced_data_ = differenced.copy()
        self.differencing_state_ = state
        return result

    def predict_differenced(self, steps: int = 1) -> FloatArray:
        """Forecast on the stationary differenced scale."""
        if self.result_ is None:
            raise RuntimeError("fit must be called before predict")
        return self.core_model.predict(steps=steps)

    def predict(self, steps: int = 1) -> FloatArray:
        """Forecast recursively and return values on the original scale."""
        if self.differencing_state_ is None:
            raise RuntimeError("fit must be called before predict")
        differenced = self.predict_differenced(steps=steps)
        return self.differencing_state_.inverse_forecast(differenced)

    def fitted_original(self) -> FloatArray:
        """Return aligned one-step fitted values on the original scale."""
        if self.result_ is None or self.data_ is None:
            raise RuntimeError("fit must be called before fitted_original")
        return restore_fitted_values(
            self.data_,
            self.result_.fitted_values,
            ordinary_order=self.integration_order,
        )

    def predict_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_simulations: int = 1000,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return an original-scale conditional innovation interval.

        Each simulated differenced path is recursively integrated with the
        immutable end-of-sample state before empirical quantiles are computed.
        Estimated-parameter uncertainty is not included.
        """
        if self.differencing_state_ is None:
            raise RuntimeError("fit must be called before predict")
        steps, level, n_simulations = validate_interval_arguments(
            steps=steps, level=level, n_simulations=n_simulations
        )
        differenced_paths = self.core_model._simulate_forecast_paths(
            steps=steps,
            n_simulations=n_simulations,
            random_state=random_state,
        )
        paths = np.stack(
            [
                self.differencing_state_.inverse_forecast(path)
                for path in differenced_paths
            ],
            axis=0,
        )
        return interval_from_paths(
            mean=self.predict(steps=steps),
            paths=paths,
            level=level,
        )
