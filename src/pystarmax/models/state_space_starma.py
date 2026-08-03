# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""State-space-enabled stationary STAR and STARMA estimators."""

from __future__ import annotations

from typing import Any

from pystarmax.models.bootstrap_starma import STARMA as _STARMA
from pystarmax.state_space import (
    Initialization,
    KalmanFilterResult,
    StateSpaceModel,
    fitted_starma_state_space,
    kalman_filter,
)


class STARMA(_STARMA):
    """STARMA estimator with bootstrap and state-space diagnostics."""

    def _new_unfitted(self) -> "STARMA":
        return STARMA(
            ar_order=self.ar_order,
            ma_order=self.ma_order,
            include_intercept=self.include_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            ridge=self.ridge,
        )

    def to_state_space(self) -> StateSpaceModel:
        """Return the state-space representation of fitted parameters."""
        return fitted_starma_state_space(self)

    def filter_state_space(
        self,
        data: Any | None = None,
        *,
        initialization: Initialization = "stationary",
        initial_state: Any | None = None,
        initial_covariance: Any | None = None,
        diffuse_scale: float = 1e6,
    ) -> KalmanFilterResult:
        """Filter fitted parameters against complete or partially missing data."""
        if data is None:
            if self.data_ is None:
                raise RuntimeError("fit must be called before state-space filtering")
            observations: Any = self.data_
        else:
            observations = data
        return kalman_filter(
            observations,
            self.to_state_space(),
            initialization=initialization,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            diffuse_scale=diffuse_scale,
        )


class STAR(STARMA):
    """Pure space-time autoregression with state-space diagnostics."""

    def __init__(
        self,
        ar_order: int = 1,
        *,
        include_intercept: bool = True,
        ridge: float = 0.0,
    ) -> None:
        super().__init__(
            ar_order=ar_order,
            ma_order=0,
            include_intercept=include_intercept,
            ridge=ridge,
        )

    def _new_unfitted(self) -> "STAR":
        return STAR(
            ar_order=self.ar_order,
            include_intercept=self.include_intercept,
            ridge=self.ridge,
        )
