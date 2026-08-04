# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Seasonal exact diffuse STARIMA facade with posterior smoothing."""

from __future__ import annotations

from typing import Any

from pystarmax.exact_diffuse_disturbance_smoothing import (
    ExactDiffuseDisturbanceResult,
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.seasonal_exact_diffuse_mle import (
    SeasonalExactDiffuseKalmanSTARIMA as _SeasonalExactDiffuseKalmanSTARIMA,
)
from pystarmax.seasonal_exact_diffuse_mle import (
    SeasonalExactDiffuseKalmanSTARIMAResult,
)


class SeasonalExactDiffuseKalmanSTARIMA(_SeasonalExactDiffuseKalmanSTARIMA):
    """Seasonal exact diffuse STARIMA with state and disturbance smoothing."""

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


__all__ = [
    "ExactDiffuseDisturbanceResult",
    "ExactDiffuseSmootherResult",
    "SeasonalExactDiffuseKalmanSTARIMA",
    "SeasonalExactDiffuseKalmanSTARIMAResult",
]
