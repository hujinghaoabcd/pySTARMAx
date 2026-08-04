# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse STARIMA estimator facade with smoothing and inference."""

from __future__ import annotations

from typing import Any

from pystarmax.exact_diffuse_disturbance_smoothing import (
    ExactDiffuseDisturbanceResult,
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_inference import infer_exact_diffuse_kalman_starima
from pystarmax.exact_diffuse_mle import (
    ExactDiffuseKalmanSTARIMA as _ExactDiffuseKalmanSTARIMA,
)
from pystarmax.exact_diffuse_mle import (
    ExactDiffuseKalmanSTARIMAResult,
)
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.likelihood_inference import LikelihoodInferenceResult


class ExactDiffuseKalmanSTARIMA(_ExactDiffuseKalmanSTARIMA):
    """Exact diffuse ordinary STARIMA with smoothing and curvature inference."""

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


__all__ = [
    "ExactDiffuseDisturbanceResult",
    "ExactDiffuseKalmanSTARIMA",
    "ExactDiffuseKalmanSTARIMAResult",
    "ExactDiffuseSmootherResult",
    "LikelihoodInferenceResult",
]
