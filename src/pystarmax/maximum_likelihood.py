# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Gaussian maximum-likelihood estimation for stationary STARMA models."""

from __future__ import annotations

from typing import Any

from pystarmax._maximum_likelihood_model import KalmanSTARMA as _KalmanSTARMA
from pystarmax._maximum_likelihood_result import KalmanSTARMAResult
from pystarmax._maximum_likelihood_utils import CovarianceType
from pystarmax.likelihood_inference import (
    LikelihoodInferenceResult,
    infer_kalman_starma,
)
from pystarmax.smoothing import KalmanSmootherResult, kalman_smoother


class KalmanSTARMA(_KalmanSTARMA):
    """Kalman STARMA estimator with likelihood inference and state smoothing."""

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


__all__ = [
    "CovarianceType",
    "KalmanSTARMA",
    "KalmanSTARMAResult",
    "KalmanSmootherResult",
    "LikelihoodInferenceResult",
]
