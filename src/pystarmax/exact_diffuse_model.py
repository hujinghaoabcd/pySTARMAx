# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse STARIMA estimator facade with state smoothing."""

from __future__ import annotations

from typing import Any

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


class ExactDiffuseKalmanSTARIMA(_ExactDiffuseKalmanSTARIMA):
    """Exact diffuse ordinary STARIMA with fixed-interval state smoothing."""

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


__all__ = [
    "ExactDiffuseKalmanSTARIMA",
    "ExactDiffuseKalmanSTARIMAResult",
    "ExactDiffuseSmootherResult",
]
