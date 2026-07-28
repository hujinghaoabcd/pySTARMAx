# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Space-time correlation and residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)
from pystarmax.weights import coerce_weights


def _as_time_space_allow_nan(data: Any, *, name: str) -> FloatArray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least two times and one location")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinite values")
    return cast(FloatArray, np.ascontiguousarray(array, dtype=float))


def _drop_nan_rows(data: FloatArray) -> FloatArray:
    valid = np.all(np.isfinite(data), axis=1)
    result = data[valid]
    if result.shape[0] < 2:
        raise ValueError("data contain too few complete rows")
    return result


def _stcov(
    data: FloatArray,
    left: FloatArray,
    right: FloatArray,
    temporal_lag: int,
) -> float:
    values = []
    for time_index in range(temporal_lag, data.shape[0]):
        left_value = left @ data[time_index]
        right_value = right @ data[time_index - temporal_lag]
        values.append(float(left_value @ right_value) / data.shape[1])
    return float(np.mean(values))


def stacf(
    data: Any,
    weights: Any,
    *,
    max_tlag: int = 10,
    center: bool = True,
) -> pd.DataFrame:
    """Estimate the sample space-time autocorrelation function.

    The implementation follows the covariance normalization used in the
    classical STARMA literature, with spatial lag zero represented by identity.
    """
    max_tlag = validate_nonnegative_int(max_tlag, name="max_tlag")
    observations = _drop_nan_rows(_as_time_space_allow_nan(data, name="data"))
    resolved = coerce_weights(weights, n_locations=observations.shape[1])
    if max_tlag >= observations.shape[0]:
        raise ValueError("max_tlag must be smaller than the number of time rows")
    values = observations.copy()
    if center:
        values -= values.mean()
    identity = resolved[0]
    base_variance = _stcov(values, identity, identity, 0)
    output = np.empty((max_tlag + 1, len(resolved)), dtype=float)
    for temporal_lag in range(max_tlag + 1):
        for spatial_lag, matrix in enumerate(resolved):
            numerator = _stcov(values, matrix, identity, temporal_lag)
            left_variance = _stcov(values, matrix, matrix, 0)
            denominator = np.sqrt(max(left_variance * base_variance, 0.0))
            output[temporal_lag, spatial_lag] = (
                numerator / denominator if denominator > 0 else np.nan
            )
    return pd.DataFrame(
        output,
        index=pd.Index(range(max_tlag + 1), name="temporal_lag"),
        columns=pd.Index(resolved.names, name="spatial_lag"),
    )


def stpacf(
    data: Any,
    weights: Any,
    *,
    max_tlag: int = 5,
    center: bool = True,
) -> pd.DataFrame:
    """Estimate a regression-based finite-sample STPACF analogue.

    For each temporal order ``h``, the process is regressed on all spatial lags
    at temporal lags ``1..h``. Coefficients associated with lag ``h`` are
    returned. This transparent diagnostic will later be complemented by a
    dedicated Yule-Walker implementation.
    """
    max_tlag = validate_nonnegative_int(max_tlag, name="max_tlag")
    if max_tlag == 0:
        raise ValueError("max_tlag must be positive")
    observations = validate_time_space(data)
    resolved = coerce_weights(weights, n_locations=observations.shape[1])
    values = observations.copy()
    if center:
        values -= values.mean()
    if max_tlag >= values.shape[0]:
        raise ValueError("max_tlag must be smaller than the number of time rows")
    output = np.empty((max_tlag, len(resolved)), dtype=float)
    for order in range(1, max_tlag + 1):
        design_rows: list[FloatArray] = []
        target_rows: list[FloatArray] = []
        for time_index in range(order, values.shape[0]):
            columns: list[FloatArray] = []
            for temporal_lag in range(1, order + 1):
                columns.extend(
                    matrix @ values[time_index - temporal_lag] for matrix in resolved
                )
            design_rows.append(np.column_stack(columns))
            target_rows.append(values[time_index])
        design = np.vstack(design_rows)
        target = np.concatenate(target_rows)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        output[order - 1] = coefficients[-len(resolved) :]
    return pd.DataFrame(
        output,
        index=pd.Index(range(1, max_tlag + 1), name="temporal_lag"),
        columns=pd.Index(resolved.names, name="spatial_lag"),
    )


@dataclass(frozen=True, slots=True)
class PortmanteauResult:
    """Result of the space-time residual portmanteau test."""

    statistic: float
    degrees_of_freedom: int
    p_value: float
    max_tlag: int
    spatial_lags: int

    def __str__(self) -> str:
        return (
            "Space-time portmanteau test("
            f"statistic={self.statistic:.6f}, df={self.degrees_of_freedom}, "
            f"p_value={self.p_value:.6g})"
        )


def space_time_portmanteau(
    residuals: Any,
    weights: Any,
    *,
    max_tlag: int = 10,
    fit_params: int = 0,
) -> PortmanteauResult:
    """Test residual space-time autocorrelation with a Box-Pierce analogue."""
    max_tlag = validate_nonnegative_int(max_tlag, name="max_tlag")
    fit_params = validate_nonnegative_int(fit_params, name="fit_params")
    values = _drop_nan_rows(_as_time_space_allow_nan(residuals, name="residuals"))
    resolved = coerce_weights(weights, n_locations=values.shape[1])
    correlations = stacf(values, resolved, max_tlag=max_tlag)
    statistic = 0.0
    for temporal_lag in range(1, max_tlag + 1):
        statistic += (
            values.shape[1]
            * (values.shape[0] - temporal_lag)
            * float(np.nansum(correlations.loc[temporal_lag].to_numpy() ** 2))
        )
    degrees_of_freedom = max(1, max_tlag * len(resolved) - fit_params)
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    return PortmanteauResult(
        statistic=float(statistic),
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        max_tlag=max_tlag,
        spatial_lags=len(resolved),
    )
