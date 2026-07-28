# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Space-time correlation, partial-correlation, and residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy import stats

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)
from pystarmax.weights import SpatialWeights, coerce_weights

STPACFMethod: TypeAlias = Literal["yule-walker", "regression"]
YuleWalkerSolver: TypeAlias = Literal["auto", "solve", "lstsq"]


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


def _center_globally(data: FloatArray, *, center: bool) -> FloatArray:
    values = data.copy()
    if center:
        values -= values.mean()
    return values


def _validate_spatial_lag(index: int, weights: SpatialWeights, *, name: str) -> int:
    resolved = validate_nonnegative_int(index, name=name)
    if resolved >= len(weights):
        raise ValueError(f"{name} must be smaller than the number of weight matrices")
    return resolved


def _stcov(
    data: FloatArray,
    past_weight: FloatArray,
    future_weight: FloatArray,
    temporal_lag: int,
) -> float:
    """Compute the classical sample covariance ``gamma_lk(h)``.

    The convention is

    ``gamma_lk(h) = mean_t[(W_l z_t)' (W_k z_{t+h})] / N``.

    Naming the matrices by their time role avoids the transpose ambiguity that
    arises for non-symmetric row-standardized spatial weights.
    """
    n_pairs = data.shape[0] - temporal_lag
    total = 0.0
    for time_index in range(n_pairs):
        past_value = past_weight @ data[time_index]
        future_value = future_weight @ data[time_index + temporal_lag]
        total += float(past_value @ future_value)
    return float(total / (n_pairs * data.shape[1]))


def stcov(
    data: Any,
    weights: Any,
    *,
    past_spatial_lag: int = 0,
    future_spatial_lag: int = 0,
    temporal_lag: int = 0,
    center: bool = True,
) -> float:
    """Estimate a sample space-time covariance.

    Parameters
    ----------
    data:
        Observation matrix with shape ``(time, location)``.
    weights:
        Spatial-weight collection. Matrix zero is conventionally identity.
    past_spatial_lag:
        Spatial lag applied to the observation at time ``t``.
    future_spatial_lag:
        Spatial lag applied to the observation at time ``t + temporal_lag``.
    temporal_lag:
        Non-negative temporal displacement.
    center:
        Subtract the global space-time mean before computing the covariance.
    """
    observations = validate_time_space(data)
    resolved = coerce_weights(weights, n_locations=observations.shape[1])
    past_spatial_lag = _validate_spatial_lag(
        past_spatial_lag, resolved, name="past_spatial_lag"
    )
    future_spatial_lag = _validate_spatial_lag(
        future_spatial_lag, resolved, name="future_spatial_lag"
    )
    temporal_lag = validate_nonnegative_int(temporal_lag, name="temporal_lag")
    if temporal_lag >= observations.shape[0]:
        raise ValueError("temporal_lag must be smaller than the number of time rows")
    values = _center_globally(observations, center=center)
    return _stcov(
        values,
        resolved[past_spatial_lag],
        resolved[future_spatial_lag],
        temporal_lag,
    )


def _covariance_block(
    data: FloatArray,
    weights: SpatialWeights,
    temporal_lag: int,
) -> FloatArray:
    block = np.empty((len(weights), len(weights)), dtype=float)
    for past_lag, past_weight in enumerate(weights):
        for future_lag, future_weight in enumerate(weights):
            block[past_lag, future_lag] = _stcov(
                data,
                past_weight,
                future_weight,
                temporal_lag,
            )
    return block


def stacf(
    data: Any,
    weights: Any,
    *,
    max_tlag: int = 10,
    center: bool = True,
) -> pd.DataFrame:
    """Estimate the classical sample space-time autocorrelation function.

    For spatial lag ``l`` and temporal lag ``h``, the estimator is

    ``gamma_l0(h) / sqrt(gamma_ll(0) * gamma_00(0))``.

    Spatial lag zero is represented by the identity matrix. Rows include
    temporal lag zero because it is useful for numerical auditing, while the
    classical identification plots usually display rows beginning at lag one.
    """
    max_tlag = validate_nonnegative_int(max_tlag, name="max_tlag")
    observations = _drop_nan_rows(_as_time_space_allow_nan(data, name="data"))
    resolved = coerce_weights(weights, n_locations=observations.shape[1])
    if max_tlag >= observations.shape[0]:
        raise ValueError("max_tlag must be smaller than the number of time rows")
    values = _center_globally(observations, center=center)
    identity = resolved[0]
    base_variance = _stcov(values, identity, identity, 0)
    output = np.empty((max_tlag + 1, len(resolved)), dtype=float)
    for temporal_lag in range(max_tlag + 1):
        for spatial_lag, matrix in enumerate(resolved):
            numerator = _stcov(values, matrix, identity, temporal_lag)
            lag_variance = _stcov(values, matrix, matrix, 0)
            denominator = np.sqrt(max(lag_variance * base_variance, 0.0))
            output[temporal_lag, spatial_lag] = (
                numerator / denominator if denominator > 0 else np.nan
            )
    result = pd.DataFrame(
        output,
        index=pd.Index(range(max_tlag + 1), name="temporal_lag"),
        columns=pd.Index(resolved.names, name="spatial_lag"),
    )
    result.attrs["definition"] = "Pfeifer-Deutsch sample STACF"
    result.attrs["centered"] = center
    return result


def _yule_walker_system(
    data: FloatArray,
    weights: SpatialWeights,
    max_tlag: int,
) -> tuple[FloatArray, FloatArray]:
    """Construct the block Yule-Walker matrix and covariance vector."""
    spatial_lags = len(weights)
    blocks = [
        _covariance_block(data, weights, temporal_lag)
        for temporal_lag in range(max_tlag)
    ]
    matrix = np.empty((max_tlag * spatial_lags, max_tlag * spatial_lags), dtype=float)
    for row_lag in range(max_tlag):
        for column_lag in range(max_tlag):
            row_slice = slice(row_lag * spatial_lags, (row_lag + 1) * spatial_lags)
            column_slice = slice(
                column_lag * spatial_lags, (column_lag + 1) * spatial_lags
            )
            if row_lag == column_lag:
                block = blocks[0]
            elif row_lag > column_lag:
                block = blocks[row_lag - column_lag]
            else:
                block = blocks[column_lag - row_lag].T
            matrix[row_slice, column_slice] = block

    vector = np.empty(max_tlag * spatial_lags, dtype=float)
    identity = weights[0]
    for temporal_lag in range(1, max_tlag + 1):
        offset = (temporal_lag - 1) * spatial_lags
        for spatial_lag, matrix_lag in enumerate(weights):
            vector[offset + spatial_lag] = _stcov(
                data,
                matrix_lag,
                identity,
                temporal_lag,
            )
    return matrix, vector


def _solve_yule_walker(
    matrix: FloatArray,
    vector: FloatArray,
    *,
    solver: YuleWalkerSolver,
) -> tuple[FloatArray, str]:
    if solver not in {"auto", "solve", "lstsq"}:
        raise ValueError("solver must be 'auto', 'solve', or 'lstsq'")
    if solver == "lstsq":
        solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
        return np.asarray(solution, dtype=float), "lstsq"
    try:
        return np.asarray(np.linalg.solve(matrix, vector), dtype=float), "solve"
    except np.linalg.LinAlgError:
        if solver == "solve":
            raise
        solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
        return np.asarray(solution, dtype=float), "lstsq"


def stpacf_yule_walker(
    data: Any,
    weights: Any,
    *,
    max_tlag: int = 5,
    center: bool = True,
    solver: YuleWalkerSolver = "auto",
) -> pd.DataFrame:
    """Estimate the classical STPACF from nested Yule-Walker systems.

    A block Toeplitz covariance system is built for all temporal and spatial
    lags through ``max_tlag``. Following the classical Durbin-style procedure,
    leading principal systems are solved in temporal-major, spatial-minor
    order, and the newest coefficient is retained as the partial correlation.

    ``solver='auto'`` uses a direct solve and falls back to least squares only
    when a leading principal covariance system is singular.
    """
    max_tlag = validate_nonnegative_int(max_tlag, name="max_tlag")
    if max_tlag == 0:
        raise ValueError("max_tlag must be positive")
    observations = validate_time_space(data)
    resolved = coerce_weights(weights, n_locations=observations.shape[1])
    if max_tlag >= observations.shape[0]:
        raise ValueError("max_tlag must be smaller than the number of time rows")
    values = _center_globally(observations, center=center)
    matrix, vector = _yule_walker_system(values, resolved, max_tlag)
    output = np.empty((max_tlag, len(resolved)), dtype=float)
    solvers_used: list[str] = []
    for index in range(vector.size):
        stop = index + 1
        solution, solver_used = _solve_yule_walker(
            matrix[:stop, :stop],
            vector[:stop],
            solver=solver,
        )
        output.flat[index] = solution[-1]
        solvers_used.append(solver_used)
    result = pd.DataFrame(
        output,
        index=pd.Index(range(1, max_tlag + 1), name="temporal_lag"),
        columns=pd.Index(resolved.names, name="spatial_lag"),
    )
    result.attrs["method"] = "yule-walker"
    result.attrs["centered"] = center
    result.attrs["solvers_used"] = tuple(solvers_used)
    return result


def stpacf_regression(
    data: Any,
    weights: Any,
    *,
    max_tlag: int = 5,
    center: bool = True,
) -> pd.DataFrame:
    """Estimate a finite-sample regression analogue of the STPACF.

    This preserves the diagnostic used in pySTARMAx 0.0.1. It is useful as a
    direct projection diagnostic but is not the classical Yule-Walker STPACF.
    """
    max_tlag = validate_nonnegative_int(max_tlag, name="max_tlag")
    if max_tlag == 0:
        raise ValueError("max_tlag must be positive")
    observations = validate_time_space(data)
    resolved = coerce_weights(weights, n_locations=observations.shape[1])
    values = _center_globally(observations, center=center)
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
    result = pd.DataFrame(
        output,
        index=pd.Index(range(1, max_tlag + 1), name="temporal_lag"),
        columns=pd.Index(resolved.names, name="spatial_lag"),
    )
    result.attrs["method"] = "regression"
    result.attrs["centered"] = center
    return result


def stpacf(
    data: Any,
    weights: Any,
    *,
    max_tlag: int = 5,
    center: bool = True,
    method: STPACFMethod = "yule-walker",
    solver: YuleWalkerSolver = "auto",
) -> pd.DataFrame:
    """Estimate a space-time partial autocorrelation function.

    ``method='yule-walker'`` is the classical default. Use
    ``method='regression'`` to reproduce the projection-based diagnostic from
    pySTARMAx 0.0.1. The ``solver`` argument applies only to Yule-Walker mode.
    """
    if method == "yule-walker":
        return stpacf_yule_walker(
            data,
            weights,
            max_tlag=max_tlag,
            center=center,
            solver=solver,
        )
    if method == "regression":
        return stpacf_regression(
            data,
            weights,
            max_tlag=max_tlag,
            center=center,
        )
    raise ValueError("method must be 'yule-walker' or 'regression'")


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
