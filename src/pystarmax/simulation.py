# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Simulation utilities for STARMA and integrated seasonal processes."""

from __future__ import annotations

from typing import Any

import numpy as np

from pystarmax._validation import FloatArray, as_float_matrix, validate_nonnegative_int
from pystarmax.differencing import (
    CombinedDifferencingState,
    DifferencingState,
    SeasonalDifferencingState,
)
from pystarmax.seasonal import (
    expand_multiplicative_operators,
    maximum_operator_lag,
)
from pystarmax.weights import SpatialWeights, coerce_weights


def _coefficient_matrix(value: Any, *, name: str, n_spatial: int) -> FloatArray:
    array = as_float_matrix(value, name=name)
    if array.shape[1] != n_spatial:
        raise ValueError(
            f"{name} has {array.shape[1]} spatial columns; expected {n_spatial}"
        )
    return array


def _resolve_weights(weights: Any) -> SpatialWeights:
    if isinstance(weights, SpatialWeights):
        return weights
    first = np.asarray(weights)
    if first.ndim == 2:
        n_locations = first.shape[0]
        return coerce_weights(weights, n_locations=n_locations)
    matrices = list(weights)
    if not matrices:
        raise ValueError("weights must not be empty")
    n_locations = np.asarray(matrices[0]).shape[0]
    return coerce_weights(matrices, n_locations=n_locations)


def _innovation_covariance(
    value: float | Any,
    *,
    n_locations: int,
) -> FloatArray:
    if np.isscalar(value):
        variance = float(value)
        if variance <= 0:
            raise ValueError("innovation_covariance must be positive")
        return np.eye(n_locations, dtype=float) * variance
    covariance = as_float_matrix(value, name="innovation_covariance")
    if covariance.shape != (n_locations, n_locations):
        raise ValueError("innovation_covariance has an incompatible shape")
    return covariance


def _rng(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    return (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )


def simulate_starma(
    *,
    phi: Any,
    theta: Any,
    weights: Any,
    n_steps: int,
    burnin: int = 200,
    intercept: float = 0.0,
    innovation_covariance: float | Any = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate a STARMA process using the pySTARMAx sign convention."""
    n_steps = validate_nonnegative_int(n_steps, name="n_steps")
    burnin = validate_nonnegative_int(burnin, name="burnin")
    if n_steps == 0:
        raise ValueError("n_steps must be positive")

    resolved = _resolve_weights(weights)
    ar = _coefficient_matrix(phi, name="phi", n_spatial=len(resolved))
    ma = _coefficient_matrix(theta, name="theta", n_spatial=len(resolved))
    max_lag = max(ar.shape[0], ma.shape[0])
    total = n_steps + burnin + max_lag

    covariance = _innovation_covariance(
        innovation_covariance, n_locations=resolved.n_locations
    )
    innovations = _rng(random_state).multivariate_normal(
        np.zeros(resolved.n_locations, dtype=float), covariance, size=total
    )
    series = np.zeros((total, resolved.n_locations), dtype=float)
    for time_index in range(max_lag, total):
        value = np.full(resolved.n_locations, float(intercept), dtype=float)
        for temporal_lag in range(1, ar.shape[0] + 1):
            for spatial_lag, matrix in enumerate(resolved):
                value += ar[temporal_lag - 1, spatial_lag] * (
                    matrix @ series[time_index - temporal_lag]
                )
        value += innovations[time_index]
        for temporal_lag in range(1, ma.shape[0] + 1):
            for spatial_lag, matrix in enumerate(resolved):
                value += ma[temporal_lag - 1, spatial_lag] * (
                    matrix @ innovations[time_index - temporal_lag]
                )
        series[time_index] = value
    start = max_lag + burnin
    return series[start : start + n_steps]


def simulate_starima(
    *,
    phi: Any,
    theta: Any,
    weights: Any,
    n_steps: int,
    integration_order: int = 1,
    burnin: int = 200,
    intercept: float = 0.0,
    innovation_covariance: float | Any = 1.0,
    random_state: int | np.random.Generator | None = None,
    initial_state: DifferencingState | None = None,
) -> FloatArray:
    """Simulate an ordinary integrated STARMA process."""
    integration_order = validate_nonnegative_int(
        integration_order, name="integration_order"
    )
    differenced = simulate_starma(
        phi=phi,
        theta=theta,
        weights=weights,
        n_steps=n_steps,
        burnin=burnin,
        intercept=intercept,
        innovation_covariance=innovation_covariance,
        random_state=random_state,
    )
    if initial_state is None:
        initial_state = DifferencingState(
            order=integration_order,
            n_locations=differenced.shape[1],
            anchors=tuple(
                np.zeros(differenced.shape[1], dtype=float)
                for _ in range(integration_order)
            ),
        )
    elif initial_state.order != integration_order:
        raise ValueError("initial_state order must match integration_order")
    elif initial_state.n_locations != differenced.shape[1]:
        raise ValueError("initial_state locations must match the simulated process")
    return initial_state.inverse_forecast(differenced)


def simulate_seasonal_starma(
    *,
    phi: Any,
    seasonal_phi: Any,
    theta: Any,
    seasonal_theta: Any,
    weights: Any,
    seasonal_period: int,
    n_steps: int,
    burnin: int = 200,
    intercept: float = 0.0,
    innovation_covariance: float | Any = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate a stationary multiplicative seasonal STARMA process."""
    n_steps = validate_nonnegative_int(n_steps, name="n_steps")
    burnin = validate_nonnegative_int(burnin, name="burnin")
    seasonal_period = validate_nonnegative_int(
        seasonal_period, name="seasonal_period"
    )
    if n_steps == 0:
        raise ValueError("n_steps must be positive")
    if seasonal_period == 0:
        raise ValueError("seasonal_period must be positive")

    resolved = _resolve_weights(weights)
    ar = _coefficient_matrix(phi, name="phi", n_spatial=len(resolved))
    sar = _coefficient_matrix(
        seasonal_phi, name="seasonal_phi", n_spatial=len(resolved)
    )
    ma = _coefficient_matrix(theta, name="theta", n_spatial=len(resolved))
    sma = _coefficient_matrix(
        seasonal_theta, name="seasonal_theta", n_spatial=len(resolved)
    )
    ar_terms = expand_multiplicative_operators(
        ar,
        sar,
        resolved,
        seasonal_period=seasonal_period,
        kind="ar",
    )
    ma_terms = expand_multiplicative_operators(
        ma,
        sma,
        resolved,
        seasonal_period=seasonal_period,
        kind="ma",
    )
    max_lag = max(
        maximum_operator_lag(ar_terms),
        maximum_operator_lag(ma_terms),
    )
    total = n_steps + burnin + max_lag
    covariance = _innovation_covariance(
        innovation_covariance, n_locations=resolved.n_locations
    )
    innovations = _rng(random_state).multivariate_normal(
        np.zeros(resolved.n_locations, dtype=float), covariance, size=total
    )
    series = np.zeros((total, resolved.n_locations), dtype=float)
    for time_index in range(max_lag, total):
        value = np.full(resolved.n_locations, float(intercept), dtype=float)
        for operator in ar_terms:
            value += operator.matrix @ series[time_index - operator.lag]
        value += innovations[time_index]
        for operator in ma_terms:
            value += operator.matrix @ innovations[time_index - operator.lag]
        series[time_index] = value
    start = max_lag + burnin
    return series[start : start + n_steps]


def simulate_seasonal_starima(
    *,
    phi: Any,
    seasonal_phi: Any,
    theta: Any,
    seasonal_theta: Any,
    weights: Any,
    seasonal_period: int,
    n_steps: int,
    integration_order: int = 0,
    seasonal_integration_order: int = 1,
    burnin: int = 200,
    intercept: float = 0.0,
    innovation_covariance: float | Any = 1.0,
    random_state: int | np.random.Generator | None = None,
    initial_state: CombinedDifferencingState | None = None,
) -> FloatArray:
    """Simulate a multiplicative seasonal STARIMA process."""
    integration_order = validate_nonnegative_int(
        integration_order, name="integration_order"
    )
    seasonal_integration_order = validate_nonnegative_int(
        seasonal_integration_order, name="seasonal_integration_order"
    )
    seasonal_period = validate_nonnegative_int(
        seasonal_period, name="seasonal_period"
    )
    if seasonal_period == 0:
        raise ValueError("seasonal_period must be positive")

    transformed = simulate_seasonal_starma(
        phi=phi,
        seasonal_phi=seasonal_phi,
        theta=theta,
        seasonal_theta=seasonal_theta,
        weights=weights,
        seasonal_period=seasonal_period,
        n_steps=n_steps,
        burnin=burnin,
        intercept=intercept,
        innovation_covariance=innovation_covariance,
        random_state=random_state,
    )
    if initial_state is None:
        n_locations = transformed.shape[1]
        initial_state = CombinedDifferencingState(
            ordinary=DifferencingState(
                order=integration_order,
                n_locations=n_locations,
                anchors=tuple(
                    np.zeros(n_locations, dtype=float)
                    for _ in range(integration_order)
                ),
            ),
            seasonal=SeasonalDifferencingState(
                order=seasonal_integration_order,
                period=seasonal_period,
                n_locations=n_locations,
                histories=tuple(
                    np.zeros((seasonal_period, n_locations), dtype=float)
                    for _ in range(seasonal_integration_order)
                ),
            ),
        )
    if initial_state.ordinary_order != integration_order:
        raise ValueError(
            "initial_state ordinary order must match integration_order"
        )
    if initial_state.seasonal_order != seasonal_integration_order:
        raise ValueError(
            "initial_state seasonal order must match seasonal_integration_order"
        )
    if initial_state.seasonal_period != seasonal_period:
        raise ValueError(
            "initial_state seasonal period must match seasonal_period"
        )
    if initial_state.n_locations != transformed.shape[1]:
        raise ValueError("initial_state locations must match the simulated process")
    return initial_state.inverse_forecast(transformed)
