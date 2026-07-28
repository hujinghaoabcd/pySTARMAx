# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Simulation utilities for STARMA processes."""

from __future__ import annotations

from typing import Any

import numpy as np

from pystarmax._validation import FloatArray, as_float_matrix, validate_nonnegative_int
from pystarmax.weights import SpatialWeights, coerce_weights


def _coefficient_matrix(value: Any, *, name: str, n_spatial: int) -> FloatArray:
    array = as_float_matrix(value, name=name)
    if array.shape[1] != n_spatial:
        raise ValueError(
            f"{name} has {array.shape[1]} spatial columns; expected {n_spatial}"
        )
    return array


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

    if isinstance(weights, SpatialWeights):
        resolved = weights
    else:
        first = np.asarray(weights)
        if first.ndim == 2:
            n_locations = first.shape[0]
        else:
            matrices = list(weights)
            if not matrices:
                raise ValueError("weights must not be empty")
            n_locations = np.asarray(matrices[0]).shape[0]
            weights = matrices
        resolved = coerce_weights(weights, n_locations=n_locations)

    ar = _coefficient_matrix(phi, name="phi", n_spatial=len(resolved))
    ma = _coefficient_matrix(theta, name="theta", n_spatial=len(resolved))
    max_lag = max(ar.shape[0], ma.shape[0])
    total = n_steps + burnin + max_lag

    if np.isscalar(innovation_covariance):
        variance = float(innovation_covariance)
        if variance <= 0:
            raise ValueError("innovation_covariance must be positive")
        covariance = np.eye(resolved.n_locations, dtype=float) * variance
    else:
        covariance = as_float_matrix(
            innovation_covariance, name="innovation_covariance"
        )
        if covariance.shape != (resolved.n_locations, resolved.n_locations):
            raise ValueError("innovation_covariance has an incompatible shape")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    innovations = rng.multivariate_normal(
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
