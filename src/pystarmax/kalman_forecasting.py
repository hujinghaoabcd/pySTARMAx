# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fixed-parameter Gaussian forecast paths for fitted Kalman models."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np

from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.forecasting import (
    ForecastInterval,
    draw_innovations,
    interval_from_paths,
    random_generator,
    validate_interval_arguments,
)
from pystarmax.state_space import KalmanFilterResult


class ForecastInverter(Protocol):
    """Structural interface for ordinary or combined inverse differencing."""

    @property
    def n_locations(self) -> int: ...

    def inverse_forecast(self, differenced_forecast: Any) -> FloatArray: ...


def _positive_semidefinite_factor(value: Any, *, name: str) -> FloatArray:
    covariance = np.asarray(value, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain only finite values")
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    tolerance = 100.0 * np.finfo(float).eps * scale
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, np.inf)))
    return cast(FloatArray, np.ascontiguousarray(factor, dtype=float))


def _forecast_mean(
    filter_result: KalmanFilterResult,
    *,
    steps: int,
) -> FloatArray:
    model = filter_result.model
    state = filter_result.filtered_state[-1].copy()
    mean = np.empty((steps, model.n_locations), dtype=float)
    for step_index in range(steps):
        state = model.state_intercept + model.transition @ state
        mean[step_index] = model.design @ state
    return cast(FloatArray, np.ascontiguousarray(mean, dtype=float))


def simulate_kalman_forecast_paths(
    filter_result: KalmanFilterResult,
    *,
    steps: int,
    n_simulations: int,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate future observation paths conditional on fitted parameters.

    Initial states are drawn from the final filtered Gaussian posterior. Each
    future transition then receives a location-level innovation drawn from the
    fitted innovation covariance. The returned array has shape
    ``(simulations, steps, locations)``.
    """
    steps = validate_nonnegative_int(steps, name="steps")
    if steps == 0:
        raise ValueError("steps must be positive")
    n_simulations = validate_nonnegative_int(
        n_simulations,
        name="n_simulations",
    )
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least two")
    if not isinstance(filter_result, KalmanFilterResult):
        raise TypeError("filter_result must be a KalmanFilterResult")

    generator = random_generator(random_state)
    model = filter_result.model
    state_factor = _positive_semidefinite_factor(
        filter_result.filtered_covariance[-1],
        name="final filtered covariance",
    )
    state_offsets = (
        generator.standard_normal((n_simulations, model.state_dim)) @ state_factor.T
    )
    states = filter_result.filtered_state[-1][None, :] + state_offsets
    innovations = draw_innovations(
        model.innovation_covariance,
        n_simulations=n_simulations,
        steps=steps,
        random_state=generator,
    )
    paths = np.empty(
        (n_simulations, steps, model.n_locations),
        dtype=float,
    )
    for step_index in range(steps):
        states = (
            model.state_intercept[None, :]
            + states @ model.transition.T
            + innovations[:, step_index, :] @ model.selection.T
        )
        paths[:, step_index, :] = states @ model.design.T
    return cast(FloatArray, np.ascontiguousarray(paths, dtype=float))


def inverse_forecast_paths(
    differencing_state: ForecastInverter,
    paths: Any,
) -> FloatArray:
    """Inverse-difference every simulated transformed forecast path."""
    values = np.asarray(paths, dtype=float)
    if values.ndim != 3:
        raise ValueError("paths must have shape (simulations, steps, locations)")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("paths must contain simulations and forecast steps")
    if values.shape[2] != int(differencing_state.n_locations):
        raise ValueError("path locations do not match the differencing state")
    if not np.all(np.isfinite(values)):
        raise ValueError("paths must contain only finite values")

    restored = np.empty_like(values, dtype=float)
    for simulation_index, path in enumerate(values):
        restored[simulation_index] = differencing_state.inverse_forecast(path)
    return cast(FloatArray, np.ascontiguousarray(restored, dtype=float))


def kalman_forecast_interval(
    filter_result: KalmanFilterResult,
    *,
    steps: int = 1,
    level: float = 0.95,
    n_simulations: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> ForecastInterval:
    """Return a transformed-scale fixed-parameter Gaussian forecast interval."""
    steps, level, n_simulations = validate_interval_arguments(
        steps=steps,
        level=level,
        n_simulations=n_simulations,
    )
    paths = simulate_kalman_forecast_paths(
        filter_result,
        steps=steps,
        n_simulations=n_simulations,
        random_state=random_state,
    )
    return interval_from_paths(
        mean=_forecast_mean(filter_result, steps=steps),
        paths=paths,
        level=level,
        method=("fixed-parameter Gaussian filtered-state and innovation simulation"),
    )


def integrated_kalman_forecast_interval(
    filter_result: KalmanFilterResult,
    differencing_state: ForecastInverter,
    *,
    steps: int = 1,
    level: float = 0.95,
    n_simulations: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> ForecastInterval:
    """Return an original-scale interval using pathwise inverse differencing."""
    steps, level, n_simulations = validate_interval_arguments(
        steps=steps,
        level=level,
        n_simulations=n_simulations,
    )
    transformed_paths = simulate_kalman_forecast_paths(
        filter_result,
        steps=steps,
        n_simulations=n_simulations,
        random_state=random_state,
    )
    original_paths = inverse_forecast_paths(
        differencing_state,
        transformed_paths,
    )
    transformed_mean = _forecast_mean(filter_result, steps=steps)
    original_mean = differencing_state.inverse_forecast(transformed_mean)
    return interval_from_paths(
        mean=original_mean,
        paths=original_paths,
        level=level,
        method=(
            "fixed-parameter Gaussian filtered-state and innovation simulation "
            "with pathwise inverse differencing"
        ),
    )


__all__ = [
    "ForecastInverter",
    "integrated_kalman_forecast_interval",
    "inverse_forecast_paths",
    "kalman_forecast_interval",
    "simulate_kalman_forecast_paths",
]
