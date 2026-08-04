# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fixed-parameter forecast paths from an exact diffuse terminal posterior."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.exact_diffuse import ExactDiffuseFilterResult
from pystarmax.forecasting import (
    ForecastInterval,
    draw_innovations,
    interval_from_paths,
    random_generator,
    validate_interval_arguments,
)


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


def _require_proper_terminal(filter_result: ExactDiffuseFilterResult) -> None:
    if not isinstance(filter_result, ExactDiffuseFilterResult):
        raise TypeError("filter_result must be an ExactDiffuseFilterResult")
    if filter_result.final_diffuse_rank != 0:
        raise RuntimeError(
            "forecast simulation requires a proper terminal posterior with zero "
            "remaining diffuse rank"
        )


def _forecast_state_mean(
    filter_result: ExactDiffuseFilterResult,
    *,
    steps: int,
) -> FloatArray:
    model = filter_result.model
    state = filter_result.filtered_state[-1].copy()
    states = np.empty((steps, model.state_dim), dtype=float)
    for step_index in range(steps):
        state = model.state_intercept + model.transition @ state
        states[step_index] = state
    return cast(FloatArray, np.ascontiguousarray(states, dtype=float))


def _simulate_state_paths(
    filter_result: ExactDiffuseFilterResult,
    *,
    steps: int,
    n_simulations: int,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    steps = validate_nonnegative_int(steps, name="steps")
    if steps == 0:
        raise ValueError("steps must be positive")
    n_simulations = validate_nonnegative_int(
        n_simulations,
        name="n_simulations",
    )
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least two")
    _require_proper_terminal(filter_result)

    generator = random_generator(random_state)
    model = filter_result.model
    state_factor = _positive_semidefinite_factor(
        filter_result.filtered_covariance[-1],
        name="final finite filtered covariance",
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
        (n_simulations, steps, model.state_dim),
        dtype=float,
    )
    for step_index in range(steps):
        states = (
            model.state_intercept[None, :]
            + states @ model.transition.T
            + innovations[:, step_index, :] @ model.selection.T
        )
        paths[:, step_index, :] = states
    return cast(FloatArray, np.ascontiguousarray(paths, dtype=float))


def _projected_forecast_interval(
    filter_result: ExactDiffuseFilterResult,
    observation_design: Any,
    *,
    steps: int = 1,
    level: float = 0.95,
    n_simulations: int = 2000,
    random_state: int | np.random.Generator | None = None,
    method: str,
) -> ForecastInterval:
    steps, level, n_simulations = validate_interval_arguments(
        steps=steps,
        level=level,
        n_simulations=n_simulations,
    )
    _require_proper_terminal(filter_result)
    design = np.asarray(observation_design, dtype=float)
    expected_columns = filter_result.model.state_dim
    if design.ndim != 2 or design.shape[0] < 1:
        raise ValueError("observation_design must be a non-empty matrix")
    if design.shape[1] != expected_columns:
        raise ValueError("observation_design columns must match the state dimension")
    if not np.all(np.isfinite(design)):
        raise ValueError("observation_design must contain only finite values")

    state_paths = _simulate_state_paths(
        filter_result,
        steps=steps,
        n_simulations=n_simulations,
        random_state=random_state,
    )
    paths = state_paths @ design.T
    mean = _forecast_state_mean(filter_result, steps=steps) @ design.T
    return interval_from_paths(
        mean=mean,
        paths=paths,
        level=level,
        method=method,
    )


def simulate_exact_diffuse_forecast_paths(
    filter_result: ExactDiffuseFilterResult,
    *,
    steps: int,
    n_simulations: int,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate future observations after exact diffuse uncertainty is resolved.

    The initial state is drawn from the final finite filtered Gaussian
    posterior. Future state transitions receive primitive process innovations
    from the fitted covariance. A nonzero final diffuse rank is rejected because
    it does not define a proper terminal Gaussian distribution.
    """
    state_paths = _simulate_state_paths(
        filter_result,
        steps=steps,
        n_simulations=n_simulations,
        random_state=random_state,
    )
    observations = state_paths @ filter_result.model.design.T
    return cast(FloatArray, np.ascontiguousarray(observations, dtype=float))


def exact_diffuse_forecast_interval(
    filter_result: ExactDiffuseFilterResult,
    *,
    steps: int = 1,
    level: float = 0.95,
    n_simulations: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> ForecastInterval:
    """Return an original-scale fixed-parameter exact diffuse forecast interval."""
    return _projected_forecast_interval(
        filter_result,
        filter_result.model.design,
        steps=steps,
        level=level,
        n_simulations=n_simulations,
        random_state=random_state,
        method=(
            "fixed-parameter exact diffuse terminal-posterior and innovation "
            "simulation"
        ),
    )


__all__ = [
    "exact_diffuse_forecast_interval",
    "simulate_exact_diffuse_forecast_paths",
]
