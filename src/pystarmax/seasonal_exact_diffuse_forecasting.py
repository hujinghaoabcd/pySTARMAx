# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Forecast paths for seasonal original-level exact diffuse models."""

from __future__ import annotations

from typing import cast

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.exact_diffuse import ExactDiffuseFilterResult
from pystarmax.exact_diffuse_forecasting import (
    _projected_forecast_interval,
    _simulate_state_paths,
)
from pystarmax.exact_seasonal_integrated import ExactSeasonalIntegratedStateSpace
from pystarmax.forecasting import ForecastInterval


def _validate_forecast_inputs(
    filter_result: ExactDiffuseFilterResult,
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
) -> None:
    if not isinstance(filter_result, ExactDiffuseFilterResult):
        raise TypeError("filter_result must be an ExactDiffuseFilterResult")
    if not isinstance(integrated_state_space, ExactSeasonalIntegratedStateSpace):
        raise TypeError(
            "integrated_state_space must be an "
            "ExactSeasonalIntegratedStateSpace"
        )
    if integrated_state_space.model is not filter_result.model:
        raise ValueError(
            "integrated_state_space must be the state space used by filter_result"
        )


def _transformed_observation_design(
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
) -> FloatArray:
    n_locations = integrated_state_space.model.n_locations
    offset = integrated_state_space.integration_degree * n_locations
    design = np.zeros(
        (n_locations, integrated_state_space.model.state_dim),
        dtype=float,
    )
    design[:, offset:] = integrated_state_space.transformed_model.design
    return cast(FloatArray, design)


def simulate_seasonal_exact_diffuse_forecast_paths(
    filter_result: ExactDiffuseFilterResult,
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
    *,
    steps: int,
    n_simulations: int,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate future original-level paths from the terminal posterior.

    The complete augmented state is propagated for every path. Consequently the
    ordinary-seasonal inverse differencing recursion is applied path by path
    before original-level quantiles are formed.
    """
    _validate_forecast_inputs(filter_result, integrated_state_space)
    state_paths = _simulate_state_paths(
        filter_result,
        steps=steps,
        n_simulations=n_simulations,
        random_state=random_state,
    )
    paths = state_paths @ integrated_state_space.model.design.T
    return cast(FloatArray, np.ascontiguousarray(paths, dtype=float))


def simulate_seasonal_exact_diffuse_differenced_forecast_paths(
    filter_result: ExactDiffuseFilterResult,
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
    *,
    steps: int,
    n_simulations: int,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate paths on the combined ordinary-seasonal transformed scale."""
    _validate_forecast_inputs(filter_result, integrated_state_space)
    state_paths = _simulate_state_paths(
        filter_result,
        steps=steps,
        n_simulations=n_simulations,
        random_state=random_state,
    )
    design = _transformed_observation_design(integrated_state_space)
    paths = state_paths @ design.T
    return cast(FloatArray, np.ascontiguousarray(paths, dtype=float))


def seasonal_exact_diffuse_forecast_interval(
    filter_result: ExactDiffuseFilterResult,
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
    *,
    steps: int = 1,
    level: float = 0.95,
    n_simulations: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> ForecastInterval:
    """Return a pathwise original-level seasonal exact diffuse interval."""
    _validate_forecast_inputs(filter_result, integrated_state_space)
    return _projected_forecast_interval(
        filter_result,
        integrated_state_space.model.design,
        steps=steps,
        level=level,
        n_simulations=n_simulations,
        random_state=random_state,
        method=(
            "fixed-parameter seasonal exact diffuse terminal-posterior and "
            "innovation simulation with pathwise ordinary-seasonal level "
            "restoration"
        ),
    )


def seasonal_exact_diffuse_differenced_forecast_interval(
    filter_result: ExactDiffuseFilterResult,
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
    *,
    steps: int = 1,
    level: float = 0.95,
    n_simulations: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> ForecastInterval:
    """Return an interval on the combined ordinary-seasonal transformed scale."""
    _validate_forecast_inputs(filter_result, integrated_state_space)
    return _projected_forecast_interval(
        filter_result,
        _transformed_observation_design(integrated_state_space),
        steps=steps,
        level=level,
        n_simulations=n_simulations,
        random_state=random_state,
        method=(
            "fixed-parameter seasonal exact diffuse terminal-posterior and "
            "innovation simulation on the combined ordinary-seasonal "
            "difference scale"
        ),
    )


__all__ = [
    "seasonal_exact_diffuse_differenced_forecast_interval",
    "seasonal_exact_diffuse_forecast_interval",
    "simulate_seasonal_exact_diffuse_differenced_forecast_paths",
    "simulate_seasonal_exact_diffuse_forecast_paths",
]
