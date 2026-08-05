# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Conditional simulation smoothing for seasonal exact diffuse models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.exact_diffuse import ExactDiffuseFilterResult
from pystarmax.exact_diffuse_simulation_smoothing import (
    ExactDiffuseSimulationSmootherResult,
    exact_diffuse_simulation_smoother,
)
from pystarmax.exact_diffuse_smoothing import ExactDiffuseSmootherResult
from pystarmax.exact_seasonal_integrated import ExactSeasonalIntegratedStateSpace


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _validate_inputs(
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


def _transformed_offset(
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
) -> int:
    return (
        integrated_state_space.integration_degree
        * integrated_state_space.model.n_locations
    )


@dataclass(frozen=True, slots=True)
class SeasonalExactDiffuseSimulationSmootherResult:
    """Conditional seasonal state paths and transformed-state projections."""

    simulation_result: ExactDiffuseSimulationSmootherResult
    integrated_state_space: ExactSeasonalIntegratedStateSpace
    transformed_state_paths: FloatArray
    transformed_observation_paths: FloatArray
    posterior_transformed_state_mean: FloatArray
    posterior_transformed_state_covariance: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(
            self.simulation_result,
            ExactDiffuseSimulationSmootherResult,
        ):
            raise TypeError(
                "simulation_result must be an "
                "ExactDiffuseSimulationSmootherResult"
            )
        if not isinstance(
            self.integrated_state_space,
            ExactSeasonalIntegratedStateSpace,
        ):
            raise TypeError(
                "integrated_state_space must be an "
                "ExactSeasonalIntegratedStateSpace"
            )
        if (
            self.integrated_state_space.model
            is not self.simulation_result.filter_result.model
        ):
            raise ValueError(
                "integrated_state_space must be the state space used by "
                "simulation_result"
            )

        base = self.simulation_result
        transformed_model = self.integrated_state_space.transformed_model
        n_simulations = base.n_simulations
        n_time = base.state_paths.shape[1]
        transformed_dim = transformed_model.state_dim
        n_locations = transformed_model.n_locations
        specifications = (
            (
                "transformed_state_paths",
                self.transformed_state_paths,
                (n_simulations, n_time, transformed_dim),
            ),
            (
                "transformed_observation_paths",
                self.transformed_observation_paths,
                (n_simulations, n_time, n_locations),
            ),
            (
                "posterior_transformed_state_mean",
                self.posterior_transformed_state_mean,
                (n_time, transformed_dim),
            ),
            (
                "posterior_transformed_state_covariance",
                self.posterior_transformed_state_covariance,
                (n_time, transformed_dim, transformed_dim),
            ),
        )
        frozen: dict[str, FloatArray] = {}
        for name, value, shape in specifications:
            array = _freeze_float(value, name=name, ndim=len(shape))
            if array.shape != shape:
                raise ValueError(f"{name} has an invalid shape")
            frozen[name] = array
        for name, array in frozen.items():
            object.__setattr__(self, name, array)

        offset = _transformed_offset(self.integrated_state_space)
        expected_state_paths = base.state_paths[..., offset:]
        expected_observation_paths = (
            expected_state_paths @ transformed_model.design.T
        )
        expected_mean = base.posterior_state_mean[:, offset:]
        expected_covariance = base.posterior_state_covariance[
            :,
            offset:,
            offset:,
        ]
        scale = max(
            1.0,
            float(np.max(np.abs(expected_state_paths), initial=0.0)),
            float(np.max(np.abs(expected_observation_paths), initial=0.0)),
            float(np.max(np.abs(expected_covariance), initial=0.0)),
        )
        atol = 100.0 * base.tolerance * scale
        checks = (
            (
                "transformed_state_paths",
                self.transformed_state_paths,
                expected_state_paths,
            ),
            (
                "transformed_observation_paths",
                self.transformed_observation_paths,
                expected_observation_paths,
            ),
            (
                "posterior_transformed_state_mean",
                self.posterior_transformed_state_mean,
                expected_mean,
            ),
            (
                "posterior_transformed_state_covariance",
                self.posterior_transformed_state_covariance,
                expected_covariance,
            ),
        )
        for name, actual, expected in checks:
            if not np.allclose(actual, expected, rtol=0.0, atol=atol):
                raise ValueError(
                    f"{name} must equal the transformed-state projection"
                )

    @property
    def n_simulations(self) -> int:
        """Number of conditional paths."""
        return self.simulation_result.n_simulations

    @property
    def filter_result(self) -> ExactDiffuseFilterResult:
        """Exact diffuse filter result defining the posterior."""
        return self.simulation_result.filter_result

    @property
    def smoother_result(self) -> ExactDiffuseSmootherResult:
        """Exact diffuse marginal smoother used for verification."""
        return self.simulation_result.smoother_result

    @property
    def state_paths(self) -> FloatArray:
        """Complete seasonal augmented-state paths."""
        return self.simulation_result.state_paths

    @property
    def observation_paths(self) -> FloatArray:
        """Original-level observation paths."""
        return self.simulation_result.observation_paths

    @property
    def posterior_state_mean(self) -> FloatArray:
        """Posterior mean of the complete augmented state."""
        return self.simulation_result.posterior_state_mean

    @property
    def posterior_state_covariance(self) -> FloatArray:
        """Posterior marginal covariance of the complete augmented state."""
        return self.simulation_result.posterior_state_covariance

    @property
    def diffuse_rank(self) -> int:
        """Number of analytically eliminated initial diffuse coordinates."""
        return self.simulation_result.diffuse_rank

    @property
    def conditioning_rank(self) -> int:
        """Rank of the proper Gaussian conditioning system."""
        return self.simulation_result.conditioning_rank

    @property
    def proper_source_dimension(self) -> int:
        """Dimension of the pre-conditioning proper source vector."""
        return self.simulation_result.proper_source_dimension

    @property
    def posterior_source_dimension(self) -> int:
        """Dimension of the sampled posterior source vector."""
        return self.simulation_result.posterior_source_dimension

    @property
    def maximum_constraint_residual(self) -> float:
        """Largest observed-cell conditioning discrepancy."""
        return self.simulation_result.maximum_constraint_residual

    @property
    def maximum_mean_discrepancy(self) -> float:
        """Largest difference from the information-smoother mean."""
        return self.simulation_result.maximum_mean_discrepancy

    @property
    def maximum_covariance_discrepancy(self) -> float:
        """Largest difference from information-smoother marginal covariance."""
        return self.simulation_result.maximum_covariance_discrepancy

    @property
    def rcond(self) -> float:
        """Rank threshold used by the dense conditioning system."""
        return self.simulation_result.rcond

    @property
    def tolerance(self) -> float:
        """Numerical tolerance used by filtering and verification."""
        return self.simulation_result.tolerance


def seasonal_exact_diffuse_simulation_smoother(
    filter_result: ExactDiffuseFilterResult,
    integrated_state_space: ExactSeasonalIntegratedStateSpace,
    *,
    n_simulations: int = 1000,
    random_state: int | np.random.Generator | None = None,
    rcond: float = 1e-10,
    tolerance: float | None = None,
) -> SeasonalExactDiffuseSimulationSmootherResult:
    """Draw complete conditional paths for a seasonal exact diffuse model.

    The generic dense exact-diffuse source-conditioning algorithm is applied to
    the complete seasonal augmented state. Original-level paths are conditioned
    exactly on every observed cell. The stationary transformed-state block is
    then projected from the same draws; no second seasonal conditioning
    algorithm and no finite diffuse approximation are introduced.
    """
    _validate_inputs(filter_result, integrated_state_space)
    simulation = exact_diffuse_simulation_smoother(
        filter_result,
        n_simulations=n_simulations,
        random_state=random_state,
        rcond=rcond,
        tolerance=tolerance,
    )
    offset = _transformed_offset(integrated_state_space)
    transformed_state_paths = simulation.state_paths[..., offset:]
    transformed_observation_paths = (
        transformed_state_paths @ integrated_state_space.transformed_model.design.T
    )
    posterior_transformed_state_mean = simulation.posterior_state_mean[:, offset:]
    posterior_transformed_state_covariance = simulation.posterior_state_covariance[
        :,
        offset:,
        offset:,
    ]
    return SeasonalExactDiffuseSimulationSmootherResult(
        simulation_result=simulation,
        integrated_state_space=integrated_state_space,
        transformed_state_paths=transformed_state_paths,
        transformed_observation_paths=transformed_observation_paths,
        posterior_transformed_state_mean=posterior_transformed_state_mean,
        posterior_transformed_state_covariance=posterior_transformed_state_covariance,
    )


__all__ = [
    "SeasonalExactDiffuseSimulationSmootherResult",
    "seasonal_exact_diffuse_simulation_smoother",
]
