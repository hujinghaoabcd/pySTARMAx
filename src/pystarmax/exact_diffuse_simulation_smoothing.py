# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse conditional simulation of complete state paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.exact_diffuse import ExactDiffuseFilterResult
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _positive_semidefinite_factor(
    value: Any,
    *,
    name: str,
    tolerance: float,
) -> FloatArray:
    covariance = np.asarray(value, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain only finite values")
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    threshold = tolerance * scale
    minimum = float(np.min(eigenvalues, initial=0.0))
    if minimum < -100.0 * threshold:
        raise np.linalg.LinAlgError(f"{name} became materially indefinite")
    retained = eigenvalues > threshold
    factor = eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])
    return cast(FloatArray, np.ascontiguousarray(factor, dtype=float))


def _diffuse_basis(value: Any, *, tolerance: float) -> FloatArray:
    covariance = np.asarray(value, dtype=float)
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    threshold = tolerance * scale
    if float(np.min(eigenvalues, initial=0.0)) < -100.0 * threshold:
        raise np.linalg.LinAlgError("predicted diffuse covariance is indefinite")
    retained = eigenvalues > threshold
    basis = eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])
    return cast(FloatArray, np.ascontiguousarray(basis, dtype=float))


def _reconstruct_observations(
    filter_result: ExactDiffuseFilterResult,
    *,
    tolerance: float,
) -> FloatArray:
    model = filter_result.model
    n_time = filter_result.predicted_state.shape[0]
    observations = np.full((n_time, model.n_locations), np.nan, dtype=float)

    for time_index in range(n_time):
        state = filter_result.predicted_state[time_index].copy()
        finite = filter_result.predicted_covariance[time_index].copy()
        diffuse = filter_result.predicted_diffuse_covariance[time_index].copy()
        for location_index in np.flatnonzero(filter_result.observed_mask[time_index]):
            design = model.design[location_index]
            innovation = float(filter_result.innovations[time_index, location_index])
            observations[time_index, location_index] = innovation + design @ state
            finite_cross = finite @ design
            diffuse_cross = diffuse @ design
            finite_value = float(
                filter_result.finite_innovation_variance[
                    time_index,
                    location_index,
                ]
            )
            diffuse_value = float(
                filter_result.diffuse_innovation_variance[
                    time_index,
                    location_index,
                ]
            )
            diffuse_scale = float(np.linalg.norm(diffuse, ord=2)) * float(
                design @ design
            )
            finite_scale = float(np.linalg.norm(finite, ord=2)) * float(design @ design)
            if diffuse_value > tolerance * max(1.0, diffuse_scale):
                gain_zero = diffuse_cross / diffuse_value
                gain_one = (
                    finite_cross / diffuse_value
                    - gain_zero * finite_value / diffuse_value
                )
                state = state + gain_zero * innovation
                finite = (
                    finite
                    - np.outer(finite_cross, gain_zero)
                    - np.outer(diffuse_cross, gain_one)
                )
                diffuse = diffuse - np.outer(diffuse_cross, gain_zero)
            elif finite_value > tolerance * max(1.0, finite_scale):
                gain = finite_cross / finite_value
                state = state + gain * innovation
                finite = finite - np.outer(finite_cross, gain)
            finite = 0.5 * (finite + finite.T)
            diffuse = 0.5 * (diffuse + diffuse.T)

    return cast(FloatArray, np.ascontiguousarray(observations, dtype=float))


def _source_loadings(
    filter_result: ExactDiffuseFilterResult,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    model = filter_result.model
    n_time = filter_result.predicted_state.shape[0]
    state_dim = model.state_dim
    diffuse_initial = _diffuse_basis(
        filter_result.predicted_diffuse_covariance[0],
        tolerance=tolerance,
    )
    finite_initial = _positive_semidefinite_factor(
        filter_result.predicted_covariance[0],
        name="first predicted finite covariance",
        tolerance=tolerance,
    )
    innovation_factor = _positive_semidefinite_factor(
        model.innovation_covariance,
        name="innovation covariance",
        tolerance=tolerance,
    )
    n_diffuse = diffuse_initial.shape[1]
    n_finite = finite_initial.shape[1]
    innovation_rank = innovation_factor.shape[1]
    n_proper = n_finite + max(0, n_time - 1) * innovation_rank

    base = np.empty((n_time, state_dim), dtype=float)
    diffuse = np.zeros((n_time, state_dim, n_diffuse), dtype=float)
    proper = np.zeros((n_time, state_dim, n_proper), dtype=float)
    base[0] = filter_result.predicted_state[0]
    diffuse[0] = diffuse_initial
    proper[0, :, :n_finite] = finite_initial
    process_loading = model.selection @ innovation_factor

    for time_index in range(1, n_time):
        base[time_index] = (
            model.state_intercept + model.transition @ base[time_index - 1]
        )
        diffuse[time_index] = model.transition @ diffuse[time_index - 1]
        proper[time_index] = model.transition @ proper[time_index - 1]
        start = n_finite + (time_index - 1) * innovation_rank
        stop = start + innovation_rank
        proper[time_index, :, start:stop] = process_loading

    return (
        cast(FloatArray, np.ascontiguousarray(base, dtype=float)),
        cast(FloatArray, np.ascontiguousarray(diffuse, dtype=float)),
        cast(FloatArray, np.ascontiguousarray(proper, dtype=float)),
    )


def _full_column_pseudoinverse(
    value: FloatArray,
    *,
    rcond: float,
) -> tuple[FloatArray, FloatArray, int]:
    n_rows, n_columns = value.shape
    if n_columns == 0:
        return (
            np.zeros((0, n_rows), dtype=float),
            np.eye(n_rows, dtype=float),
            0,
        )
    left, singular_values, right_t = np.linalg.svd(value, full_matrices=True)
    scale = max(1.0, float(np.max(singular_values, initial=0.0)))
    rank = int(np.count_nonzero(singular_values > rcond * scale))
    if rank != n_columns:
        raise RuntimeError(
            "observations do not identify every initial diffuse direction"
        )
    inverse = (right_t[:rank].T * (1.0 / singular_values[:rank])) @ left[:, :rank].T
    left_null = left[:, rank:].T
    return (
        cast(FloatArray, np.ascontiguousarray(inverse, dtype=float)),
        cast(FloatArray, np.ascontiguousarray(left_null, dtype=float)),
        rank,
    )


def _condition_standard_normal(
    matrix: FloatArray,
    target: FloatArray,
    *,
    rcond: float,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, int, float]:
    n_constraints, n_variables = matrix.shape
    if target.shape != (n_constraints,):
        raise ValueError("conditioning target has an invalid shape")
    if n_constraints == 0:
        return (
            np.zeros(n_variables, dtype=float),
            np.eye(n_variables, dtype=float),
            0,
            0.0,
        )
    if n_variables == 0:
        residual = float(np.max(np.abs(target), initial=0.0))
        if residual > tolerance * max(1.0, float(np.max(np.abs(target), initial=0.0))):
            raise np.linalg.LinAlgError(
                "observations conflict with the deterministic exact state path"
            )
        return (
            np.zeros(0, dtype=float),
            np.zeros((0, 0), dtype=float),
            0,
            residual,
        )

    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=True)
    scale = max(1.0, float(np.max(singular_values, initial=0.0)))
    rank = int(np.count_nonzero(singular_values > rcond * scale))
    if rank:
        projected = left[:, :rank].T @ target
        mean = right_t[:rank].T @ (projected / singular_values[:rank])
        supported = left[:, :rank] @ projected
    else:
        mean = np.zeros(n_variables, dtype=float)
        supported = np.zeros(n_constraints, dtype=float)
    residual = float(np.max(np.abs(target - supported), initial=0.0))
    support_scale = max(1.0, float(np.max(np.abs(target), initial=0.0)))
    if residual > 100.0 * tolerance * support_scale:
        raise np.linalg.LinAlgError(
            "observations lie outside the support of the exact Gaussian model"
        )
    null_basis = right_t[rank:].T
    return (
        cast(FloatArray, np.ascontiguousarray(mean, dtype=float)),
        cast(FloatArray, np.ascontiguousarray(null_basis, dtype=float)),
        rank,
        residual,
    )


@dataclass(frozen=True, slots=True)
class ExactDiffuseSimulationSmootherResult:
    """Conditional draws from the complete exact diffuse state path."""

    smoother_result: ExactDiffuseSmootherResult
    state_paths: FloatArray
    observation_paths: FloatArray
    posterior_state_mean: FloatArray
    posterior_state_covariance: FloatArray
    diffuse_rank: int
    conditioning_rank: int
    proper_source_dimension: int
    posterior_source_dimension: int
    maximum_constraint_residual: float
    maximum_mean_discrepancy: float
    maximum_covariance_discrepancy: float
    rcond: float
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.smoother_result, ExactDiffuseSmootherResult):
            raise TypeError("smoother_result must be an ExactDiffuseSmootherResult")
        filter_result = self.smoother_result.filter_result
        n_time = filter_result.predicted_state.shape[0]
        state_dim = filter_result.model.state_dim
        n_locations = filter_result.model.n_locations
        n_simulations = int(np.asarray(self.state_paths).shape[0])
        specifications = (
            (
                "state_paths",
                self.state_paths,
                (n_simulations, n_time, state_dim),
            ),
            (
                "observation_paths",
                self.observation_paths,
                (n_simulations, n_time, n_locations),
            ),
            (
                "posterior_state_mean",
                self.posterior_state_mean,
                (n_time, state_dim),
            ),
            (
                "posterior_state_covariance",
                self.posterior_state_covariance,
                (n_time, state_dim, state_dim),
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
        if n_simulations < 2:
            raise ValueError("state_paths must contain at least two simulations")
        for name in (
            "diffuse_rank",
            "conditioning_rank",
            "proper_source_dimension",
            "posterior_source_dimension",
        ):
            value = validate_nonnegative_int(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        for name in (
            "maximum_constraint_residual",
            "maximum_mean_discrepancy",
            "maximum_covariance_discrepancy",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        rcond = float(self.rcond)
        tolerance = float(self.tolerance)
        if not np.isfinite(rcond) or rcond <= 0.0:
            raise ValueError("rcond must be positive and finite")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        object.__setattr__(self, "rcond", rcond)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def n_simulations(self) -> int:
        """Number of conditional paths."""
        return int(self.state_paths.shape[0])

    @property
    def filter_result(self) -> ExactDiffuseFilterResult:
        """Exact diffuse filter result defining the conditional problem."""
        return self.smoother_result.filter_result


def exact_diffuse_simulation_smoother(
    filter_result: ExactDiffuseFilterResult,
    *,
    n_simulations: int = 1000,
    random_state: int | np.random.Generator | None = None,
    rcond: float = 1e-10,
    tolerance: float | None = None,
) -> ExactDiffuseSimulationSmootherResult:
    """Draw complete state paths from the exact diffuse posterior.

    The first predicted state is represented as a flat diffuse coordinate plus
    finite standard-normal coordinates. Later process innovations add further
    standard-normal coordinates. Observed cells impose exact linear constraints.
    Diffuse coordinates are eliminated analytically; the remaining proper
    Gaussian source vector is sampled from its exact conditional distribution.
    No finite diffuse scale and no lag-one smoothing recursion are used.
    """
    if not isinstance(filter_result, ExactDiffuseFilterResult):
        raise TypeError("filter_result must be an ExactDiffuseFilterResult")
    n_simulations = validate_nonnegative_int(
        n_simulations,
        name="n_simulations",
    )
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least two")
    rcond = float(rcond)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be positive and finite")
    if tolerance is None:
        tolerance = filter_result.tolerance
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if filter_result.final_diffuse_rank != 0:
        raise RuntimeError(
            "simulation smoothing requires every diffuse direction to be "
            "identified by the observed sample"
        )

    smoother = exact_diffuse_smoother(filter_result, tolerance=tolerance)
    model = filter_result.model
    n_time = filter_result.predicted_state.shape[0]
    state_dim = model.state_dim
    base, diffuse_loading, proper_loading = _source_loadings(
        filter_result,
        tolerance=tolerance,
    )
    observations = _reconstruct_observations(
        filter_result,
        tolerance=tolerance,
    )
    observed_indices = np.argwhere(filter_result.observed_mask)
    n_observed = int(observed_indices.shape[0])
    diffuse_rank = int(diffuse_loading.shape[2])
    proper_dimension = int(proper_loading.shape[2])
    diffuse_design = np.empty((n_observed, diffuse_rank), dtype=float)
    proper_design = np.empty((n_observed, proper_dimension), dtype=float)
    target = np.empty(n_observed, dtype=float)

    for row_index, (time_index, location_index) in enumerate(observed_indices):
        time_index = int(time_index)
        location_index = int(location_index)
        design = model.design[location_index]
        diffuse_design[row_index] = design @ diffuse_loading[time_index]
        proper_design[row_index] = design @ proper_loading[time_index]
        target[row_index] = (
            observations[time_index, location_index] - design @ base[time_index]
        )

    diffuse_inverse, left_null, identified_rank = _full_column_pseudoinverse(
        cast(FloatArray, diffuse_design),
        rcond=rcond,
    )
    condition_matrix = left_null @ proper_design
    condition_target = left_null @ target
    proper_mean, posterior_basis, conditioning_rank, support_residual = (
        _condition_standard_normal(
            cast(FloatArray, condition_matrix),
            cast(FloatArray, condition_target),
            rcond=rcond,
            tolerance=tolerance,
        )
    )

    flat_base = base.reshape(n_time * state_dim)
    flat_diffuse = diffuse_loading.reshape(n_time * state_dim, diffuse_rank)
    flat_proper = proper_loading.reshape(n_time * state_dim, proper_dimension)
    diffuse_offset = diffuse_inverse @ target
    effective_loading = flat_proper - flat_diffuse @ diffuse_inverse @ proper_design
    posterior_mean_flat = (
        flat_base + flat_diffuse @ diffuse_offset + effective_loading @ proper_mean
    )
    posterior_loading = effective_loading @ posterior_basis
    posterior_state_mean = posterior_mean_flat.reshape(n_time, state_dim)
    posterior_state_covariance = np.empty(
        (n_time, state_dim, state_dim),
        dtype=float,
    )
    for time_index in range(n_time):
        start = time_index * state_dim
        stop = start + state_dim
        loading = posterior_loading[start:stop]
        posterior_state_covariance[time_index] = loading @ loading.T

    generator = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    standard = generator.standard_normal((n_simulations, posterior_basis.shape[1]))
    state_paths = (
        posterior_mean_flat[None, :] + standard @ posterior_loading.T
    ).reshape(n_simulations, n_time, state_dim)
    observation_paths = state_paths @ model.design.T

    constraint_residual = 0.0
    if n_observed:
        simulated_observed = observation_paths[
            :,
            observed_indices[:, 0],
            observed_indices[:, 1],
        ]
        target_observed = observations[
            observed_indices[:, 0],
            observed_indices[:, 1],
        ]
        constraint_residual = float(
            np.max(np.abs(simulated_observed - target_observed[None, :]), initial=0.0)
        )
    mean_discrepancy = float(
        np.max(
            np.abs(posterior_state_mean - smoother.smoothed_state),
            initial=0.0,
        )
    )
    covariance_discrepancy = float(
        np.max(
            np.abs(posterior_state_covariance - smoother.smoothed_covariance),
            initial=0.0,
        )
    )
    scale = max(
        1.0,
        float(np.max(np.abs(smoother.smoothed_state), initial=0.0)),
        float(np.max(np.abs(smoother.smoothed_covariance), initial=0.0)),
    )
    if mean_discrepancy > 1000.0 * tolerance * scale:
        raise np.linalg.LinAlgError(
            "dense conditional mean disagrees with the exact information smoother"
        )
    if covariance_discrepancy > 1000.0 * tolerance * scale:
        raise np.linalg.LinAlgError(
            "dense conditional covariance disagrees with the exact information smoother"
        )

    return ExactDiffuseSimulationSmootherResult(
        smoother_result=smoother,
        state_paths=state_paths,
        observation_paths=observation_paths,
        posterior_state_mean=posterior_state_mean,
        posterior_state_covariance=posterior_state_covariance,
        diffuse_rank=identified_rank,
        conditioning_rank=conditioning_rank,
        proper_source_dimension=proper_dimension,
        posterior_source_dimension=int(posterior_basis.shape[1]),
        maximum_constraint_residual=max(constraint_residual, support_residual),
        maximum_mean_discrepancy=mean_discrepancy,
        maximum_covariance_discrepancy=covariance_discrepancy,
        rcond=rcond,
        tolerance=tolerance,
    )


__all__ = [
    "ExactDiffuseSimulationSmootherResult",
    "exact_diffuse_simulation_smoother",
]
