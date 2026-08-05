# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Dense exact-diffuse lag-one state covariance reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.exact_diffuse import ExactDiffuseFilterResult
from pystarmax.exact_diffuse_disturbance_smoothing import (
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_simulation_smoothing import (
    _condition_standard_normal,
    _full_column_pseudoinverse,
    _reconstruct_observations,
    _source_loadings,
)
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.state_space import StateSpaceModel


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _symmetric(value: FloatArray) -> FloatArray:
    return cast(FloatArray, 0.5 * (value + value.T))


def _stabilize_covariance(
    value: FloatArray,
    *,
    name: str,
    tolerance: float,
) -> tuple[FloatArray, float]:
    symmetric = _symmetric(value)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    threshold = tolerance * scale
    minimum = float(np.min(eigenvalues, initial=0.0))
    if minimum < -100.0 * threshold:
        raise np.linalg.LinAlgError(f"{name} became materially indefinite")
    clipped = np.clip(eigenvalues, 0.0, np.inf)
    stabilized = (eigenvectors * clipped) @ eigenvectors.T
    stabilized = _symmetric(cast(FloatArray, stabilized))
    correction = float(np.max(np.abs(stabilized - symmetric), initial=0.0))
    return stabilized, correction


@dataclass(frozen=True, slots=True)
class ExactDiffuseLagOneCovarianceResult:
    """Exact adjacent-time posterior covariance for a diffuse state path.

    ``lag_one_covariance[t]`` is
    ``Cov(alpha_t, alpha_(t+1) | y_1:T)``. The orientation matches
    :class:`pystarmax.smoothing.KalmanSmootherResult`.
    """

    smoother_result: ExactDiffuseSmootherResult
    lag_one_covariance: FloatArray
    observation_lag_one_covariance: FloatArray
    state_disturbance_mean: FloatArray
    state_disturbance_covariance: FloatArray
    diffuse_rank: int
    conditioning_rank: int
    proper_source_dimension: int
    posterior_source_dimension: int
    maximum_source_support_residual: float
    maximum_mean_discrepancy: float
    maximum_marginal_covariance_discrepancy: float
    maximum_state_disturbance_discrepancy: float
    maximum_state_covariance_correction: float
    rcond: float
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.smoother_result, ExactDiffuseSmootherResult):
            raise TypeError("smoother_result must be an ExactDiffuseSmootherResult")
        model = self.smoother_result.filter_result.model
        n_time = int(self.smoother_result.smoothed_state.shape[0])
        n_transitions = max(0, n_time - 1)
        state_dim = model.state_dim
        n_locations = model.n_locations
        specifications = (
            (
                "lag_one_covariance",
                self.lag_one_covariance,
                (n_transitions, state_dim, state_dim),
            ),
            (
                "observation_lag_one_covariance",
                self.observation_lag_one_covariance,
                (n_transitions, n_locations, n_locations),
            ),
            (
                "state_disturbance_mean",
                self.state_disturbance_mean,
                (n_transitions, state_dim),
            ),
            (
                "state_disturbance_covariance",
                self.state_disturbance_covariance,
                (n_transitions, state_dim, state_dim),
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

        for name in (
            "diffuse_rank",
            "conditioning_rank",
            "proper_source_dimension",
            "posterior_source_dimension",
        ):
            value = validate_nonnegative_int(getattr(self, name), name=name)
            object.__setattr__(self, name, value)

        for name in (
            "maximum_source_support_residual",
            "maximum_mean_discrepancy",
            "maximum_marginal_covariance_discrepancy",
            "maximum_state_disturbance_discrepancy",
            "maximum_state_covariance_correction",
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
    def filter_result(self) -> ExactDiffuseFilterResult:
        """Exact diffuse filter result defining the posterior."""
        return self.smoother_result.filter_result

    @property
    def model(self) -> StateSpaceModel:
        """State-space model defining the adjacent-time covariance."""
        return self.filter_result.model

    @property
    def n_transitions(self) -> int:
        """Number of adjacent state pairs represented."""
        return int(self.lag_one_covariance.shape[0])


def exact_diffuse_lag_one_covariance(
    filter_result: ExactDiffuseFilterResult,
    *,
    rcond: float = 1e-10,
    tolerance: float | None = None,
) -> ExactDiffuseLagOneCovarianceResult:
    """Return exact adjacent-time smoothed state covariance.

    The complete state path is represented as

    ``alpha = b + D delta + G xi``,

    where ``delta`` contains flat diffuse coordinates and ``xi`` is a proper
    standard-normal source vector. Exact observations impose

    ``A delta + B xi = c``.

    Every diffuse direction must be identified, so ``A`` has full column rank.
    Eliminating ``delta`` leaves a proper Gaussian equality constraint on
    ``xi``. If ``V`` spans its null space, the conditional state loading is

    ``H = (G - D A^+ B) V``

    and therefore

    ``Cov(alpha_t, alpha_s | Y) = H_t H_s'``.

    This routine evaluates only adjacent pairs. It is an exact dense reference:
    no finite diffuse scale, Monte Carlo approximation, or unavailable diffuse
    lag recursion is used.
    """
    if not isinstance(filter_result, ExactDiffuseFilterResult):
        raise TypeError("filter_result must be an ExactDiffuseFilterResult")
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
            "lag-one covariance requires every diffuse direction to be "
            "identified by the observed sample"
        )

    smoother = exact_diffuse_smoother(filter_result, tolerance=tolerance)
    model = filter_result.model
    n_time = int(filter_result.predicted_state.shape[0])
    state_dim = model.state_dim
    n_transitions = max(0, n_time - 1)
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
    lag_one_covariance = np.empty(
        (n_transitions, state_dim, state_dim),
        dtype=float,
    )

    for time_index in range(n_time):
        start = time_index * state_dim
        stop = start + state_dim
        loading = posterior_loading[start:stop]
        posterior_state_covariance[time_index] = loading @ loading.T
        if time_index < n_transitions:
            next_loading = posterior_loading[stop : stop + state_dim]
            lag_one_covariance[time_index] = loading @ next_loading.T

    observation_lag_one_covariance = np.empty(
        (n_transitions, model.n_locations, model.n_locations),
        dtype=float,
    )
    state_disturbance_mean = np.empty(
        (n_transitions, state_dim),
        dtype=float,
    )
    state_disturbance_covariance = np.empty(
        (n_transitions, state_dim, state_dim),
        dtype=float,
    )
    maximum_state_correction = 0.0

    for time_index in range(n_transitions):
        lag_covariance = lag_one_covariance[time_index]
        observation_lag_one_covariance[time_index] = (
            model.design @ lag_covariance @ model.design.T
        )
        state_disturbance_mean[time_index] = (
            posterior_state_mean[time_index + 1]
            - model.state_intercept
            - model.transition @ posterior_state_mean[time_index]
        )
        covariance = (
            posterior_state_covariance[time_index + 1]
            + model.transition
            @ posterior_state_covariance[time_index]
            @ model.transition.T
            - lag_covariance.T @ model.transition.T
            - model.transition @ lag_covariance
        )
        stabilized, correction = _stabilize_covariance(
            cast(FloatArray, covariance),
            name="state disturbance covariance reconstructed from lag-one moments",
            tolerance=tolerance,
        )
        state_disturbance_covariance[time_index] = stabilized
        maximum_state_correction = max(maximum_state_correction, correction)

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
    direct_disturbances = exact_diffuse_disturbance_smoother(
        smoother,
        tolerance=tolerance,
    )
    disturbance_discrepancy = float(
        np.max(
            np.abs(
                state_disturbance_covariance
                - direct_disturbances.state_disturbance_covariance
            ),
            initial=0.0,
        )
    )
    scale = max(
        1.0,
        float(np.max(np.abs(smoother.smoothed_state), initial=0.0)),
        float(np.max(np.abs(smoother.smoothed_covariance), initial=0.0)),
        float(
            np.max(
                np.abs(direct_disturbances.state_disturbance_covariance),
                initial=0.0,
            )
        ),
    )
    threshold = 1000.0 * tolerance * scale
    if mean_discrepancy > threshold:
        raise np.linalg.LinAlgError(
            "dense conditional mean disagrees with the exact information smoother"
        )
    if covariance_discrepancy > threshold:
        raise np.linalg.LinAlgError(
            "dense marginal covariance disagrees with the exact information smoother"
        )
    if disturbance_discrepancy > threshold:
        raise np.linalg.LinAlgError(
            "lag-one covariance disagrees with direct exact disturbance smoothing"
        )

    return ExactDiffuseLagOneCovarianceResult(
        smoother_result=smoother,
        lag_one_covariance=lag_one_covariance,
        observation_lag_one_covariance=observation_lag_one_covariance,
        state_disturbance_mean=state_disturbance_mean,
        state_disturbance_covariance=state_disturbance_covariance,
        diffuse_rank=identified_rank,
        conditioning_rank=conditioning_rank,
        proper_source_dimension=proper_dimension,
        posterior_source_dimension=int(posterior_basis.shape[1]),
        maximum_source_support_residual=support_residual,
        maximum_mean_discrepancy=mean_discrepancy,
        maximum_marginal_covariance_discrepancy=covariance_discrepancy,
        maximum_state_disturbance_discrepancy=disturbance_discrepancy,
        maximum_state_covariance_correction=maximum_state_correction,
        rcond=rcond,
        tolerance=tolerance,
    )


__all__ = [
    "ExactDiffuseLagOneCovarianceResult",
    "exact_diffuse_lag_one_covariance",
]
