# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse Kalman filtering for linear Gaussian state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from pystarmax._validation import FloatArray
from pystarmax.state_space import StateSpaceModel

BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.int_]


def _freeze_float(
    value: Any,
    *,
    name: str,
    ndim: int,
    allow_nan: bool = False,
) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    valid = ~np.isinf(array) if allow_nan else np.isfinite(array)
    if not np.all(valid):
        raise ValueError(f"{name} contains invalid values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _freeze_bool(value: Any, *, name: str, shape: tuple[int, ...]) -> BoolArray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    frozen = np.ascontiguousarray(array, dtype=bool).copy()
    frozen.setflags(write=False)
    return cast(BoolArray, frozen)


def _freeze_int(value: Any, *, name: str, shape: tuple[int, ...]) -> IntArray:
    array = np.asarray(value, dtype=int)
    if array.shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    frozen = np.ascontiguousarray(array, dtype=int).copy()
    frozen.setflags(write=False)
    return cast(IntArray, frozen)


def _symmetric(value: FloatArray) -> FloatArray:
    return cast(FloatArray, 0.5 * (value + value.T))


def _positive_semidefinite(value: Any, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    symmetric = _symmetric(cast(FloatArray, array))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = (eigenvectors * np.clip(eigenvalues, 0.0, np.inf)) @ eigenvectors.T
    return cast(FloatArray, _symmetric(clipped))


def _matrix_rank(value: FloatArray, *, tolerance: float) -> int:
    eigenvalues = np.linalg.eigvalsh(_symmetric(value))
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    return int(np.count_nonzero(eigenvalues > tolerance * scale))


def _validate_observations(data: Any, *, n_locations: int) -> FloatArray:
    observations = np.asarray(data, dtype=float)
    if observations.ndim != 2:
        raise ValueError("data must be a two-dimensional array")
    if observations.shape[0] < 1:
        raise ValueError("data must contain at least one time observation")
    if observations.shape[1] != n_locations:
        raise ValueError(
            f"data contain {observations.shape[1]} locations but the model "
            f"contains {n_locations}"
        )
    if np.any(np.isinf(observations)):
        raise ValueError("data must not contain infinite values")
    return cast(FloatArray, np.ascontiguousarray(observations, dtype=float))


def _scalar_variance(
    value: float,
    *,
    scale: float,
    tolerance: float,
    name: str,
) -> float:
    resolved = float(value)
    threshold = tolerance * max(1.0, float(scale))
    if resolved < -threshold:
        raise np.linalg.LinAlgError(f"{name} became materially negative")
    if resolved < 0.0:
        return 0.0
    return resolved


@dataclass(frozen=True, slots=True)
class ExactDiffuseFilterResult:
    """Immutable output from the sequential exact diffuse Kalman filter."""

    model: StateSpaceModel
    predicted_state: FloatArray
    filtered_state: FloatArray
    predicted_covariance: FloatArray
    filtered_covariance: FloatArray
    predicted_diffuse_covariance: FloatArray
    filtered_diffuse_covariance: FloatArray
    innovations: FloatArray
    finite_innovation_variance: FloatArray
    diffuse_innovation_variance: FloatArray
    observed_mask: BoolArray
    diffuse_update_mask: BoolArray
    log_likelihood_contributions: FloatArray
    predicted_diffuse_rank: IntArray
    filtered_diffuse_rank: IntArray
    log_likelihood: float
    n_observations: int
    n_diffuse_observations: int
    initial_diffuse_rank: int
    tolerance: float

    def __post_init__(self) -> None:
        predicted_state = np.asarray(self.predicted_state, dtype=float)
        if predicted_state.ndim != 2 or predicted_state.shape[0] == 0:
            raise ValueError("predicted_state must have shape (time, state)")
        n_time = int(predicted_state.shape[0])
        state_dim = self.model.state_dim
        n_locations = self.model.n_locations
        state_shape = (n_time, state_dim)
        covariance_shape = (n_time, state_dim, state_dim)
        observation_shape = (n_time, n_locations)

        arrays: list[FloatArray] = []
        for name, value, shape, allow_nan in (
            ("predicted_state", self.predicted_state, state_shape, False),
            ("filtered_state", self.filtered_state, state_shape, False),
            (
                "predicted_covariance",
                self.predicted_covariance,
                covariance_shape,
                False,
            ),
            (
                "filtered_covariance",
                self.filtered_covariance,
                covariance_shape,
                False,
            ),
            (
                "predicted_diffuse_covariance",
                self.predicted_diffuse_covariance,
                covariance_shape,
                False,
            ),
            (
                "filtered_diffuse_covariance",
                self.filtered_diffuse_covariance,
                covariance_shape,
                False,
            ),
            ("innovations", self.innovations, observation_shape, True),
            (
                "finite_innovation_variance",
                self.finite_innovation_variance,
                observation_shape,
                True,
            ),
            (
                "diffuse_innovation_variance",
                self.diffuse_innovation_variance,
                observation_shape,
                True,
            ),
            (
                "log_likelihood_contributions",
                self.log_likelihood_contributions,
                (n_time,),
                False,
            ),
        ):
            array = _freeze_float(
                value,
                name=name,
                ndim=len(shape),
                allow_nan=allow_nan,
            )
            if array.shape != shape:
                raise ValueError(f"{name} has an invalid shape")
            arrays.append(array)

        observed_mask = _freeze_bool(
            self.observed_mask,
            name="observed_mask",
            shape=observation_shape,
        )
        diffuse_update_mask = _freeze_bool(
            self.diffuse_update_mask,
            name="diffuse_update_mask",
            shape=observation_shape,
        )
        if np.any(diffuse_update_mask & ~observed_mask):
            raise ValueError("diffuse updates must correspond to observed cells")
        predicted_rank = _freeze_int(
            self.predicted_diffuse_rank,
            name="predicted_diffuse_rank",
            shape=(n_time,),
        )
        filtered_rank = _freeze_int(
            self.filtered_diffuse_rank,
            name="filtered_diffuse_rank",
            shape=(n_time,),
        )
        if np.any(predicted_rank < 0) or np.any(filtered_rank < 0):
            raise ValueError("diffuse ranks must be non-negative")

        n_observations = int(self.n_observations)
        if n_observations != int(np.count_nonzero(observed_mask)):
            raise ValueError("n_observations must match observed_mask")
        n_diffuse = int(self.n_diffuse_observations)
        if n_diffuse != int(np.count_nonzero(diffuse_update_mask)):
            raise ValueError("n_diffuse_observations must match diffuse_update_mask")
        initial_rank = int(self.initial_diffuse_rank)
        if initial_rank < 0 or initial_rank > state_dim:
            raise ValueError("initial_diffuse_rank is outside the state dimension")
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        log_likelihood = float(self.log_likelihood)
        if not np.isfinite(log_likelihood):
            raise ValueError("log_likelihood must be finite")

        object.__setattr__(self, "predicted_state", arrays[0])
        object.__setattr__(self, "filtered_state", arrays[1])
        object.__setattr__(self, "predicted_covariance", arrays[2])
        object.__setattr__(self, "filtered_covariance", arrays[3])
        object.__setattr__(self, "predicted_diffuse_covariance", arrays[4])
        object.__setattr__(self, "filtered_diffuse_covariance", arrays[5])
        object.__setattr__(self, "innovations", arrays[6])
        object.__setattr__(self, "finite_innovation_variance", arrays[7])
        object.__setattr__(self, "diffuse_innovation_variance", arrays[8])
        object.__setattr__(self, "log_likelihood_contributions", arrays[9])
        object.__setattr__(self, "observed_mask", observed_mask)
        object.__setattr__(self, "diffuse_update_mask", diffuse_update_mask)
        object.__setattr__(self, "predicted_diffuse_rank", predicted_rank)
        object.__setattr__(self, "filtered_diffuse_rank", filtered_rank)
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "n_observations", n_observations)
        object.__setattr__(self, "n_diffuse_observations", n_diffuse)
        object.__setattr__(self, "initial_diffuse_rank", initial_rank)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def predicted_observations(self) -> FloatArray:
        """Observation predictions before sequential measurement updates."""
        values = self.predicted_state @ self.model.design.T
        return cast(FloatArray, np.ascontiguousarray(values, dtype=float))

    @property
    def filtered_observations(self) -> FloatArray:
        """Observation-scale values after all updates at each time."""
        values = self.filtered_state @ self.model.design.T
        return cast(FloatArray, np.ascontiguousarray(values, dtype=float))

    @property
    def diffuse_end_time(self) -> int | None:
        """First zero-based time index whose filtered diffuse rank is zero."""
        indices = np.flatnonzero(self.filtered_diffuse_rank == 0)
        if indices.size == 0:
            return None
        return int(indices[0])

    @property
    def final_diffuse_rank(self) -> int:
        """Diffuse rank remaining after the final observation row."""
        return int(self.filtered_diffuse_rank[-1])


def exact_diffuse_filter(
    data: Any,
    model: StateSpaceModel,
    *,
    initial_state: Any | None = None,
    initial_covariance: Any | None = None,
    initial_diffuse_covariance: Any | None = None,
    tolerance: float = 1e-10,
) -> ExactDiffuseFilterResult:
    """Filter observations using exact diffuse initialization.

    Observed locations are processed sequentially. ``initial_covariance`` is the
    finite component ``P_*`` and ``initial_diffuse_covariance`` is ``P_inf`` in
    the decomposition ``P(kappa) = P_* + kappa P_inf`` as ``kappa`` tends to
    infinity. Defaults are a zero finite covariance and an identity diffuse
    covariance over the complete state.
    """
    if not isinstance(model, StateSpaceModel):
        raise TypeError("model must be a StateSpaceModel")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    observations = _validate_observations(data, n_locations=model.n_locations)
    state_dim = model.state_dim
    n_locations = model.n_locations
    n_time = int(observations.shape[0])

    if initial_state is None:
        state = np.zeros(state_dim, dtype=float)
    else:
        state = np.asarray(initial_state, dtype=float)
        if state.shape != (state_dim,) or not np.all(np.isfinite(state)):
            raise ValueError(
                "initial_state must be finite and match the state dimension"
            )
        state = np.ascontiguousarray(state, dtype=float)

    if initial_covariance is None:
        covariance = np.zeros((state_dim, state_dim), dtype=float)
    else:
        covariance = _positive_semidefinite(
            initial_covariance,
            name="initial_covariance",
        )

    if initial_diffuse_covariance is None:
        diffuse_covariance = np.eye(state_dim, dtype=float)
    else:
        diffuse_covariance = _positive_semidefinite(
            initial_diffuse_covariance,
            name="initial_diffuse_covariance",
        )
    initial_diffuse_rank = _matrix_rank(
        diffuse_covariance,
        tolerance=tolerance,
    )

    predicted_state = np.empty((n_time, state_dim), dtype=float)
    filtered_state = np.empty_like(predicted_state)
    predicted_covariance = np.empty((n_time, state_dim, state_dim), dtype=float)
    filtered_covariance = np.empty_like(predicted_covariance)
    predicted_diffuse_covariance = np.empty_like(predicted_covariance)
    filtered_diffuse_covariance = np.empty_like(predicted_covariance)
    innovations = np.full((n_time, n_locations), np.nan, dtype=float)
    finite_variance = np.full_like(innovations, np.nan)
    diffuse_variance = np.full_like(innovations, np.nan)
    observed_mask = np.isfinite(observations)
    diffuse_update_mask = np.zeros_like(observed_mask, dtype=bool)
    contributions = np.zeros(n_time, dtype=float)
    predicted_rank = np.empty(n_time, dtype=int)
    filtered_rank = np.empty(n_time, dtype=int)
    process_covariance = model.process_covariance
    log_two_pi = float(np.log(2.0 * np.pi))

    for time_index in range(n_time):
        state = model.state_intercept + model.transition @ state
        covariance = (
            model.transition @ covariance @ model.transition.T + process_covariance
        )
        covariance = _symmetric(cast(FloatArray, covariance))
        diffuse_covariance = model.transition @ diffuse_covariance @ model.transition.T
        diffuse_covariance = _symmetric(cast(FloatArray, diffuse_covariance))

        predicted_state[time_index] = state
        predicted_covariance[time_index] = covariance
        predicted_diffuse_covariance[time_index] = diffuse_covariance
        predicted_rank[time_index] = _matrix_rank(
            diffuse_covariance,
            tolerance=tolerance,
        )

        for location_index in np.flatnonzero(observed_mask[time_index]):
            design = model.design[location_index]
            innovation = float(
                observations[time_index, location_index] - design @ state
            )
            finite_cross = covariance @ design
            diffuse_cross = diffuse_covariance @ design
            finite_scale = float(np.linalg.norm(covariance, ord=2)) * float(
                design @ design
            )
            diffuse_scale = float(np.linalg.norm(diffuse_covariance, ord=2)) * float(
                design @ design
            )
            finite_value = _scalar_variance(
                float(design @ finite_cross),
                scale=finite_scale,
                tolerance=tolerance,
                name="finite innovation variance",
            )
            diffuse_value = _scalar_variance(
                float(design @ diffuse_cross),
                scale=diffuse_scale,
                tolerance=tolerance,
                name="diffuse innovation variance",
            )
            innovations[time_index, location_index] = innovation
            finite_variance[time_index, location_index] = finite_value
            diffuse_variance[time_index, location_index] = diffuse_value

            diffuse_threshold = tolerance * max(1.0, diffuse_scale)
            finite_threshold = tolerance * max(1.0, finite_scale)
            if diffuse_value > diffuse_threshold:
                diffuse_update_mask[time_index, location_index] = True
                gain_zero = diffuse_cross / diffuse_value
                gain_one = (
                    finite_cross / diffuse_value
                    - gain_zero * finite_value / diffuse_value
                )
                state = state + gain_zero * innovation
                covariance = (
                    covariance
                    - np.outer(finite_cross, gain_zero)
                    - np.outer(diffuse_cross, gain_one)
                )
                diffuse_covariance = diffuse_covariance - np.outer(
                    diffuse_cross,
                    gain_zero,
                )
                covariance = _symmetric(cast(FloatArray, covariance))
                diffuse_covariance = _symmetric(cast(FloatArray, diffuse_covariance))
                contributions[time_index] -= 0.5 * (
                    log_two_pi + float(np.log(diffuse_value))
                )
            elif finite_value > finite_threshold:
                gain = finite_cross / finite_value
                state = state + gain * innovation
                covariance = covariance - np.outer(finite_cross, gain)
                covariance = _symmetric(cast(FloatArray, covariance))
                contributions[time_index] -= 0.5 * (
                    log_two_pi
                    + float(np.log(finite_value))
                    + innovation * innovation / finite_value
                )
            else:
                innovation_scale = max(
                    1.0,
                    abs(float(observations[time_index, location_index])),
                    abs(float(design @ state)),
                )
                if abs(innovation) > np.sqrt(tolerance) * innovation_scale:
                    raise np.linalg.LinAlgError(
                        "an observed value conflicts with a deterministic "
                        "zero-variance measurement"
                    )

        filtered_state[time_index] = state
        filtered_covariance[time_index] = covariance
        filtered_diffuse_covariance[time_index] = diffuse_covariance
        filtered_rank[time_index] = _matrix_rank(
            diffuse_covariance,
            tolerance=tolerance,
        )

    return ExactDiffuseFilterResult(
        model=model,
        predicted_state=predicted_state,
        filtered_state=filtered_state,
        predicted_covariance=predicted_covariance,
        filtered_covariance=filtered_covariance,
        predicted_diffuse_covariance=predicted_diffuse_covariance,
        filtered_diffuse_covariance=filtered_diffuse_covariance,
        innovations=innovations,
        finite_innovation_variance=finite_variance,
        diffuse_innovation_variance=diffuse_variance,
        observed_mask=observed_mask,
        diffuse_update_mask=diffuse_update_mask,
        log_likelihood_contributions=contributions,
        predicted_diffuse_rank=predicted_rank,
        filtered_diffuse_rank=filtered_rank,
        log_likelihood=float(np.sum(contributions)),
        n_observations=int(np.count_nonzero(observed_mask)),
        n_diffuse_observations=int(np.count_nonzero(diffuse_update_mask)),
        initial_diffuse_rank=initial_diffuse_rank,
        tolerance=tolerance,
    )


def exact_diffuse_loglikelihood(
    data: Any,
    model: StateSpaceModel,
    *,
    initial_state: Any | None = None,
    initial_covariance: Any | None = None,
    initial_diffuse_covariance: Any | None = None,
    tolerance: float = 1e-10,
) -> float:
    """Return the exact diffuse Gaussian log likelihood."""
    return exact_diffuse_filter(
        data,
        model,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        initial_diffuse_covariance=initial_diffuse_covariance,
        tolerance=tolerance,
    ).log_likelihood


__all__ = [
    "ExactDiffuseFilterResult",
    "exact_diffuse_filter",
    "exact_diffuse_loglikelihood",
]
