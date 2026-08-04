# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse fixed-interval state smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.exact_diffuse import ExactDiffuseFilterResult


def _freeze_float(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> FloatArray:
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
    tolerance: float,
) -> tuple[FloatArray, float]:
    symmetric = _symmetric(value)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    threshold = tolerance * scale
    minimum = float(np.min(eigenvalues, initial=0.0))
    if minimum < -100.0 * threshold:
        raise np.linalg.LinAlgError(
            "smoothed covariance became materially indefinite"
        )
    clipped = np.clip(eigenvalues, 0.0, np.inf)
    stabilized = (eigenvectors * clipped) @ eigenvectors.T
    stabilized = _symmetric(cast(FloatArray, stabilized))
    correction = float(np.max(np.abs(stabilized - symmetric), initial=0.0))
    return stabilized, correction


@dataclass(frozen=True, slots=True)
class ExactDiffuseSmootherResult:
    """Immutable exact diffuse state-smoothing output."""

    filter_result: ExactDiffuseFilterResult
    smoothed_state: FloatArray
    smoothed_covariance: FloatArray
    smoothed_observations: FloatArray
    smoothed_observation_covariance: FloatArray
    scaled_smoothed_estimator: FloatArray
    scaled_smoothed_diffuse_estimator: FloatArray
    scaled_smoothed_estimator_covariance: FloatArray
    scaled_smoothed_diffuse1_estimator_covariance: FloatArray
    scaled_smoothed_diffuse2_estimator_covariance: FloatArray
    maximum_covariance_correction: float
    maximum_filter_reconstruction_error: float
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.filter_result, ExactDiffuseFilterResult):
            raise TypeError("filter_result must be an ExactDiffuseFilterResult")
        n_time = self.filter_result.predicted_state.shape[0]
        state_dim = self.filter_result.model.state_dim
        n_locations = self.filter_result.model.n_locations
        state_shape = (n_time, state_dim)
        covariance_shape = (n_time, state_dim, state_dim)
        observation_shape = (n_time, n_locations)

        specifications = (
            ("smoothed_state", self.smoothed_state, state_shape),
            ("smoothed_covariance", self.smoothed_covariance, covariance_shape),
            (
                "smoothed_observations",
                self.smoothed_observations,
                observation_shape,
            ),
            (
                "smoothed_observation_covariance",
                self.smoothed_observation_covariance,
                (n_time, n_locations, n_locations),
            ),
            (
                "scaled_smoothed_estimator",
                self.scaled_smoothed_estimator,
                state_shape,
            ),
            (
                "scaled_smoothed_diffuse_estimator",
                self.scaled_smoothed_diffuse_estimator,
                state_shape,
            ),
            (
                "scaled_smoothed_estimator_covariance",
                self.scaled_smoothed_estimator_covariance,
                covariance_shape,
            ),
            (
                "scaled_smoothed_diffuse1_estimator_covariance",
                self.scaled_smoothed_diffuse1_estimator_covariance,
                covariance_shape,
            ),
            (
                "scaled_smoothed_diffuse2_estimator_covariance",
                self.scaled_smoothed_diffuse2_estimator_covariance,
                covariance_shape,
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

        maximum_correction = float(self.maximum_covariance_correction)
        if not np.isfinite(maximum_correction) or maximum_correction < 0.0:
            raise ValueError(
                "maximum_covariance_correction must be finite and non-negative"
            )
        maximum_reconstruction = float(self.maximum_filter_reconstruction_error)
        if not np.isfinite(maximum_reconstruction) or maximum_reconstruction < 0.0:
            raise ValueError(
                "maximum_filter_reconstruction_error must be finite and non-negative"
            )
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        object.__setattr__(
            self,
            "maximum_covariance_correction",
            maximum_correction,
        )
        object.__setattr__(
            self,
            "maximum_filter_reconstruction_error",
            maximum_reconstruction,
        )
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def diffuse_end_time(self) -> int | None:
        """Forward-filter diffuse completion time."""
        return self.filter_result.diffuse_end_time

    @property
    def final_diffuse_rank(self) -> int:
        """Diffuse rank remaining after filtering."""
        return self.filter_result.final_diffuse_rank


def _forward_measurement_records(
    filter_result: ExactDiffuseFilterResult,
    time_index: int,
    *,
    tolerance: float,
) -> tuple[
    list[
        tuple[
            int,
            FloatArray,
            FloatArray,
            float,
            float,
            FloatArray,
            FloatArray,
        ]
    ],
    float,
]:
    model = filter_result.model
    finite = filter_result.predicted_covariance[time_index].copy()
    diffuse = filter_result.predicted_diffuse_covariance[time_index].copy()
    identity = np.eye(model.state_dim, dtype=float)
    records: list[
        tuple[
            int,
            FloatArray,
            FloatArray,
            float,
            float,
            FloatArray,
            FloatArray,
        ]
    ] = []

    for location_index in np.flatnonzero(filter_result.observed_mask[time_index]):
        design = model.design[location_index]
        finite_cross = cast(FloatArray, finite @ design)
        diffuse_cross = cast(FloatArray, diffuse @ design)
        finite_variance = float(
            filter_result.finite_innovation_variance[
                time_index,
                location_index,
            ]
        )
        diffuse_variance = float(
            filter_result.diffuse_innovation_variance[
                time_index,
                location_index,
            ]
        )
        diffuse_scale = float(np.linalg.norm(diffuse, ord=2)) * float(
            design @ design
        )
        finite_scale = float(np.linalg.norm(finite, ord=2)) * float(
            design @ design
        )
        if diffuse_variance > tolerance * max(1.0, diffuse_scale):
            gain_zero = diffuse_cross / diffuse_variance
            gain_one = (
                finite_cross / diffuse_variance
                - gain_zero * finite_variance / diffuse_variance
            )
            l_zero = identity - np.outer(gain_zero, design)
            l_one = -np.outer(gain_one, design)
            finite = (
                finite
                - np.outer(finite_cross, gain_zero)
                - np.outer(diffuse_cross, gain_one)
            )
            diffuse = diffuse - np.outer(diffuse_cross, gain_zero)
        elif finite_variance > tolerance * max(1.0, finite_scale):
            gain_zero = finite_cross / finite_variance
            l_zero = identity - np.outer(gain_zero, design)
            l_one = np.zeros_like(identity)
            finite = finite - np.outer(finite_cross, gain_zero)
        else:
            l_zero = identity.copy()
            l_one = np.zeros_like(identity)
        finite = _symmetric(cast(FloatArray, finite))
        diffuse = _symmetric(cast(FloatArray, diffuse))
        records.append(
            (
                int(location_index),
                cast(FloatArray, design),
                finite_cross,
                finite_variance,
                diffuse_variance,
                cast(FloatArray, l_zero),
                cast(FloatArray, l_one),
            )
        )

    finite_error = float(
        np.max(
            np.abs(finite - filter_result.filtered_covariance[time_index]),
            initial=0.0,
        )
    )
    diffuse_error = float(
        np.max(
            np.abs(
                diffuse
                - filter_result.filtered_diffuse_covariance[time_index]
            ),
            initial=0.0,
        )
    )
    return records, max(finite_error, diffuse_error)


def exact_diffuse_smoother(
    filter_result: ExactDiffuseFilterResult,
    *,
    tolerance: float | None = None,
) -> ExactDiffuseSmootherResult:
    """Return exact diffuse fixed-interval state moments.

    The backward recursion follows the sequential exact diffuse information
    smoother with ordinary estimators ``r, N`` and diffuse estimators
    ``r_inf, N1, N2``. Lag-one state covariance is deliberately not returned:
    the diffuse phase requires an additional higher-order transition term that
    is not present in the filtering result.
    """
    if not isinstance(filter_result, ExactDiffuseFilterResult):
        raise TypeError("filter_result must be an ExactDiffuseFilterResult")
    if tolerance is None:
        tolerance = filter_result.tolerance
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    model = filter_result.model
    n_time = filter_result.predicted_state.shape[0]
    state_dim = model.state_dim
    n_locations = model.n_locations
    r = np.zeros(state_dim, dtype=float)
    r_diffuse = np.zeros(state_dim, dtype=float)
    n_matrix = np.zeros((state_dim, state_dim), dtype=float)
    n_diffuse1 = np.zeros((state_dim, state_dim), dtype=float)
    n_diffuse2 = np.zeros((state_dim, state_dim), dtype=float)

    smoothed_state = np.empty((n_time, state_dim), dtype=float)
    smoothed_covariance = np.empty((n_time, state_dim, state_dim), dtype=float)
    scaled_estimator = np.empty_like(smoothed_state)
    scaled_diffuse_estimator = np.empty_like(smoothed_state)
    scaled_estimator_covariance = np.empty_like(smoothed_covariance)
    scaled_diffuse1_covariance = np.empty_like(smoothed_covariance)
    scaled_diffuse2_covariance = np.empty_like(smoothed_covariance)
    maximum_correction = 0.0
    maximum_reconstruction_error = 0.0

    for time_index in range(n_time - 1, -1, -1):
        records, reconstruction_error = _forward_measurement_records(
            filter_result,
            time_index,
            tolerance=tolerance,
        )
        maximum_reconstruction_error = max(
            maximum_reconstruction_error,
            reconstruction_error,
        )
        for (
            location_index,
            design,
            _finite_cross,
            finite_variance,
            diffuse_variance,
            l_zero,
            l_one,
        ) in reversed(records):
            innovation = float(
                filter_result.innovations[time_index, location_index]
            )
            diffuse_scale = max(1.0, abs(diffuse_variance))
            finite_scale = max(1.0, abs(finite_variance))
            if diffuse_variance > tolerance * diffuse_scale:
                f_one = 1.0 / diffuse_variance
                f_two = -finite_variance / (diffuse_variance**2)
                previous_r = r.copy()
                previous_r_diffuse = r_diffuse.copy()
                previous_n = n_matrix.copy()
                previous_n_diffuse1 = n_diffuse1.copy()
                previous_n_diffuse2 = n_diffuse2.copy()

                r_diffuse = (
                    design * innovation * f_one
                    + l_zero.T @ previous_r_diffuse
                    + l_one.T @ previous_r
                )
                r = l_zero.T @ previous_r
                n_diffuse2 = (
                    np.outer(design, design) * f_two
                    + l_zero.T @ previous_n_diffuse2 @ l_zero
                    + l_zero.T @ previous_n_diffuse1 @ l_one
                    + l_one.T @ previous_n_diffuse1.T @ l_zero
                    + l_one.T @ previous_n @ l_one
                )
                n_diffuse1 = (
                    np.outer(design, design) * f_one
                    + l_zero.T @ previous_n_diffuse1 @ l_zero
                    + l_one.T @ previous_n @ l_zero
                )
                n_matrix = l_zero.T @ previous_n @ l_zero
            elif finite_variance > tolerance * finite_scale:
                previous_r = r.copy()
                previous_n = n_matrix.copy()
                r = (
                    design * innovation / finite_variance
                    + l_zero.T @ previous_r
                )
                n_matrix = (
                    np.outer(design, design) / finite_variance
                    + l_zero.T @ previous_n @ l_zero
                )
                n_diffuse1 = n_diffuse1 @ l_zero

        n_matrix = _symmetric(cast(FloatArray, n_matrix))
        n_diffuse2 = _symmetric(cast(FloatArray, n_diffuse2))
        scaled_estimator[time_index] = r
        scaled_diffuse_estimator[time_index] = r_diffuse
        scaled_estimator_covariance[time_index] = n_matrix
        scaled_diffuse1_covariance[time_index] = n_diffuse1
        scaled_diffuse2_covariance[time_index] = n_diffuse2

        predicted_state = filter_result.predicted_state[time_index]
        finite_covariance = filter_result.predicted_covariance[time_index]
        diffuse_covariance = filter_result.predicted_diffuse_covariance[time_index]
        smoothed_state[time_index] = (
            predicted_state
            + finite_covariance @ r
            + diffuse_covariance @ r_diffuse
        )
        covariance = (
            finite_covariance
            - finite_covariance @ n_matrix @ finite_covariance
            - diffuse_covariance @ n_diffuse1 @ finite_covariance
            - finite_covariance @ n_diffuse1.T @ diffuse_covariance
            - diffuse_covariance @ n_diffuse2 @ diffuse_covariance
        )
        stabilized, correction = _stabilize_covariance(
            cast(FloatArray, covariance),
            tolerance=tolerance,
        )
        smoothed_covariance[time_index] = stabilized
        maximum_correction = max(maximum_correction, correction)

        if time_index > 0:
            transition = model.transition
            r = transition.T @ r
            r_diffuse = transition.T @ r_diffuse
            n_matrix = transition.T @ n_matrix @ transition
            n_diffuse1 = transition.T @ n_diffuse1 @ transition
            n_diffuse2 = transition.T @ n_diffuse2 @ transition

    smoothed_observations = smoothed_state @ model.design.T
    smoothed_observation_covariance = np.empty(
        (n_time, n_locations, n_locations),
        dtype=float,
    )
    for time_index in range(n_time):
        smoothed_observation_covariance[time_index] = (
            model.design
            @ smoothed_covariance[time_index]
            @ model.design.T
        )

    return ExactDiffuseSmootherResult(
        filter_result=filter_result,
        smoothed_state=smoothed_state,
        smoothed_covariance=smoothed_covariance,
        smoothed_observations=smoothed_observations,
        smoothed_observation_covariance=smoothed_observation_covariance,
        scaled_smoothed_estimator=scaled_estimator,
        scaled_smoothed_diffuse_estimator=scaled_diffuse_estimator,
        scaled_smoothed_estimator_covariance=scaled_estimator_covariance,
        scaled_smoothed_diffuse1_estimator_covariance=scaled_diffuse1_covariance,
        scaled_smoothed_diffuse2_estimator_covariance=scaled_diffuse2_covariance,
        maximum_covariance_correction=maximum_correction,
        maximum_filter_reconstruction_error=maximum_reconstruction_error,
        tolerance=tolerance,
    )


__all__ = ["ExactDiffuseSmootherResult", "exact_diffuse_smoother"]
