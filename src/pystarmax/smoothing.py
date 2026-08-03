# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Fixed-interval Gaussian state smoothing for Kalman STARMA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from pystarmax._validation import FloatArray
from pystarmax.state_space import KalmanFilterResult, StateSpaceModel

BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.int_]


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _freeze_int(value: Any, *, name: str, shape: tuple[int, ...]) -> IntArray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    frozen = np.ascontiguousarray(array, dtype=np.int_).copy()
    frozen.setflags(write=False)
    return cast(IntArray, frozen)


def _freeze_bool(value: Any, *, name: str, shape: tuple[int, ...]) -> BoolArray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    frozen = np.ascontiguousarray(array, dtype=bool).copy()
    frozen.setflags(write=False)
    return cast(BoolArray, frozen)


def _symmetric(value: FloatArray) -> FloatArray:
    return cast(FloatArray, 0.5 * (value + value.T))


def _project_positive_semidefinite(value: FloatArray, *, name: str) -> FloatArray:
    symmetric = _symmetric(value)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -1e-9 * scale:
        raise np.linalg.LinAlgError(f"{name} became numerically indefinite")
    projected = (eigenvectors * np.clip(eigenvalues, 0.0, np.inf)) @ eigenvectors.T
    return cast(FloatArray, _symmetric(projected))


def _right_solve_positive_semidefinite(
    left: FloatArray,
    covariance: FloatArray,
    *,
    rcond: float,
) -> tuple[FloatArray, int, bool]:
    symmetric = _symmetric(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance_scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -1e-9 * tolerance_scale:
        raise np.linalg.LinAlgError("predicted covariance is numerically indefinite")
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        np.finfo(float).tiny,
    )
    threshold = rcond * spectral_scale
    retained = eigenvalues > threshold
    rank = int(np.count_nonzero(retained))
    if rank == covariance.shape[0]:
        solved = np.linalg.solve(symmetric, left.T).T
        return cast(FloatArray, solved), rank, False
    if rank == 0:
        return np.zeros_like(left), 0, True
    retained_vectors = eigenvectors[:, retained]
    pseudoinverse = (retained_vectors / eigenvalues[retained]) @ retained_vectors.T
    return cast(FloatArray, left @ pseudoinverse), rank, True


@dataclass(frozen=True, slots=True)
class KalmanSmootherResult:
    """Immutable Rauch--Tung--Striebel fixed-interval smoothing output.

    ``lag_one_covariance[t]`` is
    ``Cov(alpha_t, alpha_(t+1) | y_1:T)``. The disturbance quantities describe
    the state-equation disturbance
    ``alpha_(t+1) - c - T alpha_t``. They are not automatically identical to
    the location-level innovation when the state selection matrix is not
    one-to-one.
    """

    filter_result: KalmanFilterResult
    smoothed_state: FloatArray
    smoothed_covariance: FloatArray
    smoothing_gain: FloatArray
    lag_one_covariance: FloatArray
    state_disturbance_mean: FloatArray
    state_disturbance_covariance: FloatArray
    prediction_rank: IntArray
    used_pseudoinverse: BoolArray

    def __post_init__(self) -> None:
        if not isinstance(self.filter_result, KalmanFilterResult):
            raise TypeError("filter_result must be a KalmanFilterResult")
        n_time = int(self.filter_result.filtered_state.shape[0])
        state_dim = self.filter_result.model.state_dim
        n_transitions = max(0, n_time - 1)
        arrays: list[FloatArray] = []
        for name, value, ndim, shape in (
            ("smoothed_state", self.smoothed_state, 2, (n_time, state_dim)),
            (
                "smoothed_covariance",
                self.smoothed_covariance,
                3,
                (n_time, state_dim, state_dim),
            ),
            (
                "smoothing_gain",
                self.smoothing_gain,
                3,
                (n_transitions, state_dim, state_dim),
            ),
            (
                "lag_one_covariance",
                self.lag_one_covariance,
                3,
                (n_transitions, state_dim, state_dim),
            ),
            (
                "state_disturbance_mean",
                self.state_disturbance_mean,
                2,
                (n_transitions, state_dim),
            ),
            (
                "state_disturbance_covariance",
                self.state_disturbance_covariance,
                3,
                (n_transitions, state_dim, state_dim),
            ),
        ):
            array = _freeze_float(value, name=name, ndim=ndim)
            if array.shape != shape:
                raise ValueError(f"{name} has an invalid shape")
            arrays.append(array)
        prediction_rank = _freeze_int(
            self.prediction_rank,
            name="prediction_rank",
            shape=(n_transitions,),
        )
        if np.any(prediction_rank < 0) or np.any(prediction_rank > state_dim):
            raise ValueError("prediction_rank is outside the state dimension")
        used_pseudoinverse = _freeze_bool(
            self.used_pseudoinverse,
            name="used_pseudoinverse",
            shape=(n_transitions,),
        )
        if not np.array_equal(used_pseudoinverse, prediction_rank < state_dim):
            raise ValueError(
                "used_pseudoinverse must identify rank-deficient predictions"
            )
        object.__setattr__(self, "smoothed_state", arrays[0])
        object.__setattr__(self, "smoothed_covariance", arrays[1])
        object.__setattr__(self, "smoothing_gain", arrays[2])
        object.__setattr__(self, "lag_one_covariance", arrays[3])
        object.__setattr__(self, "state_disturbance_mean", arrays[4])
        object.__setattr__(self, "state_disturbance_covariance", arrays[5])
        object.__setattr__(self, "prediction_rank", prediction_rank)
        object.__setattr__(self, "used_pseudoinverse", used_pseudoinverse)

    @property
    def model(self) -> StateSpaceModel:
        """State-space model used by the forward filter."""
        return self.filter_result.model

    @property
    def n_time(self) -> int:
        """Number of smoothed time points."""
        return int(self.smoothed_state.shape[0])

    @property
    def smoothed_observations(self) -> FloatArray:
        """Conditional observation means using the full data interval."""
        values = self.smoothed_state @ self.model.design.T
        return cast(FloatArray, np.ascontiguousarray(values, dtype=float))

    @property
    def smoothed_observation_covariance(self) -> FloatArray:
        """Conditional observation covariance at each time point."""
        covariance = np.empty(
            (self.n_time, self.model.n_locations, self.model.n_locations),
            dtype=float,
        )
        for time_index in range(self.n_time):
            covariance[time_index] = (
                self.model.design
                @ self.smoothed_covariance[time_index]
                @ self.model.design.T
            )
        return cast(FloatArray, np.ascontiguousarray(covariance, dtype=float))


def kalman_smoother(
    filter_result: KalmanFilterResult,
    *,
    rcond: float = 1e-10,
) -> KalmanSmootherResult:
    """Run a Rauch--Tung--Striebel fixed-interval state smoother."""
    if not isinstance(filter_result, KalmanFilterResult):
        raise TypeError("filter_result must be a KalmanFilterResult")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be positive and finite")
    model = filter_result.model
    n_time = int(filter_result.filtered_state.shape[0])
    state_dim = model.state_dim
    n_transitions = max(0, n_time - 1)
    smoothed_state = np.asarray(filter_result.filtered_state, dtype=float).copy()
    smoothed_covariance = np.asarray(
        filter_result.filtered_covariance,
        dtype=float,
    ).copy()
    smoothing_gain = np.zeros(
        (n_transitions, state_dim, state_dim),
        dtype=float,
    )
    lag_one_covariance = np.zeros_like(smoothing_gain)
    prediction_rank = np.zeros(n_transitions, dtype=np.int_)
    used_pseudoinverse = np.zeros(n_transitions, dtype=bool)

    for time_index in range(n_time - 2, -1, -1):
        state_cross_prediction = (
            filter_result.filtered_covariance[time_index] @ model.transition.T
        )
        gain, rank, used_pinv = _right_solve_positive_semidefinite(
            cast(FloatArray, state_cross_prediction),
            filter_result.predicted_covariance[time_index + 1],
            rcond=float(rcond),
        )
        smoothing_gain[time_index] = gain
        prediction_rank[time_index] = rank
        used_pseudoinverse[time_index] = used_pinv
        smoothed_state[time_index] = filter_result.filtered_state[time_index] + gain @ (
            smoothed_state[time_index + 1]
            - filter_result.predicted_state[time_index + 1]
        )
        covariance = (
            filter_result.filtered_covariance[time_index]
            + gain
            @ (
                smoothed_covariance[time_index + 1]
                - filter_result.predicted_covariance[time_index + 1]
            )
            @ gain.T
        )
        smoothed_covariance[time_index] = _project_positive_semidefinite(
            cast(FloatArray, covariance),
            name="smoothed covariance",
        )
        lag_one_covariance[time_index] = gain @ smoothed_covariance[time_index + 1]

    state_disturbance_mean = np.empty(
        (n_transitions, state_dim),
        dtype=float,
    )
    state_disturbance_covariance = np.empty(
        (n_transitions, state_dim, state_dim),
        dtype=float,
    )
    for time_index in range(n_transitions):
        state_disturbance_mean[time_index] = (
            smoothed_state[time_index + 1]
            - model.state_intercept
            - model.transition @ smoothed_state[time_index]
        )
        lag_covariance = lag_one_covariance[time_index]
        covariance = (
            smoothed_covariance[time_index + 1]
            + model.transition @ smoothed_covariance[time_index] @ model.transition.T
            - lag_covariance.T @ model.transition.T
            - model.transition @ lag_covariance
        )
        state_disturbance_covariance[time_index] = _project_positive_semidefinite(
            cast(FloatArray, covariance),
            name="state disturbance covariance",
        )

    return KalmanSmootherResult(
        filter_result=filter_result,
        smoothed_state=smoothed_state,
        smoothed_covariance=smoothed_covariance,
        smoothing_gain=smoothing_gain,
        lag_one_covariance=lag_one_covariance,
        state_disturbance_mean=state_disturbance_mean,
        state_disturbance_covariance=state_disturbance_covariance,
        prediction_rank=prediction_rank,
        used_pseudoinverse=used_pseudoinverse,
    )
