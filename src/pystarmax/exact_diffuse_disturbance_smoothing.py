# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse state and primitive innovation disturbance smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.exact_diffuse_smoothing import ExactDiffuseSmootherResult
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
class ExactDiffuseDisturbanceResult:
    """Posterior moments of primitive and state-equation disturbances.

    Transition index ``t`` represents the disturbance entering state
    ``alpha_(t+1)`` in

    ``alpha_(t+1) = c + T alpha_t + R eta_(t+1)``.

    The primitive innovation dimension is the dimension of ``Q`` and need not
    equal either the state or observation dimension.
    """

    smoother_result: ExactDiffuseSmootherResult
    innovation_mean: FloatArray
    innovation_covariance: FloatArray
    state_disturbance_mean: FloatArray
    state_disturbance_covariance: FloatArray
    maximum_innovation_covariance_correction: float
    maximum_state_covariance_correction: float
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.smoother_result, ExactDiffuseSmootherResult):
            raise TypeError("smoother_result must be an ExactDiffuseSmootherResult")
        model = self.smoother_result.filter_result.model
        n_time = int(self.smoother_result.smoothed_state.shape[0])
        n_transitions = max(0, n_time - 1)
        innovation_dim = int(model.innovation_covariance.shape[0])
        state_dim = model.state_dim

        specifications = (
            (
                "innovation_mean",
                self.innovation_mean,
                (n_transitions, innovation_dim),
            ),
            (
                "innovation_covariance",
                self.innovation_covariance,
                (n_transitions, innovation_dim, innovation_dim),
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

        innovation_correction = float(
            self.maximum_innovation_covariance_correction
        )
        state_correction = float(self.maximum_state_covariance_correction)
        if not np.isfinite(innovation_correction) or innovation_correction < 0.0:
            raise ValueError(
                "maximum_innovation_covariance_correction must be finite and "
                "non-negative"
            )
        if not np.isfinite(state_correction) or state_correction < 0.0:
            raise ValueError(
                "maximum_state_covariance_correction must be finite and non-negative"
            )
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        object.__setattr__(
            self,
            "maximum_innovation_covariance_correction",
            innovation_correction,
        )
        object.__setattr__(
            self,
            "maximum_state_covariance_correction",
            state_correction,
        )
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def model(self) -> StateSpaceModel:
        """State-space model defining ``R`` and ``Q``."""
        return self.smoother_result.filter_result.model

    @property
    def n_transitions(self) -> int:
        """Number of transition disturbances represented."""
        return int(self.innovation_mean.shape[0])

    @property
    def innovation_dim(self) -> int:
        """Dimension of the primitive process innovation."""
        return int(self.innovation_mean.shape[1])


def exact_diffuse_disturbance_smoother(
    smoother_result: ExactDiffuseSmootherResult,
    *,
    tolerance: float | None = None,
) -> ExactDiffuseDisturbanceResult:
    """Smooth primitive innovations and their state-equation images.

    For

    ``alpha_(t+1) = c + T alpha_t + R eta_(t+1)``, ``eta ~ N(0, Q)``,

    the exact diffuse information smoother gives

    ``E(eta_(t+1) | Y) = Q R' r_t``

    and

    ``Var(eta_(t+1) | Y) = Q - Q R' N_t R Q``.

    In ``ExactDiffuseSmootherResult``, index ``t+1`` stores ``r_t`` and ``N_t``.
    State disturbances are obtained exactly as ``R eta``. No lag-one state
    covariance is required, and innovation directions in the null space of
    ``R`` retain their prior uncertainty automatically.
    """
    if not isinstance(smoother_result, ExactDiffuseSmootherResult):
        raise TypeError("smoother_result must be an ExactDiffuseSmootherResult")
    if tolerance is None:
        tolerance = smoother_result.tolerance
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    model = smoother_result.filter_result.model
    n_time = int(smoother_result.smoothed_state.shape[0])
    n_transitions = max(0, n_time - 1)
    state_dim = model.state_dim
    innovation_covariance_prior = model.innovation_covariance
    innovation_dim = int(innovation_covariance_prior.shape[0])
    selection = model.selection
    process_loading_covariance = selection @ innovation_covariance_prior

    information_vector = smoother_result.scaled_smoothed_estimator[1:]
    information_covariance = (
        smoother_result.scaled_smoothed_estimator_covariance[1:]
    )
    innovation_mean = information_vector @ process_loading_covariance
    innovation_covariance = np.empty(
        (n_transitions, innovation_dim, innovation_dim),
        dtype=float,
    )
    state_disturbance_mean = innovation_mean @ selection.T
    state_disturbance_covariance = np.empty(
        (n_transitions, state_dim, state_dim),
        dtype=float,
    )
    maximum_innovation_correction = 0.0
    maximum_state_correction = 0.0

    for transition_index in range(n_transitions):
        information = information_covariance[transition_index]
        posterior_innovation_covariance = (
            innovation_covariance_prior
            - process_loading_covariance.T
            @ information
            @ process_loading_covariance
        )
        stabilized_innovation, innovation_correction = _stabilize_covariance(
            cast(FloatArray, posterior_innovation_covariance),
            name="innovation disturbance covariance",
            tolerance=tolerance,
        )
        innovation_covariance[transition_index] = stabilized_innovation
        maximum_innovation_correction = max(
            maximum_innovation_correction,
            innovation_correction,
        )

        posterior_state_covariance = (
            selection @ stabilized_innovation @ selection.T
        )
        stabilized_state, state_correction = _stabilize_covariance(
            cast(FloatArray, posterior_state_covariance),
            name="state disturbance covariance",
            tolerance=tolerance,
        )
        state_disturbance_covariance[transition_index] = stabilized_state
        maximum_state_correction = max(
            maximum_state_correction,
            state_correction,
        )

    return ExactDiffuseDisturbanceResult(
        smoother_result=smoother_result,
        innovation_mean=innovation_mean,
        innovation_covariance=innovation_covariance,
        state_disturbance_mean=state_disturbance_mean,
        state_disturbance_covariance=state_disturbance_covariance,
        maximum_innovation_covariance_correction=maximum_innovation_correction,
        maximum_state_covariance_correction=maximum_state_correction,
        tolerance=tolerance,
    )


__all__ = [
    "ExactDiffuseDisturbanceResult",
    "exact_diffuse_disturbance_smoother",
]
