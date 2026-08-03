# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Original innovation disturbance smoothing from RTS state moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.smoothing import KalmanSmootherResult
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


def _project_positive_semidefinite(value: FloatArray, *, name: str) -> FloatArray:
    symmetric = _symmetric(value)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -1e-9 * scale:
        raise np.linalg.LinAlgError(f"{name} became numerically indefinite")
    projected = (eigenvectors * np.clip(eigenvalues, 0.0, np.inf)) @ eigenvectors.T
    return cast(FloatArray, _symmetric(projected))


def _conditioning_map(
    model: StateSpaceModel,
    *,
    rcond: float,
) -> tuple[FloatArray, FloatArray, FloatArray, int, bool]:
    selection = model.selection
    innovation_covariance = model.innovation_covariance
    process_covariance = model.process_covariance
    symmetric = _symmetric(process_covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -1e-9 * scale:
        raise np.linalg.LinAlgError("process covariance is numerically indefinite")
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        np.finfo(float).tiny,
    )
    retained = eigenvalues > rcond * spectral_scale
    rank = int(np.count_nonzero(retained))
    cross_covariance = innovation_covariance @ selection.T
    state_dim = model.state_dim

    if rank == state_dim:
        mapping = np.linalg.solve(symmetric, cross_covariance.T).T
        projector = np.eye(state_dim, dtype=float)
        used_pseudoinverse = False
    elif rank == 0:
        mapping = np.zeros((model.n_locations, state_dim), dtype=float)
        projector = np.zeros((state_dim, state_dim), dtype=float)
        used_pseudoinverse = True
    else:
        retained_vectors = eigenvectors[:, retained]
        pseudoinverse = (
            retained_vectors / eigenvalues[retained]
        ) @ retained_vectors.T
        mapping = cross_covariance @ pseudoinverse
        projector = retained_vectors @ retained_vectors.T
        used_pseudoinverse = True

    unresolved_covariance = innovation_covariance - mapping @ cross_covariance.T
    unresolved_covariance = _project_positive_semidefinite(
        cast(FloatArray, unresolved_covariance),
        name="unresolved innovation covariance",
    )
    return (
        cast(FloatArray, mapping),
        cast(FloatArray, _symmetric(projector)),
        unresolved_covariance,
        rank,
        used_pseudoinverse,
    )


@dataclass(frozen=True, slots=True)
class InnovationDisturbanceResult:
    """Posterior moments of the original location-level innovations.

    Under ``alpha_t = c + T alpha_(t-1) + R eta_t``, transition index ``t``
    stores moments for ``eta_(t+1)``. ``unresolved_covariance`` is the component
    of ``Var(eta_t | R eta_t)`` that remains unidentified by the state
    disturbance. It is retained rather than removed with an arbitrary inverse
    of ``R``.
    """

    smoother_result: KalmanSmootherResult
    innovation_mean: FloatArray
    innovation_covariance: FloatArray
    conditioning_map: FloatArray
    unresolved_covariance: FloatArray
    process_support_projector: FloatArray
    process_rank: int
    used_pseudoinverse: bool
    mean_support_residual: FloatArray
    covariance_support_residual: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.smoother_result, KalmanSmootherResult):
            raise TypeError("smoother_result must be a KalmanSmootherResult")
        model = self.smoother_result.model
        n_transitions = int(self.smoother_result.state_disturbance_mean.shape[0])
        n_locations = model.n_locations
        state_dim = model.state_dim

        innovation_mean = _freeze_float(
            self.innovation_mean,
            name="innovation_mean",
            ndim=2,
        )
        if innovation_mean.shape != (n_transitions, n_locations):
            raise ValueError("innovation_mean has an invalid shape")
        innovation_covariance = _freeze_float(
            self.innovation_covariance,
            name="innovation_covariance",
            ndim=3,
        )
        if innovation_covariance.shape != (
            n_transitions,
            n_locations,
            n_locations,
        ):
            raise ValueError("innovation_covariance has an invalid shape")
        conditioning_map = _freeze_float(
            self.conditioning_map,
            name="conditioning_map",
            ndim=2,
        )
        if conditioning_map.shape != (n_locations, state_dim):
            raise ValueError("conditioning_map has an invalid shape")
        unresolved_covariance = _freeze_float(
            self.unresolved_covariance,
            name="unresolved_covariance",
            ndim=2,
        )
        if unresolved_covariance.shape != (n_locations, n_locations):
            raise ValueError("unresolved_covariance has an invalid shape")
        process_support_projector = _freeze_float(
            self.process_support_projector,
            name="process_support_projector",
            ndim=2,
        )
        if process_support_projector.shape != (state_dim, state_dim):
            raise ValueError("process_support_projector has an invalid shape")
        mean_support_residual = _freeze_float(
            self.mean_support_residual,
            name="mean_support_residual",
            ndim=1,
        )
        covariance_support_residual = _freeze_float(
            self.covariance_support_residual,
            name="covariance_support_residual",
            ndim=1,
        )
        expected_residual_shape = (n_transitions,)
        if mean_support_residual.shape != expected_residual_shape:
            raise ValueError("mean_support_residual has an invalid shape")
        if covariance_support_residual.shape != expected_residual_shape:
            raise ValueError("covariance_support_residual has an invalid shape")
        if np.any(mean_support_residual < 0.0):
            raise ValueError("mean_support_residual must be non-negative")
        if np.any(covariance_support_residual < 0.0):
            raise ValueError("covariance_support_residual must be non-negative")

        process_rank = int(self.process_rank)
        if process_rank < 0 or process_rank > state_dim:
            raise ValueError("process_rank is outside the state dimension")
        used_pseudoinverse = bool(self.used_pseudoinverse)
        if used_pseudoinverse != (process_rank < state_dim):
            raise ValueError(
                "used_pseudoinverse must identify rank-deficient process covariance"
            )

        object.__setattr__(self, "innovation_mean", innovation_mean)
        object.__setattr__(self, "innovation_covariance", innovation_covariance)
        object.__setattr__(self, "conditioning_map", conditioning_map)
        object.__setattr__(self, "unresolved_covariance", unresolved_covariance)
        object.__setattr__(
            self,
            "process_support_projector",
            process_support_projector,
        )
        object.__setattr__(self, "process_rank", process_rank)
        object.__setattr__(self, "used_pseudoinverse", used_pseudoinverse)
        object.__setattr__(self, "mean_support_residual", mean_support_residual)
        object.__setattr__(
            self,
            "covariance_support_residual",
            covariance_support_residual,
        )

    @property
    def model(self) -> StateSpaceModel:
        """State-space model defining ``R`` and the innovation covariance."""
        return self.smoother_result.model

    @property
    def n_transitions(self) -> int:
        """Number of innovation disturbances represented in the result."""
        return int(self.innovation_mean.shape[0])

    @property
    def unresolved_variance(self) -> float:
        """Total innovation variance not identified by the state disturbance."""
        return float(np.trace(self.unresolved_covariance))


def innovation_disturbance_smoother(
    smoother_result: KalmanSmootherResult,
    *,
    rcond: float = 1e-10,
) -> InnovationDisturbanceResult:
    """Map state-disturbance posterior moments to original innovations.

    Let ``w_t = R eta_t``, ``eta_t ~ N(0, Q)``, and ``S = R Q R'``. The exact
    conditional Gaussian map is ``A = Q R' S+``. Integrating ``eta_t | w_t``
    over the RTS posterior ``w_t | y_1:T`` gives

    ``E(eta_t | y) = A E(w_t | y)``

    and

    ``Var(eta_t | y) = Q - A S A' + A Var(w_t | y) A'``.

    The positive-eigenspace pseudoinverse is used only when ``S`` is rank
    deficient. Components of ``eta_t`` in the conditional null space are kept
    in ``unresolved_covariance``.
    """
    if not isinstance(smoother_result, KalmanSmootherResult):
        raise TypeError("smoother_result must be a KalmanSmootherResult")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be positive and finite")

    model = smoother_result.model
    (
        conditioning_map,
        process_support_projector,
        unresolved_covariance,
        process_rank,
        used_pseudoinverse,
    ) = _conditioning_map(model, rcond=float(rcond))

    state_mean = smoother_result.state_disturbance_mean
    state_covariance = smoother_result.state_disturbance_covariance
    innovation_mean = state_mean @ conditioning_map.T
    innovation_covariance = np.empty(
        (
            state_covariance.shape[0],
            model.n_locations,
            model.n_locations,
        ),
        dtype=float,
    )
    mean_support_residual = np.linalg.norm(
        state_mean - state_mean @ process_support_projector,
        axis=1,
    )
    covariance_support_residual = np.empty(state_covariance.shape[0], dtype=float)

    for time_index, covariance in enumerate(state_covariance):
        supported_covariance = (
            process_support_projector
            @ covariance
            @ process_support_projector
        )
        covariance_support_residual[time_index] = np.linalg.norm(
            covariance - supported_covariance,
            ord="fro",
        )
        posterior_covariance = (
            unresolved_covariance
            + conditioning_map @ covariance @ conditioning_map.T
        )
        innovation_covariance[time_index] = _project_positive_semidefinite(
            cast(FloatArray, posterior_covariance),
            name="innovation disturbance covariance",
        )

    return InnovationDisturbanceResult(
        smoother_result=smoother_result,
        innovation_mean=innovation_mean,
        innovation_covariance=innovation_covariance,
        conditioning_map=conditioning_map,
        unresolved_covariance=unresolved_covariance,
        process_support_projector=process_support_projector,
        process_rank=process_rank,
        used_pseudoinverse=used_pseudoinverse,
        mean_support_residual=mean_support_residual,
        covariance_support_residual=covariance_support_residual,
    )
