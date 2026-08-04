# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse state-space construction for ordinary integrated models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.exact_diffuse import ExactDiffuseFilterResult, exact_diffuse_filter
from pystarmax.state_space import StateSpaceModel


def _freeze(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _positive_semidefinite(value: Any, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    result = (eigenvectors * np.clip(eigenvalues, 0.0, np.inf)) @ eigenvectors.T
    return cast(FloatArray, np.ascontiguousarray(0.5 * (result + result.T)))


def _stationary_distribution(model: StateSpaceModel) -> tuple[FloatArray, FloatArray]:
    eigenvalues = np.linalg.eigvals(model.transition)
    radius = float(np.max(np.abs(eigenvalues), initial=0.0))
    if radius >= 1.0 - 1e-10:
        raise ValueError(
            "the transformed state-space transition must be stationary before "
            "ordinary integration is added"
        )
    identity = np.eye(model.state_dim, dtype=float)
    state = np.linalg.solve(identity - model.transition, model.state_intercept)
    covariance = solve_discrete_lyapunov(
        model.transition,
        model.process_covariance,
    )
    covariance = _positive_semidefinite(
        covariance,
        name="stationary transformed-state covariance",
    )
    return (
        cast(FloatArray, np.ascontiguousarray(state, dtype=float)),
        covariance,
    )


@dataclass(frozen=True, slots=True)
class ExactIntegratedStateSpace:
    """Level-state model and exact diffuse initial covariance decomposition."""

    model: StateSpaceModel
    transformed_model: StateSpaceModel
    integration_order: int
    initial_state: FloatArray
    initial_covariance: FloatArray
    initial_diffuse_covariance: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.model, StateSpaceModel):
            raise TypeError("model must be a StateSpaceModel")
        if not isinstance(self.transformed_model, StateSpaceModel):
            raise TypeError("transformed_model must be a StateSpaceModel")
        order = validate_nonnegative_int(
            self.integration_order,
            name="integration_order",
        )
        state = _freeze(self.initial_state, name="initial_state", ndim=1)
        finite = _positive_semidefinite(
            self.initial_covariance,
            name="initial_covariance",
        )
        diffuse = _positive_semidefinite(
            self.initial_diffuse_covariance,
            name="initial_diffuse_covariance",
        )
        if state.shape != (self.model.state_dim,):
            raise ValueError("initial_state must match the augmented state dimension")
        expected = (self.model.state_dim, self.model.state_dim)
        if finite.shape != expected or diffuse.shape != expected:
            raise ValueError("initial covariance matrices must match the augmented state")
        if self.model.n_locations != self.transformed_model.n_locations:
            raise ValueError("augmented and transformed models must share locations")
        object.__setattr__(self, "integration_order", order)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "initial_covariance", finite)
        object.__setattr__(self, "initial_diffuse_covariance", diffuse)

    @property
    def n_diffuse_directions(self) -> int:
        """Nominal number of ordinary-integration diffuse directions."""
        return self.integration_order * self.model.n_locations

    def filter(
        self,
        data: Any,
        *,
        tolerance: float = 1e-10,
    ) -> ExactDiffuseFilterResult:
        """Filter original level observations with exact diffuse initialization."""
        return exact_diffuse_filter(
            data,
            self.model,
            initial_state=self.initial_state,
            initial_covariance=self.initial_covariance,
            initial_diffuse_covariance=self.initial_diffuse_covariance,
            tolerance=tolerance,
        )

    def loglikelihood(
        self,
        data: Any,
        *,
        tolerance: float = 1e-10,
    ) -> float:
        """Return the original-level exact diffuse Gaussian log likelihood."""
        return self.filter(data, tolerance=tolerance).log_likelihood


def build_exact_integrated_state_space(
    transformed_model: StateSpaceModel,
    integration_order: int,
) -> ExactIntegratedStateSpace:
    """Augment a stationary transformed model with ordinary integration states.

    The transformed observation is ``x_t = Z beta_t``. For order ``d``, the
    augmented blocks are ``y_t, Delta y_t, ..., Delta^(d-1) y_t, beta_t``.
    Every integration block follows

    ``Delta^r y_t = sum_{k=r}^{d-1} Delta^k y_(t-1) + x_t``.

    The integration blocks receive exact diffuse covariance while the
    transformed state receives its stationary finite covariance.
    """
    if not isinstance(transformed_model, StateSpaceModel):
        raise TypeError("transformed_model must be a StateSpaceModel")
    order = validate_nonnegative_int(integration_order, name="integration_order")
    stationary_state, stationary_covariance = _stationary_distribution(
        transformed_model
    )
    if order == 0:
        zero_diffuse = np.zeros(
            (transformed_model.state_dim, transformed_model.state_dim),
            dtype=float,
        )
        return ExactIntegratedStateSpace(
            model=transformed_model,
            transformed_model=transformed_model,
            integration_order=0,
            initial_state=stationary_state,
            initial_covariance=stationary_covariance,
            initial_diffuse_covariance=zero_diffuse,
        )

    n_locations = transformed_model.n_locations
    transformed_dim = transformed_model.state_dim
    integrated_dim = order * n_locations
    state_dim = integrated_dim + transformed_dim
    innovation_dim = transformed_model.innovation_dim
    identity = np.eye(n_locations, dtype=float)

    transition = np.zeros((state_dim, state_dim), dtype=float)
    for row_block in range(order):
        row = row_block * n_locations
        for column_block in range(row_block, order):
            column = column_block * n_locations
            transition[row : row + n_locations, column : column + n_locations] = (
                identity
            )
    transformed_effect = transformed_model.design @ transformed_model.transition
    transformed_intercept = (
        transformed_model.design @ transformed_model.state_intercept
    )
    transformed_selection = (
        transformed_model.design @ transformed_model.selection
    )
    for row_block in range(order):
        row = row_block * n_locations
        transition[
            row : row + n_locations,
            integrated_dim:,
        ] = transformed_effect

    transition[integrated_dim:, integrated_dim:] = transformed_model.transition
    design = np.zeros((n_locations, state_dim), dtype=float)
    design[:, :n_locations] = identity
    selection = np.zeros((state_dim, innovation_dim), dtype=float)
    state_intercept = np.zeros(state_dim, dtype=float)
    for row_block in range(order):
        row = row_block * n_locations
        selection[row : row + n_locations] = transformed_selection
        state_intercept[row : row + n_locations] = transformed_intercept
    selection[integrated_dim:] = transformed_model.selection
    state_intercept[integrated_dim:] = transformed_model.state_intercept

    model = StateSpaceModel(
        transition=transition,
        design=design,
        selection=selection,
        state_intercept=state_intercept,
        innovation_covariance=transformed_model.innovation_covariance,
        ar_order=transformed_model.ar_order + order,
        ma_order=transformed_model.ma_order,
    )
    initial_state = np.zeros(state_dim, dtype=float)
    initial_state[integrated_dim:] = stationary_state
    initial_covariance = np.zeros((state_dim, state_dim), dtype=float)
    initial_covariance[integrated_dim:, integrated_dim:] = stationary_covariance
    initial_diffuse_covariance = np.zeros((state_dim, state_dim), dtype=float)
    initial_diffuse_covariance[:integrated_dim, :integrated_dim] = np.eye(
        integrated_dim,
        dtype=float,
    )
    return ExactIntegratedStateSpace(
        model=model,
        transformed_model=transformed_model,
        integration_order=order,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        initial_diffuse_covariance=initial_diffuse_covariance,
    )


def exact_integrated_filter(
    data: Any,
    transformed_model: StateSpaceModel,
    integration_order: int,
    *,
    tolerance: float = 1e-10,
) -> ExactDiffuseFilterResult:
    """Build and filter an ordinary integrated level-state model."""
    specification = build_exact_integrated_state_space(
        transformed_model,
        integration_order,
    )
    return specification.filter(data, tolerance=tolerance)


def exact_integrated_loglikelihood(
    data: Any,
    transformed_model: StateSpaceModel,
    integration_order: int,
    *,
    tolerance: float = 1e-10,
) -> float:
    """Return an exact diffuse likelihood for original level observations."""
    return exact_integrated_filter(
        data,
        transformed_model,
        integration_order,
        tolerance=tolerance,
    ).log_likelihood


__all__ = [
    "ExactIntegratedStateSpace",
    "build_exact_integrated_state_space",
    "exact_integrated_filter",
    "exact_integrated_loglikelihood",
]
