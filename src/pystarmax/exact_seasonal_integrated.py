# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Exact diffuse state-space construction for seasonal integrated models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.differencing import differencing_coefficients
from pystarmax.exact_diffuse import ExactDiffuseFilterResult, exact_diffuse_filter
from pystarmax.exact_integrated import (
    _positive_semidefinite,
    _stationary_distribution,
    build_exact_integrated_state_space,
)
from pystarmax.state_space import StateSpaceModel


def _positive_int(value: int, *, name: str) -> int:
    result = validate_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _freeze_vector(value: Any, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _freeze_covariance(value: Any, *, name: str) -> FloatArray:
    array = _positive_semidefinite(value, name=name)
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


@dataclass(frozen=True, slots=True)
class ExactSeasonalIntegratedStateSpace:
    """Original-level state space for ordinary and seasonal integration.

    The stationary transformed observation obeys

    ``x_t = (1 - B)^d (1 - B^s)^D y_t``.

    The augmented state retains the original-level lag coordinates needed to
    invert the complete differencing polynomial together with the stationary
    transformed state. Those lag coordinates receive exact diffuse covariance;
    the transformed state receives its stationary finite distribution.
    """

    model: StateSpaceModel
    transformed_model: StateSpaceModel
    ordinary_integration_order: int
    seasonal_integration_order: int
    seasonal_period: int
    polynomial_coefficients: FloatArray
    initial_state: FloatArray
    initial_covariance: FloatArray
    initial_diffuse_covariance: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.model, StateSpaceModel):
            raise TypeError("model must be a StateSpaceModel")
        if not isinstance(self.transformed_model, StateSpaceModel):
            raise TypeError("transformed_model must be a StateSpaceModel")
        ordinary_order = validate_nonnegative_int(
            self.ordinary_integration_order,
            name="ordinary_integration_order",
        )
        seasonal_order = validate_nonnegative_int(
            self.seasonal_integration_order,
            name="seasonal_integration_order",
        )
        period = _positive_int(self.seasonal_period, name="seasonal_period")
        coefficients = _freeze_vector(
            self.polynomial_coefficients,
            name="polynomial_coefficients",
        )
        degree = ordinary_order + seasonal_order * period
        if coefficients.shape != (degree + 1,):
            raise ValueError(
                "polynomial_coefficients must match the combined integration degree"
            )
        if not np.isclose(coefficients[0], 1.0, rtol=0.0, atol=1e-14):
            raise ValueError("polynomial_coefficients must start with one")

        state = _freeze_vector(self.initial_state, name="initial_state")
        finite = _freeze_covariance(
            self.initial_covariance,
            name="initial_covariance",
        )
        diffuse = _freeze_covariance(
            self.initial_diffuse_covariance,
            name="initial_diffuse_covariance",
        )
        expected_state_dim = (
            degree * self.model.n_locations + self.transformed_model.state_dim
        )
        if self.model.state_dim != expected_state_dim:
            raise ValueError(
                "model state dimension must equal the integration lag dimension "
                "plus the transformed state dimension"
            )
        if state.shape != (self.model.state_dim,):
            raise ValueError("initial_state must match the augmented state dimension")
        covariance_shape = (self.model.state_dim, self.model.state_dim)
        if finite.shape != covariance_shape or diffuse.shape != covariance_shape:
            raise ValueError(
                "initial covariance matrices must match the augmented state"
            )
        if self.model.n_locations != self.transformed_model.n_locations:
            raise ValueError("augmented and transformed models must share locations")

        object.__setattr__(
            self,
            "ordinary_integration_order",
            ordinary_order,
        )
        object.__setattr__(
            self,
            "seasonal_integration_order",
            seasonal_order,
        )
        object.__setattr__(self, "seasonal_period", period)
        object.__setattr__(self, "polynomial_coefficients", coefficients)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "initial_covariance", finite)
        object.__setattr__(self, "initial_diffuse_covariance", diffuse)

    @property
    def integration_degree(self) -> int:
        """Degree of ``(1-B)^d (1-B^s)^D``."""
        return (
            self.ordinary_integration_order
            + self.seasonal_integration_order * self.seasonal_period
        )

    @property
    def n_diffuse_directions(self) -> int:
        """Nominal number of original-level diffuse directions."""
        return self.integration_degree * self.model.n_locations

    def filter(
        self,
        data: Any,
        *,
        tolerance: float = 1e-10,
    ) -> ExactDiffuseFilterResult:
        """Filter original-level observations with exact diffuse initialization."""
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


def build_exact_seasonal_integrated_state_space(
    transformed_model: StateSpaceModel,
    ordinary_integration_order: int,
    seasonal_integration_order: int,
    seasonal_period: int,
) -> ExactSeasonalIntegratedStateSpace:
    """Augment a stationary model by ``(1-B)^d (1-B^s)^D``.

    Write the combined differencing polynomial as

    ``delta(B) = 1 + delta_1 B + ... + delta_m B^m``.

    For positive seasonal order, the augmented state is

    ``[y_t, y_(t-1), ..., y_(t-m+1), beta_t]``

    and its first block follows

    ``y_t = -sum(delta_j y_(t-j)) + Z beta_t``.

    The transformed state ``beta_t`` remains stationary and finite. The ``m``
    original-level lag blocks are diffuse. When seasonal order is zero, the
    existing ordinary exact-diffuse construction is reused exactly, including
    its difference-level state coordinates and numerical likelihood contract.
    """
    if not isinstance(transformed_model, StateSpaceModel):
        raise TypeError("transformed_model must be a StateSpaceModel")
    ordinary_order = validate_nonnegative_int(
        ordinary_integration_order,
        name="ordinary_integration_order",
    )
    seasonal_order = validate_nonnegative_int(
        seasonal_integration_order,
        name="seasonal_integration_order",
    )
    period = _positive_int(seasonal_period, name="seasonal_period")
    coefficients = differencing_coefficients(
        ordinary_order=ordinary_order,
        seasonal_order=seasonal_order,
        seasonal_period=period,
    )

    if seasonal_order == 0:
        ordinary = build_exact_integrated_state_space(
            transformed_model,
            ordinary_order,
        )
        return ExactSeasonalIntegratedStateSpace(
            model=ordinary.model,
            transformed_model=ordinary.transformed_model,
            ordinary_integration_order=ordinary_order,
            seasonal_integration_order=0,
            seasonal_period=period,
            polynomial_coefficients=coefficients,
            initial_state=ordinary.initial_state,
            initial_covariance=ordinary.initial_covariance,
            initial_diffuse_covariance=ordinary.initial_diffuse_covariance,
        )

    stationary_state, stationary_covariance = _stationary_distribution(
        transformed_model
    )
    n_locations = transformed_model.n_locations
    transformed_dim = transformed_model.state_dim
    integration_degree = int(coefficients.size - 1)
    integrated_dim = integration_degree * n_locations
    state_dim = integrated_dim + transformed_dim
    innovation_dim = int(transformed_model.selection.shape[1])
    identity = np.eye(n_locations, dtype=float)

    transition = np.zeros((state_dim, state_dim), dtype=float)
    for lag, coefficient in enumerate(coefficients[1:], start=1):
        column = (lag - 1) * n_locations
        transition[
            :n_locations,
            column : column + n_locations,
        ] = (
            -float(coefficient) * identity
        )
    transformed_effect = transformed_model.design @ transformed_model.transition
    transition[:n_locations, integrated_dim:] = transformed_effect
    for block in range(1, integration_degree):
        row = block * n_locations
        column = (block - 1) * n_locations
        transition[
            row : row + n_locations,
            column : column + n_locations,
        ] = identity
    transition[integrated_dim:, integrated_dim:] = transformed_model.transition

    design = np.zeros((n_locations, state_dim), dtype=float)
    design[:, :n_locations] = identity
    selection = np.zeros((state_dim, innovation_dim), dtype=float)
    selection[:n_locations] = transformed_model.design @ transformed_model.selection
    selection[integrated_dim:] = transformed_model.selection
    state_intercept = np.zeros(state_dim, dtype=float)
    state_intercept[:n_locations] = (
        transformed_model.design @ transformed_model.state_intercept
    )
    state_intercept[integrated_dim:] = transformed_model.state_intercept

    model = StateSpaceModel(
        transition=transition,
        design=design,
        selection=selection,
        state_intercept=state_intercept,
        innovation_covariance=transformed_model.innovation_covariance,
        ar_order=transformed_model.ar_order + integration_degree,
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
    return ExactSeasonalIntegratedStateSpace(
        model=model,
        transformed_model=transformed_model,
        ordinary_integration_order=ordinary_order,
        seasonal_integration_order=seasonal_order,
        seasonal_period=period,
        polynomial_coefficients=coefficients,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        initial_diffuse_covariance=initial_diffuse_covariance,
    )


def exact_seasonal_integrated_filter(
    data: Any,
    transformed_model: StateSpaceModel,
    ordinary_integration_order: int,
    seasonal_integration_order: int,
    seasonal_period: int,
    *,
    tolerance: float = 1e-10,
) -> ExactDiffuseFilterResult:
    """Build and filter a seasonal integrated original-level state space."""
    specification = build_exact_seasonal_integrated_state_space(
        transformed_model,
        ordinary_integration_order,
        seasonal_integration_order,
        seasonal_period,
    )
    return specification.filter(data, tolerance=tolerance)


def exact_seasonal_integrated_loglikelihood(
    data: Any,
    transformed_model: StateSpaceModel,
    ordinary_integration_order: int,
    seasonal_integration_order: int,
    seasonal_period: int,
    *,
    tolerance: float = 1e-10,
) -> float:
    """Return the seasonal integrated original-level exact diffuse likelihood."""
    return exact_seasonal_integrated_filter(
        data,
        transformed_model,
        ordinary_integration_order,
        seasonal_integration_order,
        seasonal_period,
        tolerance=tolerance,
    ).log_likelihood


__all__ = [
    "ExactSeasonalIntegratedStateSpace",
    "build_exact_seasonal_integrated_state_space",
    "exact_seasonal_integrated_filter",
    "exact_seasonal_integrated_loglikelihood",
]
