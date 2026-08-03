# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Linear Gaussian state-space tools for stationary STARMA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_discrete_lyapunov

from pystarmax._validation import FloatArray, as_float_matrix
from pystarmax.weights import SpatialWeights, coerce_weights

Initialization = Literal["stationary", "diffuse", "known"]
BoolArray = npt.NDArray[np.bool_]


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


def _positive_semidefinite(value: FloatArray, *, name: str) -> FloatArray:
    symmetric = _symmetric(value)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _parameter_rows(value: Any, *, width: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, width), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (order, {width})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, np.ascontiguousarray(array, dtype=float))


def _intercept_vector(value: Any, *, n_locations: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        result = np.full(n_locations, float(array), dtype=float)
    elif array.ndim == 1 and array.shape[0] == n_locations:
        result = np.asarray(array, dtype=float)
    else:
        raise ValueError("intercept must be scalar or have one value per location")
    if not np.all(np.isfinite(result)):
        raise ValueError("intercept must contain only finite values")
    return cast(FloatArray, np.ascontiguousarray(result, dtype=float))


def _compose_operators(parameters: FloatArray, weights: SpatialWeights) -> FloatArray:
    operators = np.empty(
        (parameters.shape[0], weights.n_locations, weights.n_locations),
        dtype=float,
    )
    for temporal_lag, row in enumerate(parameters):
        operator = np.zeros(
            (weights.n_locations, weights.n_locations),
            dtype=float,
        )
        for coefficient, matrix in zip(row, weights, strict=True):
            operator += coefficient * matrix
        operators[temporal_lag] = operator
    return cast(FloatArray, operators)


@dataclass(frozen=True, slots=True)
class StateSpaceModel:
    """Immutable linear Gaussian state-space representation.

    The transition convention is

    ``alpha_t = state_intercept + transition @ alpha_(t-1) + selection @ eta_t``

    with ``eta_t ~ N(0, innovation_covariance)`` and
    ``y_t = design @ alpha_t``.
    """

    transition: FloatArray
    design: FloatArray
    selection: FloatArray
    state_intercept: FloatArray
    innovation_covariance: FloatArray
    ar_order: int
    ma_order: int

    def __post_init__(self) -> None:
        transition = _freeze_float(
            self.transition,
            name="transition",
            ndim=2,
        )
        if transition.shape[0] != transition.shape[1] or transition.shape[0] == 0:
            raise ValueError("transition must be a non-empty square matrix")
        design = _freeze_float(self.design, name="design", ndim=2)
        selection = _freeze_float(self.selection, name="selection", ndim=2)
        state_intercept = _freeze_float(
            np.asarray(self.state_intercept, dtype=float).reshape(-1),
            name="state_intercept",
            ndim=1,
        )
        innovation_covariance = _freeze_float(
            self.innovation_covariance,
            name="innovation_covariance",
            ndim=2,
        )
        state_dim = transition.shape[0]
        n_locations = design.shape[0]
        if design.shape[1] != state_dim:
            raise ValueError("design columns must match the state dimension")
        if selection.shape != (state_dim, n_locations):
            raise ValueError("selection must have shape (state dimension, locations)")
        if state_intercept.shape != (state_dim,):
            raise ValueError("state_intercept must match the state dimension")
        if innovation_covariance.shape != (n_locations, n_locations):
            raise ValueError(
                "innovation_covariance must have one row and column per location"
            )
        covariance = _positive_semidefinite(
            innovation_covariance,
            name="innovation_covariance",
        ).copy()
        covariance.setflags(write=False)
        ar_order = int(self.ar_order)
        ma_order = int(self.ma_order)
        if ar_order < 0 or ma_order < 0:
            raise ValueError("ar_order and ma_order must be non-negative")
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "state_intercept", state_intercept)
        object.__setattr__(self, "innovation_covariance", covariance)
        object.__setattr__(self, "ar_order", ar_order)
        object.__setattr__(self, "ma_order", ma_order)

    @property
    def state_dim(self) -> int:
        """Number of latent state elements."""
        return int(self.transition.shape[0])

    @property
    def n_locations(self) -> int:
        """Number of observed locations."""
        return int(self.design.shape[0])

    @property
    def process_covariance(self) -> FloatArray:
        """State disturbance covariance ``R Q R'``."""
        covariance = self.selection @ self.innovation_covariance @ self.selection.T
        return cast(FloatArray, _symmetric(covariance))


@dataclass(frozen=True, slots=True)
class KalmanFilterResult:
    """Immutable output from a Gaussian Kalman filter."""

    model: StateSpaceModel
    predicted_state: FloatArray
    filtered_state: FloatArray
    predicted_covariance: FloatArray
    filtered_covariance: FloatArray
    innovations: FloatArray
    innovation_covariance: FloatArray
    observed_mask: BoolArray
    log_likelihood_contributions: FloatArray
    jitter: FloatArray
    log_likelihood: float
    n_observations: int
    initialization: str

    def __post_init__(self) -> None:
        predicted = np.asarray(self.predicted_state, dtype=float)
        if predicted.ndim != 2 or predicted.shape[0] == 0:
            raise ValueError("predicted_state must have shape (time, state)")
        n_time = int(predicted.shape[0])
        state_dim = self.model.state_dim
        n_locations = self.model.n_locations
        expected_state = (n_time, state_dim)
        expected_state_covariance = (n_time, state_dim, state_dim)
        expected_observation = (n_time, n_locations)
        expected_observation_covariance = (
            n_time,
            n_locations,
            n_locations,
        )

        arrays: list[FloatArray] = []
        for name, value, shape, allow_nan in (
            ("predicted_state", self.predicted_state, expected_state, False),
            ("filtered_state", self.filtered_state, expected_state, False),
            (
                "predicted_covariance",
                self.predicted_covariance,
                expected_state_covariance,
                False,
            ),
            (
                "filtered_covariance",
                self.filtered_covariance,
                expected_state_covariance,
                False,
            ),
            ("innovations", self.innovations, expected_observation, True),
            (
                "innovation_covariance",
                self.innovation_covariance,
                expected_observation_covariance,
                True,
            ),
            (
                "log_likelihood_contributions",
                self.log_likelihood_contributions,
                (n_time,),
                False,
            ),
            ("jitter", self.jitter, (n_time,), False),
        ):
            array = np.asarray(value, dtype=float)
            if array.shape != shape:
                raise ValueError(f"{name} has an invalid shape")
            valid = ~np.isinf(array) if allow_nan else np.isfinite(array)
            if not np.all(valid):
                raise ValueError(f"{name} contains invalid values")
            frozen = np.ascontiguousarray(array, dtype=float).copy()
            frozen.setflags(write=False)
            arrays.append(cast(FloatArray, frozen))

        observed_mask = np.asarray(self.observed_mask, dtype=bool)
        if observed_mask.shape != expected_observation:
            raise ValueError("observed_mask has an invalid shape")
        frozen_mask = np.ascontiguousarray(observed_mask, dtype=bool).copy()
        frozen_mask.setflags(write=False)

        n_observations = int(self.n_observations)
        if n_observations != int(np.count_nonzero(frozen_mask)):
            raise ValueError("n_observations must match observed_mask")
        log_likelihood = float(self.log_likelihood)
        if not np.isfinite(log_likelihood):
            raise ValueError("log_likelihood must be finite")

        object.__setattr__(self, "predicted_state", arrays[0])
        object.__setattr__(self, "filtered_state", arrays[1])
        object.__setattr__(self, "predicted_covariance", arrays[2])
        object.__setattr__(self, "filtered_covariance", arrays[3])
        object.__setattr__(self, "innovations", arrays[4])
        object.__setattr__(self, "innovation_covariance", arrays[5])
        object.__setattr__(self, "log_likelihood_contributions", arrays[6])
        object.__setattr__(self, "jitter", arrays[7])
        object.__setattr__(self, "observed_mask", frozen_mask)
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "n_observations", n_observations)
        object.__setattr__(self, "initialization", str(self.initialization))

    @property
    def predicted_observations(self) -> FloatArray:
        """One-step observation predictions before each measurement update."""
        values = self.predicted_state @ self.model.design.T
        return cast(FloatArray, np.ascontiguousarray(values, dtype=float))

    @property
    def filtered_observations(self) -> FloatArray:
        """Observation-scale values after each measurement update."""
        values = self.filtered_state @ self.model.design.T
        return cast(FloatArray, np.ascontiguousarray(values, dtype=float))


def build_starma_state_space(
    ar_parameters: Any,
    ma_parameters: Any,
    weights: Any,
    innovation_covariance: Any,
    *,
    intercept: Any = 0.0,
) -> StateSpaceModel:
    """Build a companion state-space representation from STARMA coefficients.

    Parameters use shape ``(temporal order, number of spatial weights)``.
    The spatial operator at temporal lag ``i`` is the weighted sum of the
    supplied spatial matrices for that parameter row.
    """
    covariance = as_float_matrix(
        innovation_covariance,
        name="innovation_covariance",
    )
    if covariance.shape[0] != covariance.shape[1] or covariance.shape[0] == 0:
        raise ValueError("innovation_covariance must be a non-empty square matrix")
    n_locations = int(covariance.shape[0])
    resolved_weights = coerce_weights(weights, n_locations=n_locations)
    ar_parameters = _parameter_rows(
        ar_parameters,
        width=len(resolved_weights),
        name="ar_parameters",
    )
    ma_parameters = _parameter_rows(
        ma_parameters,
        width=len(resolved_weights),
        name="ma_parameters",
    )
    ar_operators = _compose_operators(ar_parameters, resolved_weights)
    ma_operators = _compose_operators(ma_parameters, resolved_weights)
    ar_order = int(ar_parameters.shape[0])
    ma_order = int(ma_parameters.shape[0])
    z_blocks = max(1, ar_order)
    state_blocks = z_blocks + ma_order
    state_dim = state_blocks * n_locations
    identity = np.eye(n_locations, dtype=float)

    transition = np.zeros((state_dim, state_dim), dtype=float)
    for temporal_lag in range(ar_order):
        start = temporal_lag * n_locations
        transition[:n_locations, start : start + n_locations] = ar_operators[
            temporal_lag
        ]
    innovation_offset = z_blocks * n_locations
    for temporal_lag in range(ma_order):
        start = innovation_offset + temporal_lag * n_locations
        transition[:n_locations, start : start + n_locations] = ma_operators[
            temporal_lag
        ]
    for block in range(1, z_blocks):
        row = block * n_locations
        column = (block - 1) * n_locations
        transition[row : row + n_locations, column : column + n_locations] = identity
    for block in range(1, ma_order):
        row = innovation_offset + block * n_locations
        column = innovation_offset + (block - 1) * n_locations
        transition[row : row + n_locations, column : column + n_locations] = identity

    design = np.zeros((n_locations, state_dim), dtype=float)
    design[:, :n_locations] = identity
    selection = np.zeros((state_dim, n_locations), dtype=float)
    selection[:n_locations] = identity
    if ma_order:
        selection[innovation_offset : innovation_offset + n_locations] = identity
    state_intercept = np.zeros(state_dim, dtype=float)
    state_intercept[:n_locations] = _intercept_vector(
        intercept,
        n_locations=n_locations,
    )

    return StateSpaceModel(
        transition=transition,
        design=design,
        selection=selection,
        state_intercept=state_intercept,
        innovation_covariance=covariance,
        ar_order=ar_order,
        ma_order=ma_order,
    )


def fitted_starma_state_space(model: Any) -> StateSpaceModel:
    """Build a state-space model from a fitted STAR or STARMA estimator."""
    if getattr(model, "result_", None) is None:
        raise RuntimeError("fit must be called before building a state-space model")
    (
        weights,
        _data,
        _residuals,
        intercept,
        ar_parameters,
        ma_parameters,
    ) = model._fitted_components()
    return build_starma_state_space(
        ar_parameters,
        ma_parameters,
        weights,
        model.result_.innovation_covariance,
        intercept=intercept,
    )


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


def _initial_distribution(
    model: StateSpaceModel,
    *,
    initialization: Initialization,
    initial_state: Any | None,
    initial_covariance: Any | None,
    diffuse_scale: float,
) -> tuple[FloatArray, FloatArray]:
    if initialization not in {"stationary", "diffuse", "known"}:
        raise ValueError("initialization must be 'stationary', 'diffuse', or 'known'")
    if initialization == "known":
        if initial_state is None or initial_covariance is None:
            raise ValueError(
                "known initialization requires initial_state and initial_covariance"
            )
        state = np.asarray(initial_state, dtype=float)
        covariance = np.asarray(initial_covariance, dtype=float)
        if state.shape != (model.state_dim,):
            raise ValueError("initial_state must match the state dimension")
        if covariance.shape != (model.state_dim, model.state_dim):
            raise ValueError("initial_covariance must match the state dimension")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
            raise ValueError("known initial values must be finite")
        covariance = _positive_semidefinite(
            cast(FloatArray, covariance),
            name="initial_covariance",
        )
        return (
            cast(FloatArray, np.asarray(state, dtype=float)),
            cast(FloatArray, covariance),
        )

    if initial_state is not None or initial_covariance is not None:
        raise ValueError(
            "initial_state and initial_covariance are only used with "
            "initialization='known'"
        )
    if initialization == "diffuse":
        scale = float(diffuse_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("diffuse_scale must be positive and finite")
        return (
            np.zeros(model.state_dim, dtype=float),
            np.eye(model.state_dim, dtype=float) * scale,
        )

    eigenvalues = np.linalg.eigvals(model.transition)
    if float(np.max(np.abs(eigenvalues), initial=0.0)) >= 1.0 - 1e-10:
        raise ValueError(
            "stationary initialization requires transition spectral radius below one"
        )
    identity = np.eye(model.state_dim, dtype=float)
    state = np.linalg.solve(
        identity - model.transition,
        model.state_intercept,
    )
    covariance = solve_discrete_lyapunov(
        model.transition,
        model.process_covariance,
    )
    covariance = _positive_semidefinite(
        cast(FloatArray, np.asarray(covariance, dtype=float)),
        name="stationary covariance",
    )
    return (
        cast(FloatArray, np.asarray(state, dtype=float)),
        covariance,
    )


def _innovation_cholesky(
    covariance: FloatArray,
) -> tuple[FloatArray, float]:
    symmetric = _symmetric(covariance)
    scale = max(
        1.0,
        float(np.max(np.abs(np.diag(symmetric)), initial=0.0)),
    )
    jitter = 0.0
    for attempt in range(8):
        try:
            adjusted = symmetric + np.eye(symmetric.shape[0]) * jitter
            factor = np.linalg.cholesky(adjusted)
            return cast(FloatArray, factor), jitter
        except np.linalg.LinAlgError:
            if attempt == 0:
                jitter = np.finfo(float).eps * scale * 100.0
            else:
                jitter *= 10.0
    raise np.linalg.LinAlgError(
        "innovation covariance is not numerically positive definite"
    )


def kalman_filter(
    data: Any,
    model: StateSpaceModel,
    *,
    initialization: Initialization = "stationary",
    initial_state: Any | None = None,
    initial_covariance: Any | None = None,
    diffuse_scale: float = 1e6,
) -> KalmanFilterResult:
    """Filter complete or partially missing observations.

    Missing cells are represented by ``NaN``. At each time step the measurement
    equation is reduced to the observed locations. A fully missing row performs
    prediction only and contributes zero to the Gaussian log likelihood.
    """
    observations = _validate_observations(
        data,
        n_locations=model.n_locations,
    )
    state, covariance = _initial_distribution(
        model,
        initialization=initialization,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        diffuse_scale=diffuse_scale,
    )
    n_time = int(observations.shape[0])
    state_dim = model.state_dim
    n_locations = model.n_locations

    predicted_state = np.empty((n_time, state_dim), dtype=float)
    filtered_state = np.empty((n_time, state_dim), dtype=float)
    predicted_covariance = np.empty(
        (n_time, state_dim, state_dim),
        dtype=float,
    )
    filtered_covariance = np.empty_like(predicted_covariance)
    innovations = np.full((n_time, n_locations), np.nan, dtype=float)
    innovation_covariances = np.full(
        (n_time, n_locations, n_locations),
        np.nan,
        dtype=float,
    )
    observed_mask = np.isfinite(observations)
    contributions = np.zeros(n_time, dtype=float)
    jitters = np.zeros(n_time, dtype=float)
    process_covariance = model.process_covariance
    log_two_pi = float(np.log(2.0 * np.pi))

    for time_index in range(n_time):
        state_prediction = model.state_intercept + model.transition @ state
        covariance_prediction = (
            model.transition @ covariance @ model.transition.T + process_covariance
        )
        covariance_prediction = _symmetric(covariance_prediction)
        predicted_state[time_index] = state_prediction
        predicted_covariance[time_index] = covariance_prediction

        mask = observed_mask[time_index]
        if not np.any(mask):
            state = state_prediction
            covariance = covariance_prediction
            filtered_state[time_index] = state
            filtered_covariance[time_index] = covariance
            continue

        design = model.design[mask]
        observed = observations[time_index, mask]
        innovation = observed - design @ state_prediction
        innovation_covariance = design @ covariance_prediction @ design.T
        factor, jitter = _innovation_cholesky(cast(FloatArray, innovation_covariance))
        adjusted_covariance = (
            innovation_covariance + np.eye(innovation_covariance.shape[0]) * jitter
        )
        covariance_state_observation = covariance_prediction @ design.T
        gain = np.linalg.solve(
            factor.T,
            np.linalg.solve(factor, covariance_state_observation.T),
        ).T
        state = state_prediction + gain @ innovation
        covariance = covariance_prediction - gain @ adjusted_covariance @ gain.T
        covariance = _symmetric(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(
            1.0,
            float(np.max(np.abs(eigenvalues), initial=0.0)),
        )
        if float(np.min(eigenvalues, initial=0.0)) < -1e-9 * scale:
            raise np.linalg.LinAlgError(
                "filtered covariance became numerically indefinite"
            )
        covariance = (eigenvectors * np.clip(eigenvalues, 0.0, np.inf)) @ (
            eigenvectors.T
        )
        solved = np.linalg.solve(factor, innovation)
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
        contributions[time_index] = -0.5 * (
            int(mask.sum()) * log_two_pi + log_determinant + float(solved @ solved)
        )

        observed_indices = np.flatnonzero(mask)
        innovations[time_index, mask] = innovation
        time_covariance = innovation_covariances[time_index]
        time_covariance[np.ix_(observed_indices, observed_indices)] = (
            adjusted_covariance
        )
        jitters[time_index] = jitter
        filtered_state[time_index] = state
        filtered_covariance[time_index] = covariance

    return KalmanFilterResult(
        model=model,
        predicted_state=predicted_state,
        filtered_state=filtered_state,
        predicted_covariance=predicted_covariance,
        filtered_covariance=filtered_covariance,
        innovations=innovations,
        innovation_covariance=innovation_covariances,
        observed_mask=observed_mask,
        log_likelihood_contributions=contributions,
        jitter=jitters,
        log_likelihood=float(np.sum(contributions)),
        n_observations=int(np.count_nonzero(observed_mask)),
        initialization=initialization,
    )


def kalman_loglikelihood(
    data: Any,
    model: StateSpaceModel,
    *,
    initialization: Initialization = "stationary",
    initial_state: Any | None = None,
    initial_covariance: Any | None = None,
    diffuse_scale: float = 1e6,
) -> float:
    """Return the Gaussian Kalman log likelihood."""
    return kalman_filter(
        data,
        model,
        initialization=initialization,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        diffuse_scale=diffuse_scale,
    ).log_likelihood
