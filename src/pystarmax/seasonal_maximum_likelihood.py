# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Conditional multiplicative seasonal STARIMA Gaussian likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pystarmax._maximum_likelihood_utils import (
    CovarianceType,
    _CovarianceCodec,
    _regularize_covariance,
    _validate_incomplete_data,
)
from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.differencing import (
    CombinedDifferencingState,
    DifferencingState,
    SeasonalDifferencingState,
    differencing_coefficients,
)
from pystarmax.innovation_smoothing import (
    InnovationDisturbanceResult,
    innovation_disturbance_smoother,
)
from pystarmax.seasonal import LagOperator, expand_multiplicative_operators
from pystarmax.smoothing import KalmanSmootherResult, kalman_smoother
from pystarmax.state_space import (
    Initialization,
    KalmanFilterResult,
    StateSpaceModel,
    kalman_filter,
)
from pystarmax.weights import SpatialWeights, coerce_weights


def _positive_int(value: int, *, name: str) -> int:
    result = validate_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _incomplete_matrix(value: Any, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one time observation")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one location")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinite values")
    return cast(FloatArray, np.ascontiguousarray(array, dtype=float))


def _combined_difference_incomplete(
    data: Any,
    *,
    ordinary_order: int,
    seasonal_order: int,
    seasonal_period: int,
) -> tuple[FloatArray, FloatArray, CombinedDifferencingState | None]:
    observations = _incomplete_matrix(data, name="data")
    ordinary_order = validate_nonnegative_int(
        ordinary_order,
        name="integration_order",
    )
    seasonal_order = validate_nonnegative_int(
        seasonal_order,
        name="seasonal_integration_order",
    )
    seasonal_period = _positive_int(seasonal_period, name="seasonal_period")
    offset = ordinary_order + seasonal_order * seasonal_period
    if offset >= observations.shape[0]:
        raise ValueError(
            "ordinary order plus seasonal order times period must be smaller "
            "than the number of time rows"
        )

    ordinary_levels: list[FloatArray] = [observations.copy()]
    for _ in range(ordinary_order):
        ordinary_levels.append(cast(FloatArray, np.diff(ordinary_levels[-1], axis=0)))
    ordinary_transformed = ordinary_levels[-1]

    seasonal_levels: list[FloatArray] = [ordinary_transformed.copy()]
    for _ in range(seasonal_order):
        previous = seasonal_levels[-1]
        seasonal_levels.append(
            cast(
                FloatArray,
                previous[seasonal_period:] - previous[:-seasonal_period],
            )
        )
    transformed = cast(
        FloatArray,
        np.ascontiguousarray(seasonal_levels[-1], dtype=float),
    )

    ordinary_anchors = tuple(level[-1] for level in ordinary_levels[:-1])
    seasonal_histories = tuple(
        level[-seasonal_period:] for level in seasonal_levels[:-1]
    )
    anchors_finite = all(np.all(np.isfinite(anchor)) for anchor in ordinary_anchors)
    histories_finite = all(
        np.all(np.isfinite(history)) for history in seasonal_histories
    )
    state: CombinedDifferencingState | None = None
    if anchors_finite and histories_finite:
        state = CombinedDifferencingState(
            ordinary=DifferencingState(
                order=ordinary_order,
                n_locations=observations.shape[1],
                anchors=ordinary_anchors,
            ),
            seasonal=SeasonalDifferencingState(
                order=seasonal_order,
                period=seasonal_period,
                n_locations=observations.shape[1],
                histories=seasonal_histories,
            ),
        )
    return observations, transformed, state


def _restore_incomplete_fitted(
    observations: FloatArray,
    transformed_fitted: FloatArray,
    *,
    ordinary_order: int,
    seasonal_order: int,
    seasonal_period: int,
) -> FloatArray:
    coefficients = differencing_coefficients(
        ordinary_order=ordinary_order,
        seasonal_order=seasonal_order,
        seasonal_period=seasonal_period,
    )
    offset = int(coefficients.size - 1)
    expected_shape = (
        observations.shape[0] - offset,
        observations.shape[1],
    )
    if transformed_fitted.shape != expected_shape:
        raise ValueError("transformed fitted values have an incompatible shape")

    restored = np.full(observations.shape, np.nan, dtype=float)
    for transformed_index, row in enumerate(transformed_fitted):
        time_index = transformed_index + offset
        for location_index, value in enumerate(row):
            if not np.isfinite(value):
                continue
            original_value = float(value)
            valid = True
            for lag, coefficient in enumerate(coefficients[1:], start=1):
                history = observations[time_index - lag, location_index]
                if not np.isfinite(history):
                    valid = False
                    break
                original_value -= float(coefficient) * float(history)
            if valid:
                restored[time_index, location_index] = original_value
    return cast(FloatArray, np.ascontiguousarray(restored, dtype=float))


def _aggregate_operators(
    operators: tuple[LagOperator, ...],
    *,
    n_locations: int,
) -> tuple[tuple[int, ...], FloatArray]:
    matrices: dict[int, FloatArray] = {}
    for operator in operators:
        if operator.matrix.shape != (n_locations, n_locations):
            raise ValueError("operator matrices must match the innovation dimension")
        if operator.lag not in matrices:
            matrices[operator.lag] = np.zeros(
                (n_locations, n_locations),
                dtype=float,
            )
        matrices[operator.lag] = cast(
            FloatArray,
            matrices[operator.lag] + operator.matrix,
        )
    lags = tuple(sorted(matrices))
    if not lags:
        return (), np.empty((0, n_locations, n_locations), dtype=float)
    stacked = np.stack([matrices[lag] for lag in lags], axis=0)
    return lags, cast(FloatArray, np.ascontiguousarray(stacked, dtype=float))


def _dense_lag_matrices(
    lags: tuple[int, ...],
    matrices: FloatArray,
    *,
    n_locations: int,
) -> FloatArray:
    maximum_lag = max(lags, default=0)
    dense = np.zeros((maximum_lag, n_locations, n_locations), dtype=float)
    for lag, matrix in zip(lags, matrices, strict=True):
        dense[lag - 1] += matrix
    return cast(FloatArray, dense)


def _companion_radius(
    lags: tuple[int, ...],
    matrices: FloatArray,
    *,
    inverse_sign: bool,
    n_locations: int,
) -> float:
    dense = _dense_lag_matrices(
        lags,
        matrices,
        n_locations=n_locations,
    )
    order = int(dense.shape[0])
    if order == 0:
        return 0.0
    companion = np.zeros((order * n_locations, order * n_locations), dtype=float)
    sign = -1.0 if inverse_sign else 1.0
    for index, matrix in enumerate(dense):
        start = index * n_locations
        companion[:n_locations, start : start + n_locations] = sign * matrix
    identity = np.eye(n_locations, dtype=float)
    for block in range(1, order):
        row = block * n_locations
        column = (block - 1) * n_locations
        companion[row : row + n_locations, column : column + n_locations] = identity
    eigenvalues = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigenvalues), initial=0.0))


def _operator_state_space(
    ar_lags: tuple[int, ...],
    ar_matrices: FloatArray,
    ma_lags: tuple[int, ...],
    ma_matrices: FloatArray,
    innovation_covariance: FloatArray,
    *,
    intercept: float,
) -> StateSpaceModel:
    n_locations = int(innovation_covariance.shape[0])
    ar_dense = _dense_lag_matrices(
        ar_lags,
        ar_matrices,
        n_locations=n_locations,
    )
    ma_dense = _dense_lag_matrices(
        ma_lags,
        ma_matrices,
        n_locations=n_locations,
    )
    ar_order = int(ar_dense.shape[0])
    ma_order = int(ma_dense.shape[0])
    z_blocks = max(1, ar_order)
    state_blocks = z_blocks + ma_order
    state_dim = state_blocks * n_locations
    identity = np.eye(n_locations, dtype=float)

    transition = np.zeros((state_dim, state_dim), dtype=float)
    for temporal_lag, matrix in enumerate(ar_dense):
        start = temporal_lag * n_locations
        transition[:n_locations, start : start + n_locations] = matrix
    innovation_offset = z_blocks * n_locations
    for temporal_lag, matrix in enumerate(ma_dense):
        start = innovation_offset + temporal_lag * n_locations
        transition[:n_locations, start : start + n_locations] = matrix
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
    state_intercept[:n_locations] = float(intercept)

    return StateSpaceModel(
        transition=transition,
        design=design,
        selection=selection,
        state_intercept=state_intercept,
        innovation_covariance=innovation_covariance,
        ar_order=ar_order,
        ma_order=ma_order,
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


@dataclass(frozen=True, slots=True)
class SeasonalKalmanAdmissibility:
    """Expanded multiplicative AR stationarity and MA invertibility result."""

    ar_spectral_radius: float
    ma_inverse_spectral_radius: float
    stability_limit: float
    invertibility_limit: float
    ar_lags: tuple[int, ...]
    ar_matrices: FloatArray
    ma_lags: tuple[int, ...]
    ma_matrices: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "ar_spectral_radius",
            "ma_inverse_spectral_radius",
            "stability_limit",
            "invertibility_limit",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "ar_lags", tuple(int(lag) for lag in self.ar_lags))
        object.__setattr__(self, "ma_lags", tuple(int(lag) for lag in self.ma_lags))
        object.__setattr__(
            self,
            "ar_matrices",
            _freeze_float(self.ar_matrices, name="ar_matrices", ndim=3),
        )
        object.__setattr__(
            self,
            "ma_matrices",
            _freeze_float(self.ma_matrices, name="ma_matrices", ndim=3),
        )

    @property
    def stationary(self) -> bool:
        return self.ar_spectral_radius < self.stability_limit

    @property
    def invertible(self) -> bool:
        return self.ma_inverse_spectral_radius < self.invertibility_limit

    @property
    def admissible(self) -> bool:
        return self.stationary and self.invertible

    def summary(self) -> str:
        return "\n".join(
            [
                "Multiplicative seasonal STARMA admissibility",
                "=" * 56,
                f"AR spectral radius: {self.ar_spectral_radius:.8g}",
                f"AR limit: {self.stability_limit:.8g}",
                f"Stationary: {self.stationary}",
                f"Inverse-MA spectral radius: {self.ma_inverse_spectral_radius:.8g}",
                f"MA limit: {self.invertibility_limit:.8g}",
                f"Invertible: {self.invertible}",
                f"Jointly admissible: {self.admissible}",
            ]
        )


@dataclass(frozen=True, slots=True)
class SeasonalKalmanSTARIMAResult:
    """Immutable multiplicative seasonal Kalman STARIMA fit result."""

    params: FloatArray
    parameter_names: tuple[str, ...]
    raw_optimizer_params: FloatArray
    optimizer_parameter_names: tuple[str, ...]
    intercept: float
    ar_parameters: FloatArray
    seasonal_ar_parameters: FloatArray
    ma_parameters: FloatArray
    seasonal_ma_parameters: FloatArray
    ar_lags: tuple[int, ...]
    ar_matrices: FloatArray
    ma_lags: tuple[int, ...]
    ma_matrices: FloatArray
    innovation_covariance: FloatArray
    covariance_type: CovarianceType
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_params: int
    converged: bool
    n_iterations: int
    n_function_evaluations: int
    optimizer_message: str
    ar_spectral_radius: float
    ma_inverse_spectral_radius: float
    stability_limit: float
    invertibility_limit: float
    stationarity_enforced: bool
    invertibility_enforced: bool
    n_original_rows: int
    n_transformed_rows: int
    original_missing_cells: int
    transformed_missing_cells: int
    original_scale_forecast_available: bool
    filter_result: KalmanFilterResult

    def __post_init__(self) -> None:
        params = _freeze_float(self.params, name="params", ndim=1)
        raw = _freeze_float(
            self.raw_optimizer_params,
            name="raw_optimizer_params",
            ndim=1,
        )
        if len(self.parameter_names) != params.size:
            raise ValueError("parameter_names must match params")
        if len(self.optimizer_parameter_names) != raw.size:
            raise ValueError("optimizer_parameter_names must match raw parameters")
        if not isinstance(self.filter_result, KalmanFilterResult):
            raise TypeError("filter_result must be a KalmanFilterResult")
        for name in (
            "ar_parameters",
            "seasonal_ar_parameters",
            "ma_parameters",
            "seasonal_ma_parameters",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_float(getattr(self, name), name=name, ndim=2),
            )
        object.__setattr__(
            self,
            "ar_matrices",
            _freeze_float(self.ar_matrices, name="ar_matrices", ndim=3),
        )
        object.__setattr__(
            self,
            "ma_matrices",
            _freeze_float(self.ma_matrices, name="ma_matrices", ndim=3),
        )
        object.__setattr__(
            self,
            "innovation_covariance",
            _freeze_float(
                self.innovation_covariance,
                name="innovation_covariance",
                ndim=2,
            ),
        )
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "raw_optimizer_params", raw)
        object.__setattr__(
            self,
            "parameter_names",
            tuple(str(name) for name in self.parameter_names),
        )
        object.__setattr__(
            self,
            "optimizer_parameter_names",
            tuple(str(name) for name in self.optimizer_parameter_names),
        )
        object.__setattr__(self, "ar_lags", tuple(int(lag) for lag in self.ar_lags))
        object.__setattr__(self, "ma_lags", tuple(int(lag) for lag in self.ma_lags))
        object.__setattr__(self, "intercept", float(self.intercept))
        for name in (
            "log_likelihood",
            "aic",
            "bic",
            "ar_spectral_radius",
            "ma_inverse_spectral_radius",
            "stability_limit",
            "invertibility_limit",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "n_observations",
            "n_params",
            "n_iterations",
            "n_function_evaluations",
            "n_original_rows",
            "n_transformed_rows",
            "original_missing_cells",
            "transformed_missing_cells",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(
            self,
            "stationarity_enforced",
            bool(self.stationarity_enforced),
        )
        object.__setattr__(
            self,
            "invertibility_enforced",
            bool(self.invertibility_enforced),
        )
        object.__setattr__(
            self,
            "original_scale_forecast_available",
            bool(self.original_scale_forecast_available),
        )
        object.__setattr__(self, "optimizer_message", str(self.optimizer_message))

    @property
    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"parameter": self.parameter_names, "estimate": self.params}
        ).set_index("parameter")

    @property
    def covariance(self) -> pd.DataFrame:
        names = [
            f"location{index}" for index in range(self.innovation_covariance.shape[0])
        ]
        return pd.DataFrame(
            self.innovation_covariance,
            index=names,
            columns=names,
        )

    @property
    def admissible(self) -> bool:
        return (
            self.ar_spectral_radius < self.stability_limit
            and self.ma_inverse_spectral_radius < self.invertibility_limit
        )

    def summary(self) -> str:
        header = [
            "pySTARMAx multiplicative seasonal Kalman STARIMA result",
            "=" * 72,
            f"Order: {self.order}",
            f"Seasonal order: {self.seasonal_order}",
            f"Original rows: {self.n_original_rows}",
            f"Transformed rows: {self.n_transformed_rows}",
            f"Observed transformed cells: {self.n_observations}",
            f"Log likelihood: {self.log_likelihood:.8g}",
            f"AIC: {self.aic:.8g}",
            f"BIC: {self.bic:.8g}",
            f"AR spectral radius: {self.ar_spectral_radius:.8g}",
            f"Inverse-MA spectral radius: {self.ma_inverse_spectral_radius:.8g}",
            f"Converged: {self.converged}",
            f"Original-scale forecast available: {self.original_scale_forecast_available}",
            "Likelihood scope: transformed conditional ordinary-seasonal history",
            "-" * 72,
            self.coefficients.to_string(),
        ]
        return "\n".join(header)


class SeasonalKalmanSTARIMA:
    """Fit a multiplicative seasonal STARIMA by transformed Kalman likelihood."""

    def __init__(
        self,
        ar_order: int = 1,
        integration_order: int = 0,
        ma_order: int = 0,
        *,
        seasonal_ar_order: int = 0,
        seasonal_integration_order: int = 1,
        seasonal_ma_order: int = 0,
        seasonal_period: int = 12,
        covariance_type: CovarianceType = "scalar",
        include_intercept: bool = True,
        initialization: Literal["stationary", "diffuse"] = "stationary",
        diffuse_scale: float = 1e6,
        enforce_stationarity: bool = True,
        stability_margin: float = 1e-6,
        enforce_invertibility: bool = True,
        invertibility_margin: float = 1e-6,
        max_iter: int = 500,
        tol: float = 1e-9,
    ) -> None:
        self.ar_order = validate_nonnegative_int(ar_order, name="ar_order")
        self.integration_order = validate_nonnegative_int(
            integration_order,
            name="integration_order",
        )
        self.ma_order = validate_nonnegative_int(ma_order, name="ma_order")
        self.seasonal_ar_order = validate_nonnegative_int(
            seasonal_ar_order,
            name="seasonal_ar_order",
        )
        self.seasonal_integration_order = validate_nonnegative_int(
            seasonal_integration_order,
            name="seasonal_integration_order",
        )
        self.seasonal_ma_order = validate_nonnegative_int(
            seasonal_ma_order,
            name="seasonal_ma_order",
        )
        self.seasonal_period = _positive_int(
            seasonal_period,
            name="seasonal_period",
        )
        if covariance_type not in {"scalar", "diagonal", "full"}:
            raise ValueError("covariance_type must be 'scalar', 'diagonal', or 'full'")
        if initialization not in {"stationary", "diffuse"}:
            raise ValueError("initialization must be 'stationary' or 'diffuse'")
        if not np.isfinite(diffuse_scale) or diffuse_scale <= 0.0:
            raise ValueError("diffuse_scale must be positive and finite")
        if not np.isfinite(stability_margin) or not 0.0 < stability_margin < 1.0:
            raise ValueError("stability_margin must be between zero and one")
        if (
            not np.isfinite(invertibility_margin)
            or not 0.0 < invertibility_margin < 1.0
        ):
            raise ValueError("invertibility_margin must be between zero and one")
        self.max_iter = validate_nonnegative_int(max_iter, name="max_iter")
        if self.max_iter == 0:
            raise ValueError("max_iter must be positive")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be positive and finite")
        self.covariance_type = covariance_type
        self.include_intercept = bool(include_intercept)
        self.initialization = initialization
        self.diffuse_scale = float(diffuse_scale)
        self.enforce_stationarity = bool(enforce_stationarity)
        self.stability_margin = float(stability_margin)
        self.enforce_invertibility = bool(enforce_invertibility)
        self.invertibility_margin = float(invertibility_margin)
        self.tol = float(tol)
        self.result_: SeasonalKalmanSTARIMAResult | None = None
        self.state_space_: StateSpaceModel | None = None
        self.filter_result_: KalmanFilterResult | None = None
        self.weights_: SpatialWeights | None = None
        self.data_: FloatArray | None = None
        self.transformed_data_: FloatArray | None = None
        self.differencing_state_: CombinedDifferencingState | None = None

    @property
    def order(self) -> tuple[int, int, int]:
        return self.ar_order, self.integration_order, self.ma_order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return (
            self.seasonal_ar_order,
            self.seasonal_integration_order,
            self.seasonal_ma_order,
            self.seasonal_period,
        )

    def _coefficient_size(self, n_weights: int) -> int:
        return (
            int(self.include_intercept)
            + (
                self.ar_order
                + self.seasonal_ar_order
                + self.ma_order
                + self.seasonal_ma_order
            )
            * n_weights
        )

    def _coefficient_names(self, weights: SpatialWeights) -> tuple[str, ...]:
        names: list[str] = []
        if self.include_intercept:
            names.append("intercept")
        for lag in range(1, self.ar_order + 1):
            names.extend(f"ar.t{lag}.{name}" for name in weights.names)
        for index in range(1, self.seasonal_ar_order + 1):
            lag = index * self.seasonal_period
            names.extend(f"sar.t{lag}.{name}" for name in weights.names)
        for lag in range(1, self.ma_order + 1):
            names.extend(f"ma.t{lag}.{name}" for name in weights.names)
        for index in range(1, self.seasonal_ma_order + 1):
            lag = index * self.seasonal_period
            names.extend(f"sma.t{lag}.{name}" for name in weights.names)
        return tuple(names)

    def _split_coefficients(
        self,
        coefficients: Any,
        *,
        n_weights: int,
    ) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]:
        values = np.asarray(coefficients, dtype=float)
        expected = self._coefficient_size(n_weights)
        if values.shape != (expected,):
            raise ValueError(f"start_params must contain exactly {expected} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("start_params must contain finite values")
        cursor = 0
        intercept = 0.0
        if self.include_intercept:
            intercept = float(values[0])
            cursor = 1

        arrays: list[FloatArray] = []
        for order in (
            self.ar_order,
            self.seasonal_ar_order,
            self.ma_order,
            self.seasonal_ma_order,
        ):
            size = order * n_weights
            arrays.append(
                cast(
                    FloatArray,
                    values[cursor : cursor + size].reshape(order, n_weights),
                )
            )
            cursor += size
        return intercept, arrays[0], arrays[1], arrays[2], arrays[3]

    def _expanded(
        self,
        coefficients: FloatArray,
        weights: SpatialWeights,
    ) -> tuple[
        float,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        tuple[int, ...],
        FloatArray,
        tuple[int, ...],
        FloatArray,
    ]:
        intercept, ar, sar, ma, sma = self._split_coefficients(
            coefficients,
            n_weights=len(weights),
        )
        ar_terms = expand_multiplicative_operators(
            ar,
            sar,
            weights,
            seasonal_period=self.seasonal_period,
            kind="ar",
        )
        ma_terms = expand_multiplicative_operators(
            ma,
            sma,
            weights,
            seasonal_period=self.seasonal_period,
            kind="ma",
        )
        ar_lags, ar_matrices = _aggregate_operators(
            ar_terms,
            n_locations=weights.n_locations,
        )
        ma_lags, ma_matrices = _aggregate_operators(
            ma_terms,
            n_locations=weights.n_locations,
        )
        return (
            intercept,
            ar,
            sar,
            ma,
            sma,
            ar_lags,
            ar_matrices,
            ma_lags,
            ma_matrices,
        )

    def _initial_values(
        self,
        transformed: FloatArray,
        weights: SpatialWeights,
    ) -> tuple[FloatArray, FloatArray]:
        means = np.nanmean(transformed, axis=0)
        filled = np.where(np.isfinite(transformed), transformed, means)
        intercept = float(np.mean(filled)) if self.include_intercept else 0.0
        coefficients = np.zeros(self._coefficient_size(len(weights)), dtype=float)
        if self.include_intercept:
            coefficients[0] = intercept
        centered = filled - intercept
        covariance = _regularize_covariance(
            np.atleast_2d(np.cov(centered, rowvar=False, ddof=1)),
            n_locations=transformed.shape[1],
        )
        return cast(FloatArray, coefficients), covariance

    def _shrink_initial(
        self,
        coefficients: FloatArray,
        weights: SpatialWeights,
    ) -> FloatArray:
        values = np.asarray(coefficients, dtype=float).copy()
        coefficient_start = int(self.include_intercept)
        ar_size = (self.ar_order + self.seasonal_ar_order) * len(weights)
        ma_start = coefficient_start + ar_size
        ma_size = (self.ma_order + self.seasonal_ma_order) * len(weights)
        ar_target = 1.0 - max(self.stability_margin * 10.0, 1e-4)
        ma_target = 1.0 - max(self.invertibility_margin * 10.0, 1e-4)
        for _ in range(60):
            expanded = self._expanded(cast(FloatArray, values), weights)
            ar_radius = _companion_radius(
                expanded[5],
                expanded[6],
                inverse_sign=False,
                n_locations=weights.n_locations,
            )
            if not self.enforce_stationarity or ar_radius < ar_target:
                break
            values[coefficient_start:ma_start] *= 0.8
        for _ in range(80):
            expanded = self._expanded(cast(FloatArray, values), weights)
            ma_radius = _companion_radius(
                expanded[7],
                expanded[8],
                inverse_sign=True,
                n_locations=weights.n_locations,
            )
            if not self.enforce_invertibility or ma_radius < ma_target:
                break
            values[ma_start : ma_start + ma_size] *= 0.8
        return cast(FloatArray, values)

    def fit(
        self,
        data: Any,
        weights: Any,
        *,
        start_params: Any | None = None,
        start_covariance: Any | None = None,
    ) -> SeasonalKalmanSTARIMAResult:
        observations, transformed_raw, state = _combined_difference_incomplete(
            data,
            ordinary_order=self.integration_order,
            seasonal_order=self.seasonal_integration_order,
            seasonal_period=self.seasonal_period,
        )
        transformed = _validate_incomplete_data(transformed_raw)
        resolved_weights = coerce_weights(
            weights,
            n_locations=transformed.shape[1],
        )
        coefficient_start, covariance_start = self._initial_values(
            transformed,
            resolved_weights,
        )
        if start_params is not None:
            coefficient_start = np.asarray(start_params, dtype=float)
            self._split_coefficients(
                coefficient_start,
                n_weights=len(resolved_weights),
            )
        coefficient_start = self._shrink_initial(
            cast(FloatArray, coefficient_start),
            resolved_weights,
        )
        if start_covariance is not None:
            covariance_start = _regularize_covariance(
                start_covariance,
                n_locations=transformed.shape[1],
            )

        codec = _CovarianceCodec(self.covariance_type, transformed.shape[1])
        coefficient_size = int(coefficient_start.size)
        raw_start = np.concatenate([coefficient_start, codec.pack(covariance_start)])
        coefficient_names = self._coefficient_names(resolved_weights)
        optimizer_names = coefficient_names + codec.names
        bounds = [(None, None)] * coefficient_size + codec.bounds
        stability_limit = 1.0 - self.stability_margin
        invertibility_limit = 1.0 - self.invertibility_margin
        invalid_base = 1e12

        def decode(raw: FloatArray) -> tuple[
            float,
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            tuple[int, ...],
            FloatArray,
            tuple[int, ...],
            FloatArray,
            FloatArray,
            StateSpaceModel,
            float,
            float,
        ]:
            expanded = self._expanded(
                cast(FloatArray, raw[:coefficient_size]),
                resolved_weights,
            )
            covariance = codec.unpack(raw[coefficient_size:])
            state_space = _operator_state_space(
                expanded[5],
                expanded[6],
                expanded[7],
                expanded[8],
                covariance,
                intercept=expanded[0],
            )
            ar_radius = _companion_radius(
                expanded[5],
                expanded[6],
                inverse_sign=False,
                n_locations=resolved_weights.n_locations,
            )
            ma_radius = _companion_radius(
                expanded[7],
                expanded[8],
                inverse_sign=True,
                n_locations=resolved_weights.n_locations,
            )
            return (*expanded, covariance, state_space, ar_radius, ma_radius)

        def objective(raw: FloatArray) -> float:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    decoded = decode(raw)
                    ar_radius = decoded[-2]
                    ma_radius = decoded[-1]
                    squared_excess = 0.0
                    if self.enforce_stationarity and ar_radius >= stability_limit:
                        squared_excess += (ar_radius - stability_limit) ** 2
                    if self.enforce_invertibility and ma_radius >= invertibility_limit:
                        squared_excess += (ma_radius - invertibility_limit) ** 2
                    if squared_excess > 0.0:
                        return float(
                            invalid_base
                            + invalid_base * squared_excess
                            + 1e-8 * (raw @ raw)
                        )
                    filtered = kalman_filter(
                        transformed,
                        decoded[-3],
                        initialization=cast(Initialization, self.initialization),
                        diffuse_scale=self.diffuse_scale,
                    )
                    value = -filtered.log_likelihood
                    return float(value) if np.isfinite(value) else invalid_base
            except (
                ValueError,
                np.linalg.LinAlgError,
                FloatingPointError,
                OverflowError,
            ):
                return float(invalid_base + 1e-8 * (raw @ raw))

        optimized = minimize(
            objective,
            raw_start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol, "maxls": 50},
        )
        raw_final = cast(FloatArray, np.asarray(optimized.x, dtype=float))
        decoded = decode(raw_final)
        ar_radius = decoded[-2]
        ma_radius = decoded[-1]
        if self.enforce_stationarity and ar_radius >= stability_limit:
            raise RuntimeError("optimizer returned a non-stationary final candidate")
        if self.enforce_invertibility and ma_radius >= invertibility_limit:
            raise RuntimeError("optimizer returned a non-invertible final candidate")
        state_space = decoded[-3]
        filtered = kalman_filter(
            transformed,
            state_space,
            initialization=cast(Initialization, self.initialization),
            diffuse_scale=self.diffuse_scale,
        )
        n_params = int(raw_final.size)
        log_likelihood = filtered.log_likelihood
        aic = -2.0 * log_likelihood + 2.0 * n_params
        bic = -2.0 * log_likelihood + np.log(filtered.n_observations) * n_params
        result = SeasonalKalmanSTARIMAResult(
            params=raw_final[:coefficient_size],
            parameter_names=coefficient_names,
            raw_optimizer_params=raw_final,
            optimizer_parameter_names=optimizer_names,
            intercept=decoded[0],
            ar_parameters=decoded[1],
            seasonal_ar_parameters=decoded[2],
            ma_parameters=decoded[3],
            seasonal_ma_parameters=decoded[4],
            ar_lags=decoded[5],
            ar_matrices=decoded[6],
            ma_lags=decoded[7],
            ma_matrices=decoded[8],
            innovation_covariance=decoded[9],
            covariance_type=self.covariance_type,
            order=self.order,
            seasonal_order=self.seasonal_order,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_observations=filtered.n_observations,
            n_params=n_params,
            converged=bool(optimized.success),
            n_iterations=int(getattr(optimized, "nit", 0)),
            n_function_evaluations=int(getattr(optimized, "nfev", 0)),
            optimizer_message=str(optimized.message),
            ar_spectral_radius=ar_radius,
            ma_inverse_spectral_radius=ma_radius,
            stability_limit=stability_limit,
            invertibility_limit=invertibility_limit,
            stationarity_enforced=self.enforce_stationarity,
            invertibility_enforced=self.enforce_invertibility,
            n_original_rows=observations.shape[0],
            n_transformed_rows=transformed.shape[0],
            original_missing_cells=int(np.count_nonzero(~np.isfinite(observations))),
            transformed_missing_cells=int(np.count_nonzero(~np.isfinite(transformed))),
            original_scale_forecast_available=state is not None,
            filter_result=filtered,
        )
        self.result_ = result
        self.state_space_ = state_space
        self.filter_result_ = filtered
        self.weights_ = resolved_weights
        self.data_ = observations.copy()
        self.transformed_data_ = transformed.copy()
        self.differencing_state_ = state
        return result

    def _require_fit(
        self,
    ) -> tuple[SeasonalKalmanSTARIMAResult, StateSpaceModel, KalmanFilterResult]:
        if (
            self.result_ is None
            or self.state_space_ is None
            or self.filter_result_ is None
        ):
            raise RuntimeError("fit must be called before using the fitted model")
        return self.result_, self.state_space_, self.filter_result_

    def _transform_new(self, data: Any) -> FloatArray:
        _observations, transformed, _state = _combined_difference_incomplete(
            data,
            ordinary_order=self.integration_order,
            seasonal_order=self.seasonal_integration_order,
            seasonal_period=self.seasonal_period,
        )
        return _validate_incomplete_data(transformed)

    def admissibility(self) -> SeasonalKalmanAdmissibility:
        result, _state_space, _filtered = self._require_fit()
        return SeasonalKalmanAdmissibility(
            ar_spectral_radius=result.ar_spectral_radius,
            ma_inverse_spectral_radius=result.ma_inverse_spectral_radius,
            stability_limit=result.stability_limit,
            invertibility_limit=result.invertibility_limit,
            ar_lags=result.ar_lags,
            ar_matrices=result.ar_matrices,
            ma_lags=result.ma_lags,
            ma_matrices=result.ma_matrices,
        )

    def to_state_space(self) -> StateSpaceModel:
        _result, state_space, _filtered = self._require_fit()
        return state_space

    def filter(self, data: Any | None = None) -> KalmanFilterResult:
        _result, state_space, training_filter = self._require_fit()
        if data is None:
            return training_filter
        return kalman_filter(
            self._transform_new(data),
            state_space,
            initialization=cast(Initialization, self.initialization),
            diffuse_scale=self.diffuse_scale,
        )

    def smooth(
        self,
        data: Any | None = None,
        *,
        rcond: float = 1e-10,
    ) -> KalmanSmootherResult:
        return kalman_smoother(self.filter(data), rcond=rcond)

    def smooth_innovation_disturbances(
        self,
        data: Any | None = None,
        *,
        rcond: float = 1e-10,
    ) -> InnovationDisturbanceResult:
        return innovation_disturbance_smoother(
            self.smooth(data, rcond=rcond),
            rcond=rcond,
        )

    def predict_differenced(self, steps: int = 1) -> FloatArray:
        steps = validate_nonnegative_int(steps, name="steps")
        if steps == 0:
            raise ValueError("steps must be positive")
        _result, state_space, filtered = self._require_fit()
        state = filtered.filtered_state[-1].copy()
        forecasts = np.empty((steps, state_space.n_locations), dtype=float)
        for step_index in range(steps):
            state = state_space.state_intercept + state_space.transition @ state
            forecasts[step_index] = state_space.design @ state
        return cast(FloatArray, forecasts)

    def predict(self, steps: int = 1) -> FloatArray:
        self._require_fit()
        if self.differencing_state_ is None:
            raise RuntimeError(
                "original-scale prediction requires finite ordinary and seasonal "
                "terminal anchors; use predict_differenced() or refit with finite "
                "trailing data"
            )
        return self.differencing_state_.inverse_forecast(
            self.predict_differenced(steps=steps)
        )

    def fitted_differenced(self) -> FloatArray:
        return self.filter().predicted_observations

    def fitted_original(self) -> FloatArray:
        self._require_fit()
        if self.data_ is None:
            raise RuntimeError("training data are unavailable")
        return _restore_incomplete_fitted(
            self.data_,
            self.fitted_differenced(),
            ordinary_order=self.integration_order,
            seasonal_order=self.seasonal_integration_order,
            seasonal_period=self.seasonal_period,
        )


__all__ = [
    "SeasonalKalmanAdmissibility",
    "SeasonalKalmanSTARIMA",
    "SeasonalKalmanSTARIMAResult",
]
