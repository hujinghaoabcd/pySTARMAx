# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Gaussian maximum-likelihood estimation for stationary STARMA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.state_space import (
    Initialization,
    KalmanFilterResult,
    StateSpaceModel,
    build_starma_state_space,
    kalman_filter,
)
from pystarmax.weights import SpatialWeights, coerce_weights

CovarianceType: TypeAlias = Literal["scalar", "diagonal", "full"]


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _freeze_covariance(value: Any, *, n_locations: int) -> FloatArray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (n_locations, n_locations):
        raise ValueError("innovation_covariance has an invalid shape")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("innovation_covariance must contain finite values")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) <= 0.0:
        if float(np.min(eigenvalues, initial=0.0)) < -1e-10 * scale:
            raise ValueError("innovation_covariance must be positive definite")
        raise ValueError("innovation_covariance must be strictly positive definite")
    frozen = np.ascontiguousarray(covariance, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _validate_incomplete_data(data: Any) -> FloatArray:
    observations = np.asarray(data, dtype=float)
    if observations.ndim != 2:
        raise ValueError("data must be a two-dimensional array")
    if observations.shape[0] < 3:
        raise ValueError("data must contain at least three time observations")
    if observations.shape[1] < 1:
        raise ValueError("data must contain at least one location")
    if np.any(np.isinf(observations)):
        raise ValueError("data must not contain infinite values")
    observed_per_location = np.sum(np.isfinite(observations), axis=0)
    if np.any(observed_per_location < 2):
        raise ValueError("each location must contain at least two observed values")
    return cast(FloatArray, np.ascontiguousarray(observations, dtype=float))


def _regularize_covariance(covariance: Any, *, n_locations: int) -> FloatArray:
    array = np.asarray(covariance, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError("start_covariance scalar must be positive and finite")
        array = np.eye(n_locations, dtype=float) * scalar
    elif array.ndim == 1:
        if array.shape != (n_locations,):
            raise ValueError("start_covariance vector must match the locations")
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError("start_covariance variances must be positive and finite")
        array = np.diag(array)
    elif array.shape != (n_locations, n_locations):
        raise ValueError("start_covariance matrix must match the locations")
    if not np.all(np.isfinite(array)):
        raise ValueError("start_covariance must contain finite values")
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.mean(np.abs(np.diag(symmetric)))))
    floor = np.finfo(float).eps * scale * 1000.0
    clipped = np.clip(eigenvalues, floor, np.inf)
    regularized = (eigenvectors * clipped) @ eigenvectors.T
    return cast(FloatArray, 0.5 * (regularized + regularized.T))


class _CovarianceCodec:
    def __init__(self, covariance_type: CovarianceType, n_locations: int) -> None:
        if covariance_type not in {"scalar", "diagonal", "full"}:
            raise ValueError("covariance_type must be 'scalar', 'diagonal', or 'full'")
        self.covariance_type = covariance_type
        self.n_locations = n_locations

    @property
    def size(self) -> int:
        if self.covariance_type == "scalar":
            return 1
        if self.covariance_type == "diagonal":
            return self.n_locations
        return self.n_locations * (self.n_locations + 1) // 2

    @property
    def names(self) -> tuple[str, ...]:
        if self.covariance_type == "scalar":
            return ("cov.log_std",)
        if self.covariance_type == "diagonal":
            return tuple(
                f"cov.log_std.location{location}"
                for location in range(self.n_locations)
            )
        names: list[str] = []
        for row in range(self.n_locations):
            for column in range(row + 1):
                if row == column:
                    names.append(f"cov.log_cholesky.location{row}")
                else:
                    names.append(f"cov.cholesky.location{row}.{column}")
        return tuple(names)

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        if self.covariance_type == "scalar":
            return [(-20.0, 20.0)]
        if self.covariance_type == "diagonal":
            return [(-20.0, 20.0)] * self.n_locations
        bounds: list[tuple[float | None, float | None]] = []
        for row in range(self.n_locations):
            for column in range(row + 1):
                bounds.append((-20.0, 20.0) if row == column else (None, None))
        return bounds

    def pack(self, covariance: Any) -> FloatArray:
        regularized = _regularize_covariance(
            covariance,
            n_locations=self.n_locations,
        )
        if self.covariance_type == "scalar":
            variance = float(np.mean(np.diag(regularized)))
            return np.array([0.5 * np.log(variance)], dtype=float)
        if self.covariance_type == "diagonal":
            return cast(FloatArray, 0.5 * np.log(np.diag(regularized)))
        factor = np.linalg.cholesky(regularized)
        packed: list[float] = []
        for row in range(self.n_locations):
            for column in range(row + 1):
                value = float(factor[row, column])
                packed.append(np.log(value) if row == column else value)
        return np.asarray(packed, dtype=float)

    def unpack(self, parameters: Any) -> FloatArray:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.size,):
            raise ValueError("covariance parameter vector has an invalid shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("covariance parameters must be finite")
        if self.covariance_type == "scalar":
            standard_deviation = float(np.exp(values[0]))
            return np.eye(self.n_locations, dtype=float) * standard_deviation**2
        if self.covariance_type == "diagonal":
            standard_deviations = np.exp(values)
            return cast(FloatArray, np.diag(standard_deviations**2))
        factor = np.zeros((self.n_locations, self.n_locations), dtype=float)
        cursor = 0
        for row in range(self.n_locations):
            for column in range(row + 1):
                value = float(values[cursor])
                factor[row, column] = np.exp(value) if row == column else value
                cursor += 1
        return cast(FloatArray, factor @ factor.T)


@dataclass(frozen=True, slots=True)
class KalmanSTARMAResult:
    """Immutable result from :class:`KalmanSTARMA`."""

    params: FloatArray
    parameter_names: tuple[str, ...]
    raw_optimizer_params: FloatArray
    optimizer_parameter_names: tuple[str, ...]
    intercept: float
    ar_parameters: FloatArray
    ma_parameters: FloatArray
    innovation_covariance: FloatArray
    covariance_type: str
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_params: int
    converged: bool
    n_iterations: int
    n_function_evaluations: int
    optimizer_method: str
    optimizer_message: str
    spectral_radius: float
    filter_result: KalmanFilterResult

    def __post_init__(self) -> None:
        params = _freeze_float(self.params, name="params", ndim=1)
        raw = _freeze_float(
            self.raw_optimizer_params,
            name="raw_optimizer_params",
            ndim=1,
        )
        ar_parameters = _freeze_float(
            self.ar_parameters,
            name="ar_parameters",
            ndim=2,
        )
        ma_parameters = _freeze_float(
            self.ma_parameters,
            name="ma_parameters",
            ndim=2,
        )
        covariance = _freeze_covariance(
            self.innovation_covariance,
            n_locations=self.filter_result.model.n_locations,
        )
        if len(self.parameter_names) != params.size:
            raise ValueError("parameter_names must match params")
        if len(self.optimizer_parameter_names) != raw.size:
            raise ValueError("optimizer_parameter_names must match raw_optimizer_params")
        if int(self.n_params) != raw.size:
            raise ValueError("n_params must match raw_optimizer_params")
        if int(self.n_observations) != self.filter_result.n_observations:
            raise ValueError("n_observations must match filter_result")
        for name, value in (
            ("intercept", self.intercept),
            ("log_likelihood", self.log_likelihood),
            ("aic", self.aic),
            ("bic", self.bic),
            ("spectral_radius", self.spectral_radius),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "raw_optimizer_params", raw)
        object.__setattr__(self, "ar_parameters", ar_parameters)
        object.__setattr__(self, "ma_parameters", ma_parameters)
        object.__setattr__(self, "innovation_covariance", covariance)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "log_likelihood", float(self.log_likelihood))
        object.__setattr__(self, "aic", float(self.aic))
        object.__setattr__(self, "bic", float(self.bic))
        object.__setattr__(self, "n_observations", int(self.n_observations))
        object.__setattr__(self, "n_params", int(self.n_params))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "n_iterations", int(self.n_iterations))
        object.__setattr__(
            self,
            "n_function_evaluations",
            int(self.n_function_evaluations),
        )
        object.__setattr__(self, "spectral_radius", float(self.spectral_radius))

    @property
    def coefficients(self) -> pd.DataFrame:
        """Dynamic coefficient table."""
        return pd.DataFrame(
            {"coefficient": self.params},
            index=pd.Index(self.parameter_names, name="parameter"),
        )

    @property
    def covariance(self) -> pd.DataFrame:
        """Innovation covariance table."""
        labels = [
            f"location{location}"
            for location in range(self.innovation_covariance.shape[0])
        ]
        return pd.DataFrame(
            self.innovation_covariance,
            index=pd.Index(labels, name="location"),
            columns=labels,
        )

    def summary(self) -> str:
        """Return a compact plain-text maximum-likelihood summary."""
        header = [
            "pySTARMAx Kalman maximum-likelihood result",
            "=" * 72,
            f"Optimizer: {self.optimizer_method}",
            f"Covariance: {self.covariance_type}",
            f"Observed cells: {self.n_observations}",
            f"Estimated parameters: {self.n_params}",
            f"Converged: {self.converged} ({self.n_iterations} iteration(s))",
            f"Function evaluations: {self.n_function_evaluations}",
            f"Transition spectral radius: {self.spectral_radius:.6f}",
            f"Log likelihood: {self.log_likelihood:.6f}",
            f"AIC: {self.aic:.6f}",
            f"BIC: {self.bic:.6f}",
            f"Message: {self.optimizer_message}",
            "-" * 72,
        ]
        coefficients = self.coefficients.to_string(
            float_format=lambda value: f"{value: .6f}"
        )
        covariance = self.covariance.to_string(
            float_format=lambda value: f"{value: .6f}"
        )
        return "\n".join(
            header
            + [coefficients, "-" * 72, "Innovation covariance", covariance]
        )


class KalmanSTARMA:
    """Estimate a stationary Gaussian STARMA model by Kalman likelihood.

    Parameters
    ----------
    ar_order, ma_order:
        Temporal AR and MA orders. All supplied spatial lags are included at
        every positive temporal order. Both may be zero for white noise.
    covariance_type:
        ``"scalar"`` estimates one variance shared by all locations,
        ``"diagonal"`` estimates one variance per location, and ``"full"``
        estimates a positive-definite covariance through a Cholesky factor.
    include_intercept:
        Estimate one common intercept across locations.
    initialization:
        ``"stationary"`` or approximate ``"diffuse"`` Kalman initialization.
    enforce_stationarity:
        Reject candidates whose transition spectral radius is not below
        ``1 - stability_margin``.
    """

    def __init__(
        self,
        ar_order: int = 1,
        ma_order: int = 0,
        *,
        covariance_type: CovarianceType = "scalar",
        include_intercept: bool = True,
        initialization: Literal["stationary", "diffuse"] = "stationary",
        diffuse_scale: float = 1e6,
        enforce_stationarity: bool = True,
        stability_margin: float = 1e-6,
        max_iter: int = 500,
        tol: float = 1e-9,
    ) -> None:
        self.ar_order = validate_nonnegative_int(ar_order, name="ar_order")
        self.ma_order = validate_nonnegative_int(ma_order, name="ma_order")
        if covariance_type not in {"scalar", "diagonal", "full"}:
            raise ValueError("covariance_type must be 'scalar', 'diagonal', or 'full'")
        if initialization not in {"stationary", "diffuse"}:
            raise ValueError("initialization must be 'stationary' or 'diffuse'")
        if not np.isfinite(diffuse_scale) or diffuse_scale <= 0.0:
            raise ValueError("diffuse_scale must be positive and finite")
        if not np.isfinite(stability_margin) or not 0.0 < stability_margin < 1.0:
            raise ValueError("stability_margin must be between zero and one")
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
        self.tol = float(tol)
        self.result_: KalmanSTARMAResult | None = None
        self.state_space_: StateSpaceModel | None = None
        self.filter_result_: KalmanFilterResult | None = None
        self.weights_: SpatialWeights | None = None
        self.data_: FloatArray | None = None

    def _coefficient_size(self, n_weights: int) -> int:
        return (
            int(self.include_intercept)
            + self.ar_order * n_weights
            + self.ma_order * n_weights
        )

    def _coefficient_names(self, weights: SpatialWeights) -> tuple[str, ...]:
        names: list[str] = []
        if self.include_intercept:
            names.append("intercept")
        for temporal_lag in range(1, self.ar_order + 1):
            for spatial_name in weights.names:
                names.append(f"ar.t{temporal_lag}.{spatial_name}")
        for temporal_lag in range(1, self.ma_order + 1):
            for spatial_name in weights.names:
                names.append(f"ma.t{temporal_lag}.{spatial_name}")
        return tuple(names)

    def _split_coefficients(
        self,
        coefficients: Any,
        *,
        n_weights: int,
    ) -> tuple[float, FloatArray, FloatArray]:
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
        ar_size = self.ar_order * n_weights
        ar_parameters = values[cursor : cursor + ar_size].reshape(
            self.ar_order,
            n_weights,
        )
        cursor += ar_size
        ma_parameters = values[cursor:].reshape(self.ma_order, n_weights)
        return (
            intercept,
            np.asarray(ar_parameters, dtype=float),
            np.asarray(ma_parameters, dtype=float),
        )

    def _join_coefficients(
        self,
        intercept: float,
        ar_parameters: FloatArray,
        ma_parameters: FloatArray,
    ) -> FloatArray:
        pieces: list[FloatArray] = []
        if self.include_intercept:
            pieces.append(np.array([intercept], dtype=float))
        pieces.extend([ar_parameters.reshape(-1), ma_parameters.reshape(-1)])
        if not pieces:
            return np.empty(0, dtype=float)
        return cast(FloatArray, np.concatenate(pieces))

    def _initial_values(
        self,
        observations: FloatArray,
        weights: SpatialWeights,
    ) -> tuple[FloatArray, FloatArray]:
        means = np.nanmean(observations, axis=0)
        filled = np.where(np.isfinite(observations), observations, means)
        n_weights = len(weights)
        intercept = float(np.mean(filled)) if self.include_intercept else 0.0
        ar_parameters = np.zeros((self.ar_order, n_weights), dtype=float)
        ma_parameters = np.zeros((self.ma_order, n_weights), dtype=float)
        covariance_source: FloatArray

        if self.ar_order > 0 or self.ma_order > 0:
            try:
                from pystarmax.models.state_space_starma import (
                    STARMA as ConditionalSTARMA,
                )

                conditional = ConditionalSTARMA(
                    ar_order=self.ar_order,
                    ma_order=self.ma_order,
                    include_intercept=self.include_intercept,
                    max_iter=min(100, self.max_iter),
                    tol=max(self.tol, 1e-8),
                )
                conditional.fit(filled, weights)
                (
                    _weights,
                    _data,
                    _residuals,
                    intercept,
                    ar_parameters,
                    ma_parameters,
                ) = conditional._fitted_components()
                if conditional.result_ is None:
                    raise RuntimeError("conditional initializer did not return a result")
                covariance_source = conditional.result_.innovation_covariance
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                centered = filled - intercept
                covariance_source = np.atleast_2d(
                    np.cov(centered, rowvar=False, ddof=1)
                ).astype(float)
        else:
            centered = filled - intercept
            covariance_source = np.atleast_2d(
                np.cov(centered, rowvar=False, ddof=1)
            ).astype(float)

        covariance = _regularize_covariance(
            covariance_source,
            n_locations=observations.shape[1],
        )
        if self.ar_order:
            target = 1.0 - max(self.stability_margin * 10.0, 1e-4)
            for _ in range(20):
                candidate = build_starma_state_space(
                    ar_parameters,
                    ma_parameters,
                    weights,
                    covariance,
                    intercept=intercept,
                )
                spectral_radius = float(
                    np.max(np.abs(np.linalg.eigvals(candidate.transition)), initial=0.0)
                )
                if spectral_radius < target:
                    break
                ar_parameters *= 0.9 * target / max(spectral_radius, 1e-12)
        coefficients = self._join_coefficients(
            intercept,
            ar_parameters,
            ma_parameters,
        )
        return coefficients, covariance

    def fit(
        self,
        data: Any,
        weights: Any,
        *,
        start_params: Any | None = None,
        start_covariance: Any | None = None,
    ) -> KalmanSTARMAResult:
        """Estimate model parameters by minimizing the negative Kalman likelihood."""
        observations = _validate_incomplete_data(data)
        resolved_weights = coerce_weights(
            weights,
            n_locations=observations.shape[1],
        )
        n_weights = len(resolved_weights)
        coefficient_start, covariance_start = self._initial_values(
            observations,
            resolved_weights,
        )
        if start_params is not None:
            coefficient_start = np.asarray(start_params, dtype=float)
            self._split_coefficients(coefficient_start, n_weights=n_weights)
        if start_covariance is not None:
            covariance_start = _regularize_covariance(
                start_covariance,
                n_locations=observations.shape[1],
            )

        codec = _CovarianceCodec(
            self.covariance_type,
            observations.shape[1],
        )
        covariance_parameters = codec.pack(covariance_start)
        raw_start = np.concatenate([coefficient_start, covariance_parameters])
        coefficient_names = self._coefficient_names(resolved_weights)
        optimizer_names = coefficient_names + codec.names
        coefficient_size = coefficient_start.size
        bounds = [(None, None)] * coefficient_size + codec.bounds
        stability_limit = 1.0 - self.stability_margin
        invalid_base = 1e12

        def decode(
            raw: FloatArray,
        ) -> tuple[float, FloatArray, FloatArray, FloatArray, StateSpaceModel, float]:
            intercept, ar_parameters, ma_parameters = self._split_coefficients(
                raw[:coefficient_size],
                n_weights=n_weights,
            )
            covariance = codec.unpack(raw[coefficient_size:])
            state_space = build_starma_state_space(
                ar_parameters,
                ma_parameters,
                resolved_weights,
                covariance,
                intercept=intercept,
            )
            spectral_radius = float(
                np.max(
                    np.abs(np.linalg.eigvals(state_space.transition)),
                    initial=0.0,
                )
            )
            return (
                intercept,
                ar_parameters,
                ma_parameters,
                covariance,
                state_space,
                spectral_radius,
            )

        def objective(raw: FloatArray) -> float:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    (
                        _intercept,
                        _ar,
                        _ma,
                        _covariance,
                        state_space,
                        spectral_radius,
                    ) = decode(raw)
                    if self.enforce_stationarity and spectral_radius >= stability_limit:
                        excess = spectral_radius - stability_limit
                        return float(
                            invalid_base
                            + invalid_base * excess**2
                            + 1e-8 * (raw @ raw)
                        )
                    filtered = kalman_filter(
                        observations,
                        state_space,
                        initialization=cast(Initialization, self.initialization),
                        diffuse_scale=self.diffuse_scale,
                    )
                    value = -filtered.log_likelihood
                    if not np.isfinite(value):
                        return invalid_base
                    return float(value)
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
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "maxls": 50,
            },
        )
        raw_final = np.asarray(optimized.x, dtype=float)
        (
            intercept,
            ar_parameters,
            ma_parameters,
            covariance,
            state_space,
            spectral_radius,
        ) = decode(raw_final)
        if self.enforce_stationarity and spectral_radius >= stability_limit:
            raise RuntimeError("optimizer returned a non-stationary final candidate")
        filtered = kalman_filter(
            observations,
            state_space,
            initialization=cast(Initialization, self.initialization),
            diffuse_scale=self.diffuse_scale,
        )
        coefficients = raw_final[:coefficient_size]
        n_params = int(raw_final.size)
        n_observations = filtered.n_observations
        log_likelihood = filtered.log_likelihood
        aic = -2.0 * log_likelihood + 2.0 * n_params
        bic = -2.0 * log_likelihood + np.log(n_observations) * n_params
        result = KalmanSTARMAResult(
            params=coefficients,
            parameter_names=coefficient_names,
            raw_optimizer_params=raw_final,
            optimizer_parameter_names=optimizer_names,
            intercept=intercept,
            ar_parameters=ar_parameters,
            ma_parameters=ma_parameters,
            innovation_covariance=covariance,
            covariance_type=self.covariance_type,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_observations=n_observations,
            n_params=n_params,
            converged=bool(optimized.success),
            n_iterations=int(getattr(optimized, "nit", 0)),
            n_function_evaluations=int(getattr(optimized, "nfev", 0)),
            optimizer_method="L-BFGS-B",
            optimizer_message=str(optimized.message),
            spectral_radius=spectral_radius,
            filter_result=filtered,
        )
        self.result_ = result
        self.state_space_ = state_space
        self.filter_result_ = filtered
        self.weights_ = resolved_weights
        self.data_ = observations.copy()
        return result

    def _fitted_components(
        self,
    ) -> tuple[StateSpaceModel, KalmanFilterResult]:
        if self.state_space_ is None or self.filter_result_ is None:
            raise RuntimeError("fit must be called before using the fitted model")
        return self.state_space_, self.filter_result_

    def to_state_space(self) -> StateSpaceModel:
        """Return the fitted state-space representation."""
        state_space, _filtered = self._fitted_components()
        return state_space

    def filter(self, data: Any | None = None) -> KalmanFilterResult:
        """Filter training data or a new incomplete observation matrix."""
        state_space, training_filter = self._fitted_components()
        if data is None:
            return training_filter
        return kalman_filter(
            data,
            state_space,
            initialization=cast(Initialization, self.initialization),
            diffuse_scale=self.diffuse_scale,
        )

    def predict(self, steps: int = 1) -> FloatArray:
        """Generate recursive conditional-mean forecasts from the filtered state."""
        steps = validate_nonnegative_int(steps, name="steps")
        if steps == 0:
            raise ValueError("steps must be positive")
        state_space, filtered = self._fitted_components()
        state = filtered.filtered_state[-1].copy()
        forecasts = np.empty((steps, state_space.n_locations), dtype=float)
        for step_index in range(steps):
            state = state_space.state_intercept + state_space.transition @ state
            forecasts[step_index] = state_space.design @ state
        return forecasts
