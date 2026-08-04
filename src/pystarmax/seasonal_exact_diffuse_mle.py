# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Optimizer-facing seasonal exact diffuse maximum likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pystarmax._maximum_likelihood_utils import (
    CovarianceType,
    _CovarianceCodec,
    _freeze_covariance,
    _freeze_float,
    _regularize_covariance,
    _validate_incomplete_data,
)
from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.exact_diffuse import ExactDiffuseFilterResult
from pystarmax.exact_seasonal_integrated import (
    ExactSeasonalIntegratedStateSpace,
    build_exact_seasonal_integrated_state_space,
)
from pystarmax.seasonal_maximum_likelihood import (
    SeasonalKalmanAdmissibility,
    SeasonalKalmanSTARIMA,
    _combined_difference_incomplete,
    _companion_radius,
    _operator_state_space,
)
from pystarmax.state_space import StateSpaceModel
from pystarmax.weights import SpatialWeights, coerce_weights


@dataclass(frozen=True, slots=True)
class SeasonalExactDiffuseKalmanSTARIMAResult:
    """Immutable multiplicative seasonal exact diffuse fit result."""

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
    integration_order: int
    seasonal_integration_order: int
    seasonal_period: int
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_diffuse_observations: int
    n_params: int
    converged: bool
    n_iterations: int
    n_function_evaluations: int
    optimizer_method: str
    optimizer_message: str
    ar_spectral_radius: float
    ma_inverse_spectral_radius: float
    stability_limit: float
    invertibility_limit: float
    stationarity_enforced: bool
    invertibility_enforced: bool
    n_original_rows: int
    original_missing_cells: int
    filter_result: ExactDiffuseFilterResult
    transformed_state_space: StateSpaceModel
    integrated_state_space: ExactSeasonalIntegratedStateSpace

    def __post_init__(self) -> None:
        params = _freeze_float(self.params, name="params", ndim=1)
        raw = _freeze_float(
            self.raw_optimizer_params,
            name="raw_optimizer_params",
            ndim=1,
        )
        arrays = {
            "ar_parameters": _freeze_float(
                self.ar_parameters,
                name="ar_parameters",
                ndim=2,
            ),
            "seasonal_ar_parameters": _freeze_float(
                self.seasonal_ar_parameters,
                name="seasonal_ar_parameters",
                ndim=2,
            ),
            "ma_parameters": _freeze_float(
                self.ma_parameters,
                name="ma_parameters",
                ndim=2,
            ),
            "seasonal_ma_parameters": _freeze_float(
                self.seasonal_ma_parameters,
                name="seasonal_ma_parameters",
                ndim=2,
            ),
            "ar_matrices": _freeze_float(
                self.ar_matrices,
                name="ar_matrices",
                ndim=3,
            ),
            "ma_matrices": _freeze_float(
                self.ma_matrices,
                name="ma_matrices",
                ndim=3,
            ),
        }
        if not isinstance(self.filter_result, ExactDiffuseFilterResult):
            raise TypeError("filter_result must be an ExactDiffuseFilterResult")
        if not isinstance(self.transformed_state_space, StateSpaceModel):
            raise TypeError("transformed_state_space must be a StateSpaceModel")
        if not isinstance(
            self.integrated_state_space,
            ExactSeasonalIntegratedStateSpace,
        ):
            raise TypeError(
                "integrated_state_space must be an " "ExactSeasonalIntegratedStateSpace"
            )
        covariance = _freeze_covariance(
            self.innovation_covariance,
            n_locations=self.filter_result.model.n_locations,
        )
        ordinary_order = validate_nonnegative_int(
            self.integration_order,
            name="integration_order",
        )
        seasonal_order = validate_nonnegative_int(
            self.seasonal_integration_order,
            name="seasonal_integration_order",
        )
        period = validate_nonnegative_int(
            self.seasonal_period,
            name="seasonal_period",
        )
        if period == 0:
            raise ValueError("seasonal_period must be positive")
        if len(self.parameter_names) != params.size:
            raise ValueError("parameter_names must match params")
        if len(self.optimizer_parameter_names) != raw.size:
            raise ValueError(
                "optimizer_parameter_names must match raw_optimizer_params"
            )
        if int(self.n_params) != raw.size:
            raise ValueError("n_params must match raw_optimizer_params")
        if self.integrated_state_space.model is not self.filter_result.model:
            raise ValueError("filter_result must use the integrated state space")
        if (
            self.integrated_state_space.transformed_model
            is not self.transformed_state_space
        ):
            raise ValueError(
                "integrated_state_space must reference transformed_state_space"
            )
        if self.integrated_state_space.ordinary_integration_order != ordinary_order:
            raise ValueError("ordinary integration orders must agree")
        if self.integrated_state_space.seasonal_integration_order != seasonal_order:
            raise ValueError("seasonal integration orders must agree")
        if self.integrated_state_space.seasonal_period != period:
            raise ValueError("seasonal periods must agree")
        if int(self.n_observations) != self.filter_result.n_observations:
            raise ValueError("n_observations must match filter_result")
        if int(self.n_diffuse_observations) != (
            self.filter_result.n_diffuse_observations
        ):
            raise ValueError("n_diffuse_observations must match filter_result")
        if arrays["ar_matrices"].shape[0] != len(self.ar_lags):
            raise ValueError("ar_lags must match ar_matrices")
        if arrays["ma_matrices"].shape[0] != len(self.ma_lags):
            raise ValueError("ma_lags must match ma_matrices")
        for name in (
            "intercept",
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
        if self.ar_spectral_radius < 0.0:
            raise ValueError("ar_spectral_radius must be non-negative")
        if self.ma_inverse_spectral_radius < 0.0:
            raise ValueError("ma_inverse_spectral_radius must be non-negative")
        if not 0.0 < self.stability_limit < 1.0:
            raise ValueError("stability_limit must be between zero and one")
        if not 0.0 < self.invertibility_limit < 1.0:
            raise ValueError("invertibility_limit must be between zero and one")
        for name in (
            "n_observations",
            "n_diffuse_observations",
            "n_params",
            "n_iterations",
            "n_function_evaluations",
            "n_original_rows",
            "original_missing_cells",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "raw_optimizer_params", raw)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "innovation_covariance", covariance)
        object.__setattr__(self, "integration_order", ordinary_order)
        object.__setattr__(
            self,
            "seasonal_integration_order",
            seasonal_order,
        )
        object.__setattr__(self, "seasonal_period", period)
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
        object.__setattr__(
            self,
            "ar_lags",
            tuple(int(lag) for lag in self.ar_lags),
        )
        object.__setattr__(
            self,
            "ma_lags",
            tuple(int(lag) for lag in self.ma_lags),
        )
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
        object.__setattr__(self, "optimizer_method", str(self.optimizer_method))
        object.__setattr__(self, "optimizer_message", str(self.optimizer_message))

    @property
    def order(self) -> tuple[int, int, int]:
        """Return ordinary order ``(p, d, q)``."""
        return (
            int(self.ar_parameters.shape[0]),
            self.integration_order,
            int(self.ma_parameters.shape[0]),
        )

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        """Return seasonal order ``(P, D, Q, s)``."""
        return (
            int(self.seasonal_ar_parameters.shape[0]),
            self.seasonal_integration_order,
            int(self.seasonal_ma_parameters.shape[0]),
            self.seasonal_period,
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

    @property
    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coefficient": self.params},
            index=pd.Index(self.parameter_names, name="parameter"),
        )

    @property
    def covariance(self) -> pd.DataFrame:
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
        header = [
            "pySTARMAx seasonal exact diffuse Kalman STARIMA result",
            "=" * 72,
            f"Order: {self.order}",
            f"Seasonal order: {self.seasonal_order}",
            f"Covariance: {self.covariance_type}",
            f"Original rows: {self.n_original_rows}",
            f"Observed cells: {self.n_observations}",
            f"Diffuse observations: {self.n_diffuse_observations}",
            f"Diffuse end time: {self.filter_result.diffuse_end_time}",
            f"Estimated factor parameters: {self.n_params}",
            f"Converged: {self.converged}",
            f"Function evaluations: {self.n_function_evaluations}",
            f"AR spectral radius: {self.ar_spectral_radius:.6f}",
            f"Inverse-MA spectral radius: " f"{self.ma_inverse_spectral_radius:.6f}",
            f"Log likelihood: {self.log_likelihood:.6f}",
            f"AIC: {self.aic:.6f}",
            f"BIC: {self.bic:.6f}",
            "Likelihood scope: original levels with exact diffuse initialization",
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
            header + [coefficients, "-" * 72, "Innovation covariance", covariance]
        )


@dataclass(frozen=True, slots=True)
class _DecodedSeasonalExactDiffuse:
    intercept: float
    ar_parameters: FloatArray
    seasonal_ar_parameters: FloatArray
    ma_parameters: FloatArray
    seasonal_ma_parameters: FloatArray
    ar_lags: tuple[int, ...]
    ar_matrices: FloatArray
    ma_lags: tuple[int, ...]
    ma_matrices: FloatArray
    covariance: FloatArray
    transformed: StateSpaceModel
    integrated: ExactSeasonalIntegratedStateSpace
    ar_radius: float
    ma_radius: float


class SeasonalExactDiffuseKalmanSTARIMA:
    """Estimate multiplicative seasonal STARIMA on original levels exactly."""

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
        enforce_stationarity: bool = True,
        stability_margin: float = 1e-6,
        enforce_invertibility: bool = True,
        invertibility_margin: float = 1e-6,
        diffuse_tolerance: float = 1e-10,
        max_iter: int = 500,
        tol: float = 1e-9,
    ) -> None:
        if not np.isfinite(diffuse_tolerance) or diffuse_tolerance <= 0.0:
            raise ValueError("diffuse_tolerance must be positive and finite")
        self.core_model = SeasonalKalmanSTARIMA(
            ar_order=ar_order,
            integration_order=integration_order,
            ma_order=ma_order,
            seasonal_ar_order=seasonal_ar_order,
            seasonal_integration_order=seasonal_integration_order,
            seasonal_ma_order=seasonal_ma_order,
            seasonal_period=seasonal_period,
            covariance_type=covariance_type,
            include_intercept=include_intercept,
            initialization="stationary",
            enforce_stationarity=enforce_stationarity,
            stability_margin=stability_margin,
            enforce_invertibility=enforce_invertibility,
            invertibility_margin=invertibility_margin,
            max_iter=max_iter,
            tol=tol,
        )
        self.diffuse_tolerance = float(diffuse_tolerance)
        self.result_: SeasonalExactDiffuseKalmanSTARIMAResult | None = None
        self.weights_: SpatialWeights | None = None
        self.data_: FloatArray | None = None
        self.transformed_state_space_: StateSpaceModel | None = None
        self.integrated_state_space_: ExactSeasonalIntegratedStateSpace | None = None
        self.filter_result_: ExactDiffuseFilterResult | None = None

    @property
    def ar_order(self) -> int:
        return self.core_model.ar_order

    @property
    def integration_order(self) -> int:
        return self.core_model.integration_order

    @property
    def ma_order(self) -> int:
        return self.core_model.ma_order

    @property
    def seasonal_ar_order(self) -> int:
        return self.core_model.seasonal_ar_order

    @property
    def seasonal_integration_order(self) -> int:
        return self.core_model.seasonal_integration_order

    @property
    def seasonal_ma_order(self) -> int:
        return self.core_model.seasonal_ma_order

    @property
    def seasonal_period(self) -> int:
        return self.core_model.seasonal_period

    @property
    def order(self) -> tuple[int, int, int]:
        return self.core_model.order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return self.core_model.seasonal_order

    def fit(
        self,
        data: Any,
        weights: Any,
        *,
        start_params: Any | None = None,
        start_covariance: Any | None = None,
    ) -> SeasonalExactDiffuseKalmanSTARIMAResult:
        observations, transformed_raw, _state = _combined_difference_incomplete(
            data,
            ordinary_order=self.integration_order,
            seasonal_order=self.seasonal_integration_order,
            seasonal_period=self.seasonal_period,
        )
        transformed = _validate_incomplete_data(transformed_raw)
        resolved_weights = coerce_weights(
            weights,
            n_locations=observations.shape[1],
        )
        coefficient_start, covariance_start = self.core_model._initial_values(
            transformed,
            resolved_weights,
        )
        if start_params is not None:
            coefficient_start = np.asarray(start_params, dtype=float)
            self.core_model._split_coefficients(
                coefficient_start,
                n_weights=len(resolved_weights),
            )
        coefficient_start = self.core_model._shrink_initial(
            cast(FloatArray, coefficient_start),
            resolved_weights,
        )
        if start_covariance is not None:
            covariance_start = _regularize_covariance(
                start_covariance,
                n_locations=observations.shape[1],
            )

        codec = _CovarianceCodec(
            self.core_model.covariance_type,
            observations.shape[1],
        )
        coefficient_size = int(coefficient_start.size)
        raw_start = np.concatenate([coefficient_start, codec.pack(covariance_start)])
        coefficient_names = self.core_model._coefficient_names(resolved_weights)
        optimizer_names = coefficient_names + codec.names
        bounds = [(None, None)] * coefficient_size + codec.bounds
        stability_limit = 1.0 - self.core_model.stability_margin
        invertibility_limit = 1.0 - self.core_model.invertibility_margin
        invalid_base = 1e12

        def decode(raw: FloatArray) -> _DecodedSeasonalExactDiffuse:
            expanded = self.core_model._expanded(
                cast(FloatArray, raw[:coefficient_size]),
                resolved_weights,
            )
            covariance = codec.unpack(raw[coefficient_size:])
            transformed_state = _operator_state_space(
                expanded[5],
                expanded[6],
                expanded[7],
                expanded[8],
                covariance,
                intercept=expanded[0],
            )
            integrated_state = build_exact_seasonal_integrated_state_space(
                transformed_state,
                self.integration_order,
                self.seasonal_integration_order,
                self.seasonal_period,
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
            return _DecodedSeasonalExactDiffuse(
                intercept=expanded[0],
                ar_parameters=expanded[1],
                seasonal_ar_parameters=expanded[2],
                ma_parameters=expanded[3],
                seasonal_ma_parameters=expanded[4],
                ar_lags=expanded[5],
                ar_matrices=expanded[6],
                ma_lags=expanded[7],
                ma_matrices=expanded[8],
                covariance=covariance,
                transformed=transformed_state,
                integrated=integrated_state,
                ar_radius=ar_radius,
                ma_radius=ma_radius,
            )

        def objective(raw: FloatArray) -> float:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    decoded = decode(raw)
                    squared_excess = 0.0
                    if (
                        self.core_model.enforce_stationarity
                        and decoded.ar_radius >= stability_limit
                    ):
                        squared_excess += (decoded.ar_radius - stability_limit) ** 2
                    if (
                        self.core_model.enforce_invertibility
                        and decoded.ma_radius >= invertibility_limit
                    ):
                        squared_excess += (decoded.ma_radius - invertibility_limit) ** 2
                    if squared_excess > 0.0:
                        return float(
                            invalid_base
                            + invalid_base * squared_excess
                            + 1e-8 * (raw @ raw)
                        )
                    filtered = decoded.integrated.filter(
                        observations,
                        tolerance=self.diffuse_tolerance,
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
            options={
                "maxiter": self.core_model.max_iter,
                "ftol": self.core_model.tol,
                "maxls": 50,
            },
        )
        raw_final = cast(FloatArray, np.asarray(optimized.x, dtype=float))
        decoded = decode(raw_final)
        if (
            self.core_model.enforce_stationarity
            and decoded.ar_radius >= stability_limit
        ):
            raise RuntimeError("optimizer returned a non-stationary final candidate")
        if (
            self.core_model.enforce_invertibility
            and decoded.ma_radius >= invertibility_limit
        ):
            raise RuntimeError("optimizer returned a non-invertible final candidate")
        filtered = decoded.integrated.filter(
            observations,
            tolerance=self.diffuse_tolerance,
        )
        n_params = int(raw_final.size)
        log_likelihood = filtered.log_likelihood
        aic = -2.0 * log_likelihood + 2.0 * n_params
        bic = -2.0 * log_likelihood + np.log(filtered.n_observations) * n_params
        result = SeasonalExactDiffuseKalmanSTARIMAResult(
            params=raw_final[:coefficient_size],
            parameter_names=coefficient_names,
            raw_optimizer_params=raw_final,
            optimizer_parameter_names=optimizer_names,
            intercept=decoded.intercept,
            ar_parameters=decoded.ar_parameters,
            seasonal_ar_parameters=decoded.seasonal_ar_parameters,
            ma_parameters=decoded.ma_parameters,
            seasonal_ma_parameters=decoded.seasonal_ma_parameters,
            ar_lags=decoded.ar_lags,
            ar_matrices=decoded.ar_matrices,
            ma_lags=decoded.ma_lags,
            ma_matrices=decoded.ma_matrices,
            innovation_covariance=decoded.covariance,
            covariance_type=self.core_model.covariance_type,
            integration_order=self.integration_order,
            seasonal_integration_order=self.seasonal_integration_order,
            seasonal_period=self.seasonal_period,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_observations=filtered.n_observations,
            n_diffuse_observations=filtered.n_diffuse_observations,
            n_params=n_params,
            converged=bool(optimized.success),
            n_iterations=int(getattr(optimized, "nit", 0)),
            n_function_evaluations=int(getattr(optimized, "nfev", 0)),
            optimizer_method="L-BFGS-B",
            optimizer_message=str(optimized.message),
            ar_spectral_radius=decoded.ar_radius,
            ma_inverse_spectral_radius=decoded.ma_radius,
            stability_limit=stability_limit,
            invertibility_limit=invertibility_limit,
            stationarity_enforced=self.core_model.enforce_stationarity,
            invertibility_enforced=self.core_model.enforce_invertibility,
            n_original_rows=observations.shape[0],
            original_missing_cells=int(np.count_nonzero(~np.isfinite(observations))),
            filter_result=filtered,
            transformed_state_space=decoded.transformed,
            integrated_state_space=decoded.integrated,
        )
        self.result_ = result
        self.weights_ = resolved_weights
        self.data_ = observations.copy()
        self.transformed_state_space_ = decoded.transformed
        self.integrated_state_space_ = decoded.integrated
        self.filter_result_ = filtered
        return result

    def _require_fit(
        self,
    ) -> tuple[
        SeasonalExactDiffuseKalmanSTARIMAResult,
        ExactSeasonalIntegratedStateSpace,
        ExactDiffuseFilterResult,
    ]:
        if (
            self.result_ is None
            or self.integrated_state_space_ is None
            or self.filter_result_ is None
        ):
            raise RuntimeError("fit must be called before using the fitted model")
        return self.result_, self.integrated_state_space_, self.filter_result_

    def admissibility(self) -> SeasonalKalmanAdmissibility:
        result, _integrated, _filtered = self._require_fit()
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

    def to_transformed_state_space(self) -> StateSpaceModel:
        result, _integrated, _filtered = self._require_fit()
        return result.transformed_state_space

    def to_state_space(self) -> StateSpaceModel:
        _result, integrated, _filtered = self._require_fit()
        return integrated.model

    def filter(self, data: Any | None = None) -> ExactDiffuseFilterResult:
        _result, integrated, training_filter = self._require_fit()
        if data is None:
            return training_filter
        return integrated.filter(data, tolerance=self.diffuse_tolerance)

    def predict(self, steps: int = 1) -> FloatArray:
        steps = validate_nonnegative_int(steps, name="steps")
        if steps == 0:
            raise ValueError("steps must be positive")
        _result, integrated, filtered = self._require_fit()
        state = filtered.filtered_state[-1].copy()
        forecasts = np.empty((steps, integrated.model.n_locations), dtype=float)
        for step_index in range(steps):
            state = (
                integrated.model.state_intercept + integrated.model.transition @ state
            )
            forecasts[step_index] = integrated.model.design @ state
        return cast(FloatArray, forecasts)

    def predict_differenced(self, steps: int = 1) -> FloatArray:
        steps = validate_nonnegative_int(steps, name="steps")
        if steps == 0:
            raise ValueError("steps must be positive")
        result, integrated, filtered = self._require_fit()
        offset = integrated.integration_degree * integrated.model.n_locations
        state = filtered.filtered_state[-1, offset:].copy()
        forecasts = np.empty(
            (steps, result.transformed_state_space.n_locations),
            dtype=float,
        )
        for step_index in range(steps):
            state = (
                result.transformed_state_space.state_intercept
                + result.transformed_state_space.transition @ state
            )
            forecasts[step_index] = result.transformed_state_space.design @ state
        return cast(FloatArray, forecasts)

    def fitted_original(self) -> FloatArray:
        """Return one-step original-level predictions from the exact filter."""
        return self.filter().predicted_observations


__all__ = [
    "SeasonalExactDiffuseKalmanSTARIMA",
    "SeasonalExactDiffuseKalmanSTARIMAResult",
]
