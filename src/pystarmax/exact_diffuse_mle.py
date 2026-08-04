# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Optimizer-facing exact diffuse maximum likelihood for ordinary STARIMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pystarmax._maximum_likelihood_result import KalmanSTARMAResult
from pystarmax._maximum_likelihood_utils import (
    CovarianceType,
    _CovarianceCodec,
    _freeze_covariance,
    _freeze_float,
    _regularize_covariance,
    _validate_incomplete_data,
)
from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.admissibility import (
    STARMAAdmissibility,
    autoregressive_spectral_radius,
    moving_average_inverse_spectral_radius,
    starma_admissibility,
)
from pystarmax.exact_diffuse import ExactDiffuseFilterResult
from pystarmax.exact_integrated import (
    ExactIntegratedStateSpace,
    build_exact_integrated_state_space,
)
from pystarmax.integrated_maximum_likelihood import _difference_incomplete
from pystarmax.maximum_likelihood import KalmanSTARMA
from pystarmax.state_space import StateSpaceModel, build_starma_state_space
from pystarmax.weights import SpatialWeights, coerce_weights


@dataclass(frozen=True, slots=True)
class ExactDiffuseKalmanSTARIMAResult:
    """Immutable optimizer result for exact diffuse ordinary STARIMA."""

    params: FloatArray
    parameter_names: tuple[str, ...]
    raw_optimizer_params: FloatArray
    optimizer_parameter_names: tuple[str, ...]
    intercept: float
    ar_parameters: FloatArray
    ma_parameters: FloatArray
    innovation_covariance: FloatArray
    covariance_type: str
    integration_order: int
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
    spectral_radius: float
    ma_inverse_spectral_radius: float
    stability_limit: float
    invertibility_limit: float
    stationarity_enforced: bool
    invertibility_enforced: bool
    filter_result: ExactDiffuseFilterResult
    transformed_state_space: StateSpaceModel
    integrated_state_space: ExactIntegratedStateSpace

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
        if not isinstance(self.filter_result, ExactDiffuseFilterResult):
            raise TypeError("filter_result must be an ExactDiffuseFilterResult")
        if not isinstance(self.transformed_state_space, StateSpaceModel):
            raise TypeError("transformed_state_space must be a StateSpaceModel")
        if not isinstance(self.integrated_state_space, ExactIntegratedStateSpace):
            raise TypeError(
                "integrated_state_space must be an ExactIntegratedStateSpace"
            )
        covariance = _freeze_covariance(
            self.innovation_covariance,
            n_locations=self.filter_result.model.n_locations,
        )
        integration_order = validate_nonnegative_int(
            self.integration_order,
            name="integration_order",
        )
        if self.integrated_state_space.integration_order != integration_order:
            raise ValueError(
                "integrated_state_space order must match integration_order"
            )
        if self.integrated_state_space.model is not self.filter_result.model:
            raise ValueError("filter_result must use the integrated state-space model")
        if (
            self.integrated_state_space.transformed_model
            is not self.transformed_state_space
        ):
            raise ValueError(
                "integrated_state_space must reference transformed_state_space"
            )
        if len(self.parameter_names) != params.size:
            raise ValueError("parameter_names must match params")
        if len(self.optimizer_parameter_names) != raw.size:
            raise ValueError(
                "optimizer_parameter_names must match raw_optimizer_params"
            )
        if int(self.n_params) != raw.size:
            raise ValueError("n_params must match raw_optimizer_params")
        if int(self.n_observations) != self.filter_result.n_observations:
            raise ValueError("n_observations must match filter_result")
        if int(self.n_diffuse_observations) != (
            self.filter_result.n_diffuse_observations
        ):
            raise ValueError("n_diffuse_observations must match filter_result")
        for name, value in (
            ("intercept", self.intercept),
            ("log_likelihood", self.log_likelihood),
            ("aic", self.aic),
            ("bic", self.bic),
            ("spectral_radius", self.spectral_radius),
            ("ma_inverse_spectral_radius", self.ma_inverse_spectral_radius),
            ("stability_limit", self.stability_limit),
            ("invertibility_limit", self.invertibility_limit),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        spectral_radius = float(self.spectral_radius)
        ma_radius = float(self.ma_inverse_spectral_radius)
        stability_limit = float(self.stability_limit)
        invertibility_limit = float(self.invertibility_limit)
        if spectral_radius < 0.0 or ma_radius < 0.0:
            raise ValueError("spectral radii must be non-negative")
        if not 0.0 < stability_limit < 1.0:
            raise ValueError("stability_limit must be between zero and one")
        if not 0.0 < invertibility_limit < 1.0:
            raise ValueError("invertibility_limit must be between zero and one")
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "raw_optimizer_params", raw)
        object.__setattr__(self, "ar_parameters", ar_parameters)
        object.__setattr__(self, "ma_parameters", ma_parameters)
        object.__setattr__(self, "innovation_covariance", covariance)
        object.__setattr__(self, "integration_order", integration_order)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "log_likelihood", float(self.log_likelihood))
        object.__setattr__(self, "aic", float(self.aic))
        object.__setattr__(self, "bic", float(self.bic))
        object.__setattr__(self, "n_observations", int(self.n_observations))
        object.__setattr__(
            self,
            "n_diffuse_observations",
            int(self.n_diffuse_observations),
        )
        object.__setattr__(self, "n_params", int(self.n_params))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "n_iterations", int(self.n_iterations))
        object.__setattr__(
            self,
            "n_function_evaluations",
            int(self.n_function_evaluations),
        )
        object.__setattr__(self, "spectral_radius", spectral_radius)
        object.__setattr__(self, "ma_inverse_spectral_radius", ma_radius)
        object.__setattr__(self, "stability_limit", stability_limit)
        object.__setattr__(self, "invertibility_limit", invertibility_limit)
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

    @property
    def order(self) -> tuple[int, int, int]:
        """Return model order ``(p, d, q)``."""
        return (
            int(self.ar_parameters.shape[0]),
            self.integration_order,
            int(self.ma_parameters.shape[0]),
        )

    @property
    def stationary(self) -> bool:
        """Whether the transformed AR companion is below its limit."""
        return self.spectral_radius < self.stability_limit

    @property
    def invertible(self) -> bool:
        """Whether the transformed inverse-MA companion is below its limit."""
        return self.ma_inverse_spectral_radius < self.invertibility_limit

    @property
    def admissible(self) -> bool:
        """Whether transformed AR and MA dynamics are jointly admissible."""
        return self.stationary and self.invertible

    @property
    def coefficients(self) -> pd.DataFrame:
        """Dynamic parameter table."""
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
        """Return a compact exact diffuse maximum-likelihood summary."""
        header = [
            "pySTARMAx exact diffuse Kalman STARIMA result",
            "=" * 72,
            f"Order: {self.order}",
            f"Optimizer: {self.optimizer_method}",
            f"Covariance: {self.covariance_type}",
            f"Observed cells: {self.n_observations}",
            f"Diffuse observations: {self.n_diffuse_observations}",
            f"Diffuse end time: {self.filter_result.diffuse_end_time}",
            f"Estimated parameters: {self.n_params}",
            f"Converged: {self.converged} ({self.n_iterations} iteration(s))",
            f"Function evaluations: {self.n_function_evaluations}",
            f"AR spectral radius: {self.spectral_radius:.6f}",
            f"AR limit: {self.stability_limit:.6f}",
            f"Stationary transformed state: {self.stationary}",
            f"Inverse-MA spectral radius: {self.ma_inverse_spectral_radius:.6f}",
            f"MA limit: {self.invertibility_limit:.6f}",
            f"Invertible transformed state: {self.invertible}",
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


class ExactDiffuseKalmanSTARIMA:
    """Estimate ordinary STARIMA by exact diffuse original-level likelihood."""

    def __init__(
        self,
        ar_order: int = 1,
        integration_order: int = 1,
        ma_order: int = 0,
        *,
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
        self.integration_order = validate_nonnegative_int(
            integration_order,
            name="integration_order",
        )
        if not np.isfinite(diffuse_tolerance) or diffuse_tolerance <= 0.0:
            raise ValueError("diffuse_tolerance must be positive and finite")
        self.core_model = KalmanSTARMA(
            ar_order=ar_order,
            ma_order=ma_order,
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
        self.result_: ExactDiffuseKalmanSTARIMAResult | None = None
        self.weights_: SpatialWeights | None = None
        self.data_: FloatArray | None = None
        self.transformed_state_space_: StateSpaceModel | None = None
        self.integrated_state_space_: ExactIntegratedStateSpace | None = None
        self.filter_result_: ExactDiffuseFilterResult | None = None

    @property
    def ar_order(self) -> int:
        """Transformed autoregressive order ``p``."""
        return self.core_model.ar_order

    @property
    def ma_order(self) -> int:
        """Transformed moving-average order ``q``."""
        return self.core_model.ma_order

    @property
    def order(self) -> tuple[int, int, int]:
        """Return model order ``(p, d, q)``."""
        return self.ar_order, self.integration_order, self.ma_order

    def fit(
        self,
        data: Any,
        weights: Any,
        *,
        start_params: Any | None = None,
        start_covariance: Any | None = None,
    ) -> ExactDiffuseKalmanSTARIMAResult:
        """Maximize the original-level exact diffuse Gaussian likelihood."""
        observations = _validate_incomplete_data(data)
        if self.integration_order >= observations.shape[0]:
            raise ValueError(
                "integration_order must be smaller than the number of time rows"
            )
        resolved_weights = coerce_weights(
            weights,
            n_locations=observations.shape[1],
        )
        _levels, differenced, _state = _difference_incomplete(
            observations,
            order=self.integration_order,
        )
        n_weights = len(resolved_weights)
        coefficient_start, covariance_start = self.core_model._initial_values(
            differenced,
            resolved_weights,
        )
        if start_params is not None:
            coefficient_start = np.asarray(start_params, dtype=float)
            self.core_model._split_coefficients(
                coefficient_start,
                n_weights=n_weights,
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
        covariance_parameters = codec.pack(covariance_start)
        raw_start = np.concatenate([coefficient_start, covariance_parameters])
        coefficient_names = self.core_model._coefficient_names(resolved_weights)
        optimizer_names = coefficient_names + codec.names
        coefficient_size = coefficient_start.size
        bounds = [(None, None)] * coefficient_size + codec.bounds
        stability_limit = 1.0 - self.core_model.stability_margin
        invertibility_limit = 1.0 - self.core_model.invertibility_margin
        invalid_base = 1e12

        def decode(
            raw: FloatArray,
        ) -> tuple[
            float,
            FloatArray,
            FloatArray,
            FloatArray,
            StateSpaceModel,
            ExactIntegratedStateSpace,
            float,
            float,
        ]:
            intercept, ar_parameters, ma_parameters = (
                self.core_model._split_coefficients(
                    raw[:coefficient_size],
                    n_weights=n_weights,
                )
            )
            covariance = codec.unpack(raw[coefficient_size:])
            transformed = build_starma_state_space(
                ar_parameters,
                ma_parameters,
                resolved_weights,
                covariance,
                intercept=intercept,
            )
            integrated = build_exact_integrated_state_space(
                transformed,
                self.integration_order,
            )
            spectral_radius = autoregressive_spectral_radius(
                ar_parameters,
                resolved_weights,
            )
            ma_inverse_radius = moving_average_inverse_spectral_radius(
                ma_parameters,
                resolved_weights,
            )
            return (
                intercept,
                ar_parameters,
                ma_parameters,
                covariance,
                transformed,
                integrated,
                spectral_radius,
                ma_inverse_radius,
            )

        def objective(raw: FloatArray) -> float:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    (
                        _intercept,
                        _ar_parameters,
                        _ma_parameters,
                        _covariance,
                        _transformed,
                        integrated,
                        spectral_radius,
                        ma_inverse_radius,
                    ) = decode(raw)
                    squared_excess = 0.0
                    if (
                        self.core_model.enforce_stationarity
                        and spectral_radius >= stability_limit
                    ):
                        squared_excess += (spectral_radius - stability_limit) ** 2
                    if (
                        self.core_model.enforce_invertibility
                        and ma_inverse_radius >= invertibility_limit
                    ):
                        squared_excess += (ma_inverse_radius - invertibility_limit) ** 2
                    if squared_excess > 0.0:
                        return float(
                            invalid_base
                            + invalid_base * squared_excess
                            + 1e-8 * (raw @ raw)
                        )
                    filtered = integrated.filter(
                        observations,
                        tolerance=self.diffuse_tolerance,
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
                "maxiter": self.core_model.max_iter,
                "ftol": self.core_model.tol,
                "maxls": 50,
            },
        )
        raw_final = np.asarray(optimized.x, dtype=float)
        (
            intercept,
            ar_parameters,
            ma_parameters,
            covariance,
            transformed,
            integrated,
            spectral_radius,
            ma_inverse_radius,
        ) = decode(raw_final)
        if self.core_model.enforce_stationarity and spectral_radius >= stability_limit:
            raise RuntimeError("optimizer returned a non-stationary final candidate")
        if (
            self.core_model.enforce_invertibility
            and ma_inverse_radius >= invertibility_limit
        ):
            raise RuntimeError("optimizer returned a non-invertible final candidate")
        filtered = integrated.filter(
            observations,
            tolerance=self.diffuse_tolerance,
        )
        coefficients = raw_final[:coefficient_size]
        n_params = int(raw_final.size)
        n_observations = filtered.n_observations
        log_likelihood = filtered.log_likelihood
        aic = -2.0 * log_likelihood + 2.0 * n_params
        bic = -2.0 * log_likelihood + np.log(n_observations) * n_params
        result = ExactDiffuseKalmanSTARIMAResult(
            params=coefficients,
            parameter_names=coefficient_names,
            raw_optimizer_params=raw_final,
            optimizer_parameter_names=optimizer_names,
            intercept=intercept,
            ar_parameters=ar_parameters,
            ma_parameters=ma_parameters,
            innovation_covariance=covariance,
            covariance_type=self.core_model.covariance_type,
            integration_order=self.integration_order,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_observations=n_observations,
            n_diffuse_observations=filtered.n_diffuse_observations,
            n_params=n_params,
            converged=bool(optimized.success),
            n_iterations=int(getattr(optimized, "nit", 0)),
            n_function_evaluations=int(getattr(optimized, "nfev", 0)),
            optimizer_method="L-BFGS-B",
            optimizer_message=str(optimized.message),
            spectral_radius=spectral_radius,
            ma_inverse_spectral_radius=ma_inverse_radius,
            stability_limit=stability_limit,
            invertibility_limit=invertibility_limit,
            stationarity_enforced=self.core_model.enforce_stationarity,
            invertibility_enforced=self.core_model.enforce_invertibility,
            filter_result=filtered,
            transformed_state_space=transformed,
            integrated_state_space=integrated,
        )
        self.result_ = result
        self.weights_ = resolved_weights
        self.data_ = observations.copy()
        self.transformed_state_space_ = transformed
        self.integrated_state_space_ = integrated
        self.filter_result_ = filtered
        return result

    def _require_fit(
        self,
    ) -> tuple[
        ExactDiffuseKalmanSTARIMAResult,
        ExactIntegratedStateSpace,
        ExactDiffuseFilterResult,
    ]:
        if (
            self.result_ is None
            or self.integrated_state_space_ is None
            or self.filter_result_ is None
        ):
            raise RuntimeError("fit must be called before using the fitted model")
        return self.result_, self.integrated_state_space_, self.filter_result_

    def admissibility(self) -> STARMAAdmissibility:
        """Return fitted transformed AR and inverse-MA diagnostics."""
        result, _integrated, _filtered = self._require_fit()
        if self.weights_ is None:
            raise RuntimeError("fit must be called before admissibility diagnostics")
        return starma_admissibility(
            result.ar_parameters,
            result.ma_parameters,
            self.weights_,
            stability_margin=self.core_model.stability_margin,
            invertibility_margin=self.core_model.invertibility_margin,
        )

    def to_transformed_state_space(self) -> StateSpaceModel:
        """Return the fitted stationary state space for ``Delta^d y_t``."""
        result, _integrated, _filtered = self._require_fit()
        return result.transformed_state_space

    def to_state_space(self) -> StateSpaceModel:
        """Return the fitted exact diffuse original-level state space."""
        _result, integrated, _filtered = self._require_fit()
        return integrated.model

    def filter(self, data: Any | None = None) -> ExactDiffuseFilterResult:
        """Filter training or new original-level observations exactly."""
        _result, integrated, training_filter = self._require_fit()
        if data is None:
            return training_filter
        return integrated.filter(data, tolerance=self.diffuse_tolerance)

    def predict(self, steps: int = 1) -> FloatArray:
        """Return recursive original-level means from the final filtered state."""
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
        """Return recursive means on the highest ordinary-difference scale."""
        steps = validate_nonnegative_int(steps, name="steps")
        if steps == 0:
            raise ValueError("steps must be positive")
        result, integrated, filtered = self._require_fit()
        offset = self.integration_order * integrated.model.n_locations
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


__all__ = [
    "ExactDiffuseKalmanSTARIMA",
    "ExactDiffuseKalmanSTARIMAResult",
]
