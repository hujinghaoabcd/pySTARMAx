# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Kalman maximum-likelihood estimator for stationary invertible STARMA models."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from scipy.optimize import minimize

from pystarmax._maximum_likelihood_result import KalmanSTARMAResult
from pystarmax._maximum_likelihood_utils import (
    CovarianceType,
    _CovarianceCodec,
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
from pystarmax.state_space import (
    Initialization,
    KalmanFilterResult,
    StateSpaceModel,
    build_starma_state_space,
    kalman_filter,
)
from pystarmax.weights import SpatialWeights, coerce_weights


class KalmanSTARMA:
    """Estimate a stationary invertible Gaussian STARMA model by likelihood."""

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
        enforce_invertibility: bool = True,
        invertibility_margin: float = 1e-6,
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

    def _shrink_initial_dynamics(
        self,
        ar_parameters: FloatArray,
        ma_parameters: FloatArray,
        weights: SpatialWeights,
    ) -> tuple[FloatArray, FloatArray]:
        ar_values = np.asarray(ar_parameters, dtype=float).copy()
        ma_values = np.asarray(ma_parameters, dtype=float).copy()
        if self.ar_order and self.enforce_stationarity:
            target = 1.0 - max(self.stability_margin * 10.0, 1e-4)
            for _ in range(40):
                radius = autoregressive_spectral_radius(ar_values, weights)
                if radius < target:
                    break
                factor = min(0.9, 0.9 * target / max(radius, 1e-12))
                ar_values *= factor
            if autoregressive_spectral_radius(ar_values, weights) >= target:
                raise RuntimeError("failed to construct a stationary initial AR point")
        if self.ma_order and self.enforce_invertibility:
            target = 1.0 - max(self.invertibility_margin * 10.0, 1e-4)
            for _ in range(80):
                radius = moving_average_inverse_spectral_radius(ma_values, weights)
                if radius < target:
                    break
                factor = min(0.9, 0.9 * target / max(radius, 1e-12))
                ma_values *= factor
            if moving_average_inverse_spectral_radius(ma_values, weights) >= target:
                raise RuntimeError("failed to construct an invertible initial MA point")
        return cast(FloatArray, ar_values), cast(FloatArray, ma_values)

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
                    raise RuntimeError(
                        "conditional initializer did not return a result"
                    )
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
        ar_parameters, ma_parameters = self._shrink_initial_dynamics(
            ar_parameters,
            ma_parameters,
            weights,
        )
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
        """Estimate parameters by minimizing the negative Kalman likelihood."""
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
        invertibility_limit = 1.0 - self.invertibility_margin
        invalid_base = 1e12

        def decode(
            raw: FloatArray,
        ) -> tuple[
            float,
            FloatArray,
            FloatArray,
            FloatArray,
            StateSpaceModel,
            float,
            float,
        ]:
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
                state_space,
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
                        state_space,
                        spectral_radius,
                        ma_inverse_radius,
                    ) = decode(raw)
                    squared_excess = 0.0
                    if self.enforce_stationarity and spectral_radius >= stability_limit:
                        squared_excess += (spectral_radius - stability_limit) ** 2
                    if (
                        self.enforce_invertibility
                        and ma_inverse_radius >= invertibility_limit
                    ):
                        squared_excess += (
                            ma_inverse_radius - invertibility_limit
                        ) ** 2
                    if squared_excess > 0.0:
                        return float(
                            invalid_base
                            + invalid_base * squared_excess
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
            ma_inverse_radius,
        ) = decode(raw_final)
        if self.enforce_stationarity and spectral_radius >= stability_limit:
            raise RuntimeError("optimizer returned a non-stationary final candidate")
        if self.enforce_invertibility and ma_inverse_radius >= invertibility_limit:
            raise RuntimeError("optimizer returned a non-invertible final candidate")
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
            ma_inverse_spectral_radius=ma_inverse_radius,
            stability_limit=stability_limit,
            invertibility_limit=invertibility_limit,
            stationarity_enforced=self.enforce_stationarity,
            invertibility_enforced=self.enforce_invertibility,
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

    def admissibility(self) -> STARMAAdmissibility:
        """Return fitted AR stationarity and MA invertibility diagnostics."""
        if self.result_ is None or self.weights_ is None:
            raise RuntimeError("fit must be called before admissibility diagnostics")
        return starma_admissibility(
            self.result_.ar_parameters,
            self.result_.ma_parameters,
            self.weights_,
            stability_margin=self.stability_margin,
            invertibility_margin=self.invertibility_margin,
        )

    def to_state_space(self) -> StateSpaceModel:
        """Return the fitted state-space representation."""
        state_space, _filter_result = self._fitted_components()
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
        """Generate recursive conditional means from the final filtered state."""
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
