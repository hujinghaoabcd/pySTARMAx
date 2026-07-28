# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""STAR and STARMA estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)
from pystarmax.results import STARMAResult
from pystarmax.weights import SpatialWeights, coerce_weights


def _spatial_lags(vector: FloatArray, weights: SpatialWeights) -> list[FloatArray]:
    return [matrix @ vector for matrix in weights]


class STARMA:
    """Estimate a classical STARMA model by conditional least squares.

    Parameters
    ----------
    ar_order:
        Maximum temporal autoregressive order.
    ma_order:
        Maximum temporal moving-average order.
    include_intercept:
        Include one common intercept across locations.
    max_iter:
        Maximum iterations when moving-average terms are present.
    tol:
        Relative coefficient convergence tolerance.
    ridge:
        Optional non-negative ridge value added to the normal equations.

    Notes
    -----
    All combinations of temporal lag and supplied spatial-lag matrix are fitted.
    The sign convention is ``+ theta * W * epsilon``.
    """

    def __init__(
        self,
        ar_order: int = 1,
        ma_order: int = 0,
        *,
        include_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-8,
        ridge: float = 0.0,
    ) -> None:
        self.ar_order = validate_nonnegative_int(ar_order, name="ar_order")
        self.ma_order = validate_nonnegative_int(ma_order, name="ma_order")
        if self.ar_order == 0 and self.ma_order == 0:
            raise ValueError("ar_order and ma_order cannot both be zero")
        self.include_intercept = bool(include_intercept)
        self.max_iter = validate_nonnegative_int(max_iter, name="max_iter")
        if self.max_iter == 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.result_: STARMAResult | None = None
        self.weights_: SpatialWeights | None = None
        self.data_: FloatArray | None = None
        self._internal_residuals: FloatArray | None = None

    @property
    def max_lag(self) -> int:
        """Maximum temporal lag used by the model."""
        return max(self.ar_order, self.ma_order)

    def _parameter_names(self, weights: SpatialWeights) -> tuple[str, ...]:
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

    def _design(
        self,
        data: FloatArray,
        weights: SpatialWeights,
        residuals: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        rows: list[FloatArray] = []
        targets: list[FloatArray] = []
        for time_index in range(self.max_lag, data.shape[0]):
            columns: list[FloatArray] = []
            if self.include_intercept:
                columns.append(np.ones(data.shape[1], dtype=float))
            for temporal_lag in range(1, self.ar_order + 1):
                columns.extend(_spatial_lags(data[time_index - temporal_lag], weights))
            for temporal_lag in range(1, self.ma_order + 1):
                columns.extend(
                    _spatial_lags(residuals[time_index - temporal_lag], weights)
                )
            rows.append(np.column_stack(columns))
            targets.append(data[time_index])
        return np.vstack(rows), np.concatenate(targets)

    def _solve(self, design: FloatArray, target: FloatArray) -> FloatArray:
        if self.ridge == 0:
            params, *_ = np.linalg.lstsq(design, target, rcond=None)
            return np.asarray(params, dtype=float)
        gram = design.T @ design
        penalty = np.eye(gram.shape[0], dtype=float) * self.ridge
        if self.include_intercept:
            penalty[0, 0] = 0.0
        return np.linalg.solve(gram + penalty, design.T @ target)

    def _initial_residuals(
        self, data: FloatArray, weights: SpatialWeights
    ) -> FloatArray:
        residuals = np.zeros_like(data)
        if self.ar_order == 0:
            center = data.mean() if self.include_intercept else 0.0
            residuals[:] = data - center
            return residuals
        design, target = self._design_ar_only(data, weights)
        params = self._solve(design, target)
        fitted = design @ params
        residuals[self.max_lag :] = (target - fitted).reshape(
            data.shape[0] - self.max_lag, data.shape[1]
        )
        if self.max_lag:
            residuals[: self.max_lag] = data[: self.max_lag] - data.mean(axis=0)
        return residuals

    def _design_ar_only(
        self, data: FloatArray, weights: SpatialWeights
    ) -> tuple[FloatArray, FloatArray]:
        rows: list[FloatArray] = []
        targets: list[FloatArray] = []
        for time_index in range(self.max_lag, data.shape[0]):
            columns: list[FloatArray] = []
            if self.include_intercept:
                columns.append(np.ones(data.shape[1], dtype=float))
            for temporal_lag in range(1, self.ar_order + 1):
                columns.extend(_spatial_lags(data[time_index - temporal_lag], weights))
            rows.append(np.column_stack(columns))
            targets.append(data[time_index])
        return np.vstack(rows), np.concatenate(targets)

    def fit(self, data: Any, weights: Any) -> STARMAResult:
        """Fit the model and return an immutable result object."""
        observations = validate_time_space(data)
        resolved_weights = coerce_weights(weights, n_locations=observations.shape[1])
        if observations.shape[0] <= self.max_lag + 1:
            raise ValueError(
                "data contain too few time observations for the model order"
            )

        residuals = self._initial_residuals(observations, resolved_weights)
        previous: FloatArray | None = None
        converged = self.ma_order == 0
        iterations = 1

        iteration_limit = 1 if self.ma_order == 0 else self.max_iter
        for iteration in range(1, iteration_limit + 1):
            design, target = self._design(observations, resolved_weights, residuals)
            params = self._solve(design, target)
            fitted_vector = design @ params
            updated = np.zeros_like(observations)
            updated[: self.max_lag] = residuals[: self.max_lag]
            updated[self.max_lag :] = (target - fitted_vector).reshape(
                observations.shape[0] - self.max_lag, observations.shape[1]
            )
            iterations = iteration
            if previous is not None:
                denominator = max(1.0, float(np.linalg.norm(previous)))
                relative_change = float(np.linalg.norm(params - previous)) / denominator
                if relative_change <= self.tol:
                    residuals = updated
                    converged = True
                    break
            previous = params.copy()
            residuals = updated

        design, target = self._design(observations, resolved_weights, residuals)
        params = self._solve(design, target)
        fitted_vector = design @ params
        residual_vector = target - fitted_vector
        n_observations = int(target.size)
        n_params = int(params.size)
        degrees_of_freedom = max(1, n_observations - n_params)
        sum_squared = float(residual_vector @ residual_vector)
        sigma2 = sum_squared / degrees_of_freedom
        gram = design.T @ design
        if self.ridge:
            penalty = np.eye(gram.shape[0], dtype=float) * self.ridge
            if self.include_intercept:
                penalty[0, 0] = 0.0
            gram = gram + penalty
        covariance = sigma2 * np.linalg.pinv(gram)
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = params / standard_errors
        p_values = 2.0 * stats.t.sf(np.abs(t_values), df=degrees_of_freedom)

        fitted_values = np.full_like(observations, np.nan, dtype=float)
        fitted_values[self.max_lag :] = fitted_vector.reshape(
            observations.shape[0] - self.max_lag, observations.shape[1]
        )
        public_residuals = np.full_like(observations, np.nan, dtype=float)
        public_residuals[self.max_lag :] = residual_vector.reshape(
            observations.shape[0] - self.max_lag, observations.shape[1]
        )
        residual_matrix = public_residuals[self.max_lag :]
        innovation_covariance = np.atleast_2d(
            np.cov(residual_matrix, rowvar=False, ddof=1)
        ).astype(float)
        sigma2_ml = max(sum_squared / n_observations, np.finfo(float).tiny)
        log_likelihood = -0.5 * n_observations * (np.log(2.0 * np.pi * sigma2_ml) + 1.0)
        aic = -2.0 * log_likelihood + 2.0 * n_params
        bic = -2.0 * log_likelihood + np.log(n_observations) * n_params
        method = (
            "ordinary least squares"
            if self.ma_order == 0
            else "iterative conditional least squares"
        )

        result = STARMAResult(
            params=np.asarray(params, dtype=float),
            parameter_names=self._parameter_names(resolved_weights),
            standard_errors=np.asarray(standard_errors, dtype=float),
            t_values=np.asarray(t_values, dtype=float),
            p_values=np.asarray(p_values, dtype=float),
            fitted_values=fitted_values,
            residuals=public_residuals,
            innovation_covariance=innovation_covariance,
            sigma2=float(sigma2),
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            n_observations=n_observations,
            degrees_of_freedom=degrees_of_freedom,
            converged=converged,
            n_iterations=iterations,
            method=method,
            conditional_lag=self.max_lag,
        )
        final_internal_residuals = residuals.copy()
        final_internal_residuals[self.max_lag :] = residual_vector.reshape(
            observations.shape[0] - self.max_lag, observations.shape[1]
        )
        self.result_ = result
        self.weights_ = resolved_weights
        self.data_ = observations.copy()
        self._internal_residuals = final_internal_residuals
        return result

    def predict(self, steps: int = 1) -> FloatArray:
        """Generate recursive forecasts after fitting.

        Future innovations are set to zero while available historical fitted
        innovations are retained for the first forecast steps.
        """
        steps = validate_nonnegative_int(steps, name="steps")
        if steps == 0:
            raise ValueError("steps must be positive")
        if (
            self.result_ is None
            or self.weights_ is None
            or self.data_ is None
            or self._internal_residuals is None
        ):
            raise RuntimeError("fit must be called before predict")
        history = [row.copy() for row in self.data_]
        innovation_history = [row.copy() for row in self._internal_residuals]
        params = self.result_.params
        cursor = 0
        intercept = 0.0
        if self.include_intercept:
            intercept = float(params[0])
            cursor = 1
        ar_end = cursor + self.ar_order * len(self.weights_)
        ar_params = params[cursor:ar_end].reshape(self.ar_order, len(self.weights_))
        cursor += self.ar_order * len(self.weights_)
        ma_params = params[cursor:].reshape(self.ma_order, len(self.weights_))
        forecasts: list[FloatArray] = []
        for _ in range(steps):
            value = np.full(self.weights_.n_locations, intercept, dtype=float)
            for temporal_lag in range(1, self.ar_order + 1):
                for spatial_lag, matrix in enumerate(self.weights_):
                    value += ar_params[temporal_lag - 1, spatial_lag] * (
                        matrix @ history[-temporal_lag]
                    )
            for temporal_lag in range(1, self.ma_order + 1):
                for spatial_lag, matrix in enumerate(self.weights_):
                    value += ma_params[temporal_lag - 1, spatial_lag] * (
                        matrix @ innovation_history[-temporal_lag]
                    )
            forecasts.append(value)
            history.append(value)
            innovation_history.append(np.zeros_like(value))
        return np.vstack(forecasts)


class STAR(STARMA):
    """Convenience class for a pure space-time autoregressive model."""

    def __init__(
        self,
        ar_order: int = 1,
        *,
        include_intercept: bool = True,
        ridge: float = 0.0,
    ) -> None:
        super().__init__(
            ar_order=ar_order,
            ma_order=0,
            include_intercept=include_intercept,
            ridge=ridge,
        )
