# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Ordinary integrated STARMA estimation by differenced Kalman likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from pystarmax._maximum_likelihood_result import KalmanSTARMAResult
from pystarmax._maximum_likelihood_utils import CovarianceType
from pystarmax._validation import FloatArray, validate_nonnegative_int
from pystarmax.admissibility import STARMAAdmissibility
from pystarmax.differencing import DifferencingState, differencing_coefficients
from pystarmax.innovation_smoothing import InnovationDisturbanceResult
from pystarmax.likelihood_inference import LikelihoodInferenceResult
from pystarmax.maximum_likelihood import KalmanSTARMA
from pystarmax.smoothing import KalmanSmootherResult
from pystarmax.state_space import KalmanFilterResult, StateSpaceModel


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


def _difference_incomplete(
    data: Any,
    *,
    order: int,
    name: str = "data",
) -> tuple[FloatArray, FloatArray, DifferencingState | None]:
    observations = _incomplete_matrix(data, name=name)
    order = validate_nonnegative_int(order, name="integration_order")
    if order >= observations.shape[0]:
        raise ValueError(
            "integration_order must be smaller than the number of time rows"
        )

    levels: list[FloatArray] = [observations.copy()]
    for _ in range(order):
        levels.append(cast(FloatArray, np.diff(levels[-1], axis=0)))
    differenced = cast(
        FloatArray,
        np.ascontiguousarray(levels[-1], dtype=float),
    )

    if order == 0:
        state: DifferencingState | None = DifferencingState(
            order=0,
            n_locations=observations.shape[1],
            anchors=(),
        )
    else:
        anchors = tuple(level[-1] for level in levels[:-1])
        state = None
        if all(np.all(np.isfinite(anchor)) for anchor in anchors):
            state = DifferencingState(
                order=order,
                n_locations=observations.shape[1],
                anchors=anchors,
            )
    return observations, differenced, state


def _restore_incomplete_fitted(
    observations: FloatArray,
    transformed_fitted: FloatArray,
    *,
    order: int,
) -> FloatArray:
    coefficients = differencing_coefficients(ordinary_order=order)
    offset = int(coefficients.size - 1)
    expected_rows = observations.shape[0] - offset
    if transformed_fitted.shape != (expected_rows, observations.shape[1]):
        raise ValueError("transformed fitted values have an incompatible shape")

    restored = np.full(observations.shape, np.nan, dtype=float)
    if order == 0:
        restored[:] = transformed_fitted
        return cast(FloatArray, restored)

    for transformed_index, row in enumerate(transformed_fitted):
        time_index = transformed_index + offset
        for location_index, value in enumerate(row):
            if not np.isfinite(value):
                continue
            history = observations[
                time_index - offset : time_index,
                location_index,
            ]
            if not np.all(np.isfinite(history)):
                continue
            original_value = float(value)
            for lag, coefficient in enumerate(coefficients[1:], start=1):
                original_value -= float(coefficient) * float(
                    observations[time_index - lag, location_index]
                )
            restored[time_index, location_index] = original_value
    return cast(FloatArray, np.ascontiguousarray(restored, dtype=float))


@dataclass(frozen=True, slots=True)
class KalmanSTARIMAResult:
    """Immutable metadata and stationary result for `KalmanSTARIMA`."""

    core_result: KalmanSTARMAResult
    integration_order: int
    n_original_rows: int
    n_differenced_rows: int
    original_missing_cells: int
    differenced_missing_cells: int
    original_scale_forecast_available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.core_result, KalmanSTARMAResult):
            raise TypeError("core_result must be a KalmanSTARMAResult")
        integration_order = validate_nonnegative_int(
            self.integration_order,
            name="integration_order",
        )
        n_original_rows = int(self.n_original_rows)
        n_differenced_rows = int(self.n_differenced_rows)
        if n_original_rows < 1:
            raise ValueError("n_original_rows must be positive")
        if n_differenced_rows != n_original_rows - integration_order:
            raise ValueError(
                "n_differenced_rows must equal n_original_rows - integration_order"
            )
        for name, value in (
            ("original_missing_cells", self.original_missing_cells),
            ("differenced_missing_cells", self.differenced_missing_cells),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "integration_order", integration_order)
        object.__setattr__(self, "n_original_rows", n_original_rows)
        object.__setattr__(self, "n_differenced_rows", n_differenced_rows)
        object.__setattr__(
            self,
            "original_missing_cells",
            int(self.original_missing_cells),
        )
        object.__setattr__(
            self,
            "differenced_missing_cells",
            int(self.differenced_missing_cells),
        )
        object.__setattr__(
            self,
            "original_scale_forecast_available",
            bool(self.original_scale_forecast_available),
        )

    @property
    def order(self) -> tuple[int, int, int]:
        """Model order `(p, d, q)`."""
        return (
            int(self.core_result.ar_parameters.shape[0]),
            self.integration_order,
            int(self.core_result.ma_parameters.shape[0]),
        )

    @property
    def params(self) -> FloatArray:
        """Dynamic coefficients on the differenced scale."""
        return self.core_result.params

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Dynamic parameter names on the differenced scale."""
        return self.core_result.parameter_names

    @property
    def coefficients(self) -> pd.DataFrame:
        """Dynamic coefficient table from the stationary core."""
        return self.core_result.coefficients

    @property
    def covariance(self) -> pd.DataFrame:
        """Innovation covariance table from the stationary core."""
        return self.core_result.covariance

    @property
    def filter_result(self) -> KalmanFilterResult:
        """Training filter result on the differenced scale."""
        return self.core_result.filter_result

    @property
    def log_likelihood(self) -> float:
        """Conditional differenced-scale Gaussian log likelihood."""
        return self.core_result.log_likelihood

    @property
    def aic(self) -> float:
        """AIC for the conditional differenced likelihood."""
        return self.core_result.aic

    @property
    def bic(self) -> float:
        """BIC for the conditional differenced likelihood."""
        return self.core_result.bic

    @property
    def converged(self) -> bool:
        """Whether the stationary optimizer reported convergence."""
        return self.core_result.converged

    def summary(self) -> str:
        """Return an integrated-model header followed by the core summary."""
        header = [
            "pySTARMAx conditional integrated Kalman result",
            "=" * 72,
            f"Order: {self.order}",
            f"Original rows: {self.n_original_rows}",
            f"Differenced rows: {self.n_differenced_rows}",
            f"Conditioned initial rows: {self.integration_order}",
            f"Original missing cells: {self.original_missing_cells}",
            f"Differenced missing cells: {self.differenced_missing_cells}",
            "Original-scale forecast available: "
            f"{self.original_scale_forecast_available}",
            "Likelihood scope: ordinary differences conditional on initial history",
            "-" * 72,
        ]
        return "\n".join(header + [self.core_result.summary()])


class KalmanSTARIMA:
    """Estimate `STARIMA(p, d, q)` by a conditional differenced likelihood.

    Ordinary differences are formed before fitting `KalmanSTARMA`. The first
    `d` original-scale rows enter only through the differencing transformation;
    they are conditioned on rather than assigned an exact diffuse state-space
    likelihood. Internal missing cells propagate through the finite-difference
    operator and are handled by the stationary Kalman filter.
    """

    def __init__(
        self,
        ar_order: int = 1,
        integration_order: int = 1,
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
        self.integration_order = validate_nonnegative_int(
            integration_order,
            name="integration_order",
        )
        self.core_model = KalmanSTARMA(
            ar_order=ar_order,
            ma_order=ma_order,
            covariance_type=covariance_type,
            include_intercept=include_intercept,
            initialization=initialization,
            diffuse_scale=diffuse_scale,
            enforce_stationarity=enforce_stationarity,
            stability_margin=stability_margin,
            enforce_invertibility=enforce_invertibility,
            invertibility_margin=invertibility_margin,
            max_iter=max_iter,
            tol=tol,
        )
        self.result_: KalmanSTARIMAResult | None = None
        self.data_: FloatArray | None = None
        self.differenced_data_: FloatArray | None = None
        self.differencing_state_: DifferencingState | None = None

    @property
    def ar_order(self) -> int:
        """Stationary autoregressive order `p`."""
        return self.core_model.ar_order

    @property
    def ma_order(self) -> int:
        """Stationary moving-average order `q`."""
        return self.core_model.ma_order

    @property
    def order(self) -> tuple[int, int, int]:
        """Model order `(p, d, q)`."""
        return self.ar_order, self.integration_order, self.ma_order

    @property
    def original_scale_forecast_available(self) -> bool:
        """Whether finite terminal anchors permit inverse differencing."""
        return self.result_ is not None and self.differencing_state_ is not None

    def fit(
        self,
        data: Any,
        weights: Any,
        *,
        start_params: Any | None = None,
        start_covariance: Any | None = None,
    ) -> KalmanSTARIMAResult:
        """Fit the differenced stationary model by Gaussian Kalman likelihood."""
        observations, differenced, state = _difference_incomplete(
            data,
            order=self.integration_order,
        )
        core_result = self.core_model.fit(
            differenced,
            weights,
            start_params=start_params,
            start_covariance=start_covariance,
        )
        result = KalmanSTARIMAResult(
            core_result=core_result,
            integration_order=self.integration_order,
            n_original_rows=observations.shape[0],
            n_differenced_rows=differenced.shape[0],
            original_missing_cells=int(np.count_nonzero(~np.isfinite(observations))),
            differenced_missing_cells=int(np.count_nonzero(~np.isfinite(differenced))),
            original_scale_forecast_available=state is not None,
        )
        self.result_ = result
        self.data_ = observations.copy()
        self.differenced_data_ = differenced.copy()
        self.differencing_state_ = state
        return result

    def _require_fit(self) -> KalmanSTARIMAResult:
        if self.result_ is None:
            raise RuntimeError("fit must be called before using the fitted model")
        return self.result_

    def _transform_new(self, data: Any) -> FloatArray:
        _observations, differenced, _state = _difference_incomplete(
            data,
            order=self.integration_order,
        )
        return differenced

    def admissibility(self) -> STARMAAdmissibility:
        """Return stationary-core AR and inverse-MA diagnostics."""
        self._require_fit()
        return self.core_model.admissibility()

    def to_state_space(self) -> StateSpaceModel:
        """Return the stationary state space for the differenced process."""
        self._require_fit()
        return self.core_model.to_state_space()

    def filter(self, data: Any | None = None) -> KalmanFilterResult:
        """Filter the differenced training sample or transformed new data."""
        self._require_fit()
        if data is None:
            return self.core_model.filter()
        return self.core_model.filter(self._transform_new(data))

    def smooth(
        self,
        data: Any | None = None,
        *,
        rcond: float = 1e-10,
    ) -> KalmanSmootherResult:
        """Smooth the differenced training sample or transformed new data."""
        self._require_fit()
        if data is None:
            return self.core_model.smooth(rcond=rcond)
        return self.core_model.smooth(self._transform_new(data), rcond=rcond)

    def smooth_innovation_disturbances(
        self,
        data: Any | None = None,
        *,
        rcond: float = 1e-10,
    ) -> InnovationDisturbanceResult:
        """Smooth stationary innovations driving the differenced process."""
        self._require_fit()
        if data is None:
            return self.core_model.smooth_innovation_disturbances(rcond=rcond)
        return self.core_model.smooth_innovation_disturbances(
            self._transform_new(data),
            rcond=rcond,
        )

    def infer(
        self,
        *,
        relative_step: float = 1e-4,
        absolute_step: float = 1e-6,
        rcond: float = 1e-10,
        allow_singular: bool = False,
    ) -> LikelihoodInferenceResult:
        """Infer stationary parameters under the conditional likelihood."""
        self._require_fit()
        return self.core_model.infer(
            relative_step=relative_step,
            absolute_step=absolute_step,
            rcond=rcond,
            allow_singular=allow_singular,
        )

    def predict_differenced(self, steps: int = 1) -> FloatArray:
        """Return recursive means on the highest ordinary-difference scale."""
        self._require_fit()
        return self.core_model.predict(steps=steps)

    def predict(self, steps: int = 1) -> FloatArray:
        """Return recursively inverse-differenced original-scale means."""
        self._require_fit()
        if self.differencing_state_ is None:
            raise RuntimeError(
                "original-scale prediction requires finite terminal differencing "
                "anchors; use predict_differenced() or refit with finite trailing data"
            )
        return self.differencing_state_.inverse_forecast(
            self.predict_differenced(steps=steps)
        )

    def fitted_differenced(self) -> FloatArray:
        """Return one-step predictions on the differenced scale."""
        return self.filter().predicted_observations

    def fitted_original(self) -> FloatArray:
        """Return aligned one-step predictions on the original scale."""
        self._require_fit()
        if self.data_ is None:
            raise RuntimeError("training data are unavailable")
        return _restore_incomplete_fitted(
            self.data_,
            self.fitted_differenced(),
            order=self.integration_order,
        )


__all__ = ["KalmanSTARIMA", "KalmanSTARIMAResult"]
