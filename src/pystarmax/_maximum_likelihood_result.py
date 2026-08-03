# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Result objects for Kalman STARMA maximum likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pystarmax._maximum_likelihood_utils import _freeze_covariance, _freeze_float
from pystarmax._validation import FloatArray
from pystarmax.state_space import KalmanFilterResult


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
            raise ValueError(
                "optimizer_parameter_names must match raw_optimizer_params"
            )
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
            header + [coefficients, "-" * 72, "Innovation covariance", covariance]
        )
