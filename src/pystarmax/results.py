# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Structured model results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class STARMAResult:
    """Immutable result returned by :meth:`pystarmax.STARMA.fit`."""

    params: FloatArray
    parameter_names: tuple[str, ...]
    standard_errors: FloatArray
    t_values: FloatArray
    p_values: FloatArray
    fitted_values: FloatArray
    residuals: FloatArray
    innovation_covariance: FloatArray
    sigma2: float
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    degrees_of_freedom: int
    converged: bool
    n_iterations: int
    method: str
    conditional_lag: int

    @property
    def n_params(self) -> int:
        """Number of estimated coefficients."""
        return int(self.params.size)

    @property
    def coefficients(self) -> pd.DataFrame:
        """Coefficient table as a pandas data frame."""
        return pd.DataFrame(
            {
                "coefficient": self.params,
                "std_error": self.standard_errors,
                "t_value": self.t_values,
                "p_value": self.p_values,
            },
            index=pd.Index(self.parameter_names, name="parameter"),
        )

    def summary(self) -> str:
        """Return a compact plain-text model summary."""
        header = [
            "pySTARMAx model result",
            "=" * 72,
            f"Method: {self.method}",
            f"Observations: {self.n_observations}",
            f"Parameters: {self.n_params}",
            f"Converged: {self.converged} ({self.n_iterations} iteration(s))",
            f"Conditional lag: {self.conditional_lag}",
            f"Log likelihood: {self.log_likelihood:.6f}",
            f"AIC: {self.aic:.6f}",
            f"BIC: {self.bic:.6f}",
            "-" * 72,
        ]
        table = self.coefficients.to_string(float_format=lambda value: f"{value: .6f}")
        footer = ["-" * 72, "Pre-sample fitted values and residuals are NaN."]
        return "\n".join(header + [table] + footer)
