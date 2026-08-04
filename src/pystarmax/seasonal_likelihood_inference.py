# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Observed-information inference for multiplicative seasonal Kalman STARIMA."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import stats

from pystarmax._maximum_likelihood_utils import _CovarianceCodec
from pystarmax._validation import FloatArray
from pystarmax.likelihood_inference import (
    LikelihoodInferenceResult,
    finite_difference_curvature,
)
from pystarmax.seasonal_maximum_likelihood import (
    SeasonalKalmanSTARIMA as _SeasonalKalmanSTARIMA,
)
from pystarmax.seasonal_maximum_likelihood import (
    _companion_radius,
    _operator_state_space,
)
from pystarmax.state_space import Initialization, kalman_filter


def _negative_log_likelihood(model: Any, raw: FloatArray) -> float:
    if (
        model.result_ is None
        or model.transformed_data_ is None
        or model.weights_ is None
    ):
        raise RuntimeError("fit must be called before likelihood inference")
    result = model.result_
    transformed = model.transformed_data_
    weights = model.weights_
    coefficient_size = int(result.params.size)
    codec = _CovarianceCodec(model.covariance_type, transformed.shape[1])
    expanded = model._expanded(raw[:coefficient_size], weights)
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
        n_locations=weights.n_locations,
    )
    ma_radius = _companion_radius(
        expanded[7],
        expanded[8],
        inverse_sign=True,
        n_locations=weights.n_locations,
    )
    invalid_base = 1e12
    squared_excess = 0.0
    if model.enforce_stationarity and ar_radius >= result.stability_limit:
        squared_excess += (ar_radius - result.stability_limit) ** 2
    if model.enforce_invertibility and ma_radius >= result.invertibility_limit:
        squared_excess += (ma_radius - result.invertibility_limit) ** 2
    if squared_excess > 0.0:
        return float(invalid_base + invalid_base * squared_excess + 1e-8 * (raw @ raw))
    filtered = kalman_filter(
        transformed,
        state_space,
        initialization=cast(Initialization, model.initialization),
        diffuse_scale=model.diffuse_scale,
    )
    return float(-filtered.log_likelihood)


def infer_seasonal_kalman_starima(
    model: Any,
    *,
    relative_step: float = 1e-4,
    absolute_step: float = 1e-6,
    rcond: float = 1e-10,
    allow_singular: bool = False,
) -> LikelihoodInferenceResult:
    """Compute observed-information inference for fitted seasonal factors.

    Curvature is evaluated on the raw optimizer scale. Every stencil point is
    reconstructed through the complete multiplicative matrix expansion, and
    points entering an enabled expanded AR-stationarity or MA-invertibility
    penalty region are rejected rather than interpreted as likelihood
    curvature.
    """
    if model.result_ is None:
        raise RuntimeError("fit must be called before likelihood inference")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be positive and finite")
    estimates = np.asarray(model.result_.raw_optimizer_params, dtype=float)
    curvature = finite_difference_curvature(
        lambda value: _negative_log_likelihood(model, value),
        estimates,
        relative_step=relative_step,
        absolute_step=absolute_step,
        invalid_threshold=1e11,
    )
    hessian = 0.5 * (curvature.hessian + curvature.hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    threshold = rcond * scale
    rank = int(np.count_nonzero(np.abs(eigenvalues) > threshold))
    positive_definite = bool(np.all(eigenvalues > threshold))
    used_pseudoinverse = False

    if positive_definite:
        inverse_eigenvalues = 1.0 / eigenvalues
    else:
        if not allow_singular:
            raise np.linalg.LinAlgError(
                "observed-information Hessian is not positive definite and full "
                f"rank (rank={rank}, minimum eigenvalue={eigenvalues[0]:.6g}); "
                "inspect curvature or set allow_singular=True explicitly"
            )
        used_pseudoinverse = True
        inverse_eigenvalues = np.zeros_like(eigenvalues)
        np.divide(
            1.0,
            eigenvalues,
            out=inverse_eigenvalues,
            where=eigenvalues > threshold,
        )

    covariance = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    positive_error = standard_errors > 0.0
    z_values = np.divide(
        estimates,
        standard_errors,
        out=np.zeros_like(estimates),
        where=positive_error,
    )
    p_values = 2.0 * stats.norm.sf(np.abs(z_values))
    p_values = np.where(positive_error, p_values, 1.0)
    denominator = np.outer(standard_errors, standard_errors)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, np.where(positive_error, 1.0, 0.0))
    retained = np.abs(eigenvalues) > threshold
    if np.any(retained):
        condition_number = float(
            np.max(np.abs(eigenvalues[retained]))
            / np.min(np.abs(eigenvalues[retained]))
        )
    else:
        condition_number = np.inf

    result = model.result_
    return LikelihoodInferenceResult(
        parameter_names=result.optimizer_parameter_names,
        estimates=estimates,
        hessian=hessian,
        covariance=covariance,
        standard_errors=standard_errors,
        z_values=z_values,
        p_values=p_values,
        correlation=correlation,
        eigenvalues=eigenvalues,
        gradient=curvature.gradient,
        steps=curvature.steps,
        n_dynamic_params=result.params.size,
        covariance_type=model.covariance_type,
        n_locations=result.innovation_covariance.shape[0],
        rank=rank,
        condition_number=condition_number,
        positive_definite=positive_definite,
        used_pseudoinverse=used_pseudoinverse,
        n_function_evaluations=curvature.n_function_evaluations,
        objective_value=curvature.function_value,
        stability_boundary_distance=(
            result.stability_limit - result.ar_spectral_radius
        ),
        invertibility_boundary_distance=(
            result.invertibility_limit - result.ma_inverse_spectral_radius
        ),
    )


class SeasonalKalmanSTARIMA(_SeasonalKalmanSTARIMA):
    """Seasonal Kalman STARIMA with observed-information inference."""

    def infer(
        self,
        *,
        relative_step: float = 1e-4,
        absolute_step: float = 1e-6,
        rcond: float = 1e-10,
        allow_singular: bool = False,
    ) -> LikelihoodInferenceResult:
        """Infer seasonal factors and covariance parameters from curvature."""
        return infer_seasonal_kalman_starima(
            self,
            relative_step=relative_step,
            absolute_step=absolute_step,
            rcond=rcond,
            allow_singular=allow_singular,
        )


__all__ = [
    "SeasonalKalmanSTARIMA",
    "infer_seasonal_kalman_starima",
]
