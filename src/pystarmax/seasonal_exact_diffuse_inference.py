# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Observed-information inference for seasonal exact diffuse STARIMA.

Every curvature candidate expands the ordinary and seasonal factor parameters,
rebuilds the stationary transformed state, rebuilds the original-level seasonal
exact diffuse state, and evaluates the original observations. Conditional
transformed-data likelihoods and finite large-variance diffuse approximations are
not used.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import stats

from pystarmax._maximum_likelihood_utils import CovarianceType, _CovarianceCodec
from pystarmax._validation import FloatArray
from pystarmax.exact_seasonal_integrated import (
    build_exact_seasonal_integrated_state_space,
)
from pystarmax.likelihood_inference import (
    LikelihoodInferenceResult,
    finite_difference_curvature,
)
from pystarmax.seasonal_maximum_likelihood import (
    _companion_radius,
    _operator_state_space,
)


def _negative_seasonal_exact_diffuse_log_likelihood(
    model: Any,
    raw: FloatArray,
) -> float:
    if model.result_ is None or model.data_ is None or model.weights_ is None:
        raise RuntimeError("fit must be called before seasonal exact diffuse inference")
    observations = model.data_
    weights = model.weights_
    result = model.result_
    coefficient_size = int(result.params.size)
    covariance_type = cast(CovarianceType, model.core_model.covariance_type)
    codec = _CovarianceCodec(covariance_type, observations.shape[1])
    invalid_base = 1e12

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            expanded = model.core_model._expanded(
                cast(FloatArray, raw[:coefficient_size]),
                weights,
            )
            covariance = codec.unpack(raw[coefficient_size:])
            transformed = _operator_state_space(
                expanded[5],
                expanded[6],
                expanded[7],
                expanded[8],
                covariance,
                intercept=expanded[0],
            )
            integrated = build_exact_seasonal_integrated_state_space(
                transformed,
                model.integration_order,
                model.seasonal_integration_order,
                model.seasonal_period,
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
            squared_excess = 0.0
            if (
                model.core_model.enforce_stationarity
                and ar_radius >= result.stability_limit
            ):
                squared_excess += (ar_radius - result.stability_limit) ** 2
            if (
                model.core_model.enforce_invertibility
                and ma_radius >= result.invertibility_limit
            ):
                squared_excess += (ma_radius - result.invertibility_limit) ** 2
            if squared_excess > 0.0:
                return float(
                    invalid_base + invalid_base * squared_excess + 1e-8 * (raw @ raw)
                )
            filtered = integrated.filter(
                observations,
                tolerance=model.diffuse_tolerance,
            )
            value = -filtered.log_likelihood
            return float(value) if np.isfinite(value) else invalid_base
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
        return float(invalid_base + 1e-8 * (raw @ raw))


def infer_seasonal_exact_diffuse_kalman_starima(
    model: Any,
    *,
    relative_step: float = 1e-4,
    absolute_step: float = 1e-6,
    rcond: float = 1e-10,
    allow_singular: bool = False,
) -> LikelihoodInferenceResult:
    """Compute observed-information inference on the seasonal exact objective.

    Curvature is evaluated on the raw optimizer scale. Dynamic coordinates are
    the ordinary and seasonal AR/MA factor parameters retained by the fitted
    model; expanded cross-lag matrices are deterministic and are not additional
    parameters.
    """
    if model.result_ is None:
        raise RuntimeError("fit must be called before seasonal exact diffuse inference")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be positive and finite")

    result = model.result_
    estimates = np.asarray(result.raw_optimizer_params, dtype=float)
    curvature = finite_difference_curvature(
        lambda value: _negative_seasonal_exact_diffuse_log_likelihood(model, value),
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
                "seasonal exact diffuse observed-information Hessian is not "
                "positive definite and full rank "
                f"(rank={rank}, minimum eigenvalue={eigenvalues[0]:.6g}); "
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
        covariance_type=cast(CovarianceType, result.covariance_type),
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


__all__ = ["infer_seasonal_exact_diffuse_kalman_starima"]
