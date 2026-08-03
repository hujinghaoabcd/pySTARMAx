# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Likelihood-curvature inference for Kalman STARMA estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import pandas as pd
from scipy import stats

from pystarmax._maximum_likelihood_utils import CovarianceType, _CovarianceCodec
from pystarmax._validation import FloatArray
from pystarmax.admissibility import (
    autoregressive_spectral_radius,
    moving_average_inverse_spectral_radius,
)
from pystarmax.state_space import (
    Initialization,
    build_starma_state_space,
    kalman_filter,
)

if TYPE_CHECKING:
    from pystarmax.covariance_inference import InnovationCovarianceInference

ScalarFunction = Callable[[FloatArray], float]


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _step_sizes(
    point: FloatArray,
    *,
    relative_step: float,
    absolute_step: float,
) -> FloatArray:
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be positive and finite")
    if not np.isfinite(absolute_step) or absolute_step <= 0.0:
        raise ValueError("absolute_step must be positive and finite")
    steps = np.maximum(
        absolute_step,
        relative_step * np.maximum(1.0, np.abs(point)),
    )
    return cast(FloatArray, np.asarray(steps, dtype=float))


@dataclass(frozen=True, slots=True)
class FiniteDifferenceCurvature:
    """Central finite-difference gradient and Hessian."""

    point: FloatArray
    steps: FloatArray
    function_value: float
    gradient: FloatArray
    hessian: FloatArray
    n_function_evaluations: int

    def __post_init__(self) -> None:
        point = _freeze_float(self.point, name="point", ndim=1)
        steps = _freeze_float(self.steps, name="steps", ndim=1)
        gradient = _freeze_float(self.gradient, name="gradient", ndim=1)
        hessian = _freeze_float(self.hessian, name="hessian", ndim=2)
        if steps.shape != point.shape or gradient.shape != point.shape:
            raise ValueError("point, steps, and gradient must have matching shapes")
        if hessian.shape != (point.size, point.size):
            raise ValueError("hessian must be square with one row per parameter")
        function_value = float(self.function_value)
        if not np.isfinite(function_value):
            raise ValueError("function_value must be finite")
        evaluations = int(self.n_function_evaluations)
        if evaluations < 1:
            raise ValueError("n_function_evaluations must be positive")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "function_value", function_value)
        object.__setattr__(self, "n_function_evaluations", evaluations)


def finite_difference_curvature(
    function: ScalarFunction,
    point: Any,
    *,
    relative_step: float = 1e-4,
    absolute_step: float = 1e-6,
    invalid_threshold: float | None = None,
) -> FiniteDifferenceCurvature:
    """Evaluate a central finite-difference gradient and Hessian.

    The diagonal stencil uses three points and each off-diagonal element uses the
    four-corner mixed-partial stencil. ``invalid_threshold`` can reject objective
    values that indicate a feasibility penalty rather than a valid likelihood.
    """
    center = np.asarray(point, dtype=float)
    if center.ndim != 1 or center.size == 0:
        raise ValueError("point must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(center)):
        raise ValueError("point must contain finite values")
    if invalid_threshold is not None:
        threshold = float(invalid_threshold)
        if not np.isfinite(threshold):
            raise ValueError("invalid_threshold must be finite")
    else:
        threshold = np.inf
    steps = _step_sizes(
        cast(FloatArray, center),
        relative_step=relative_step,
        absolute_step=absolute_step,
    )
    evaluations = 0

    def evaluate(candidate: FloatArray) -> float:
        nonlocal evaluations
        value = float(function(candidate))
        evaluations += 1
        if not np.isfinite(value):
            raise ValueError("finite-difference objective returned a non-finite value")
        if value >= threshold:
            raise ValueError(
                "finite-difference stencil entered an invalid parameter region; "
                "reduce the step size or move away from the feasibility boundary"
            )
        return value

    function_value = evaluate(cast(FloatArray, center))
    n_params = center.size
    gradient = np.empty(n_params, dtype=float)
    hessian = np.empty((n_params, n_params), dtype=float)
    plus_values = np.empty(n_params, dtype=float)
    minus_values = np.empty(n_params, dtype=float)

    for index in range(n_params):
        plus = center.copy()
        minus = center.copy()
        plus[index] += steps[index]
        minus[index] -= steps[index]
        plus_values[index] = evaluate(cast(FloatArray, plus))
        minus_values[index] = evaluate(cast(FloatArray, minus))
        gradient[index] = (plus_values[index] - minus_values[index]) / (
            2.0 * steps[index]
        )
        hessian[index, index] = (
            plus_values[index] - 2.0 * function_value + minus_values[index]
        ) / steps[index] ** 2

    for first in range(n_params):
        for second in range(first + 1, n_params):
            plus_plus = center.copy()
            plus_minus = center.copy()
            minus_plus = center.copy()
            minus_minus = center.copy()
            plus_plus[first] += steps[first]
            plus_plus[second] += steps[second]
            plus_minus[first] += steps[first]
            plus_minus[second] -= steps[second]
            minus_plus[first] -= steps[first]
            minus_plus[second] += steps[second]
            minus_minus[first] -= steps[first]
            minus_minus[second] -= steps[second]
            mixed = (
                evaluate(cast(FloatArray, plus_plus))
                - evaluate(cast(FloatArray, plus_minus))
                - evaluate(cast(FloatArray, minus_plus))
                + evaluate(cast(FloatArray, minus_minus))
            ) / (4.0 * steps[first] * steps[second])
            hessian[first, second] = mixed
            hessian[second, first] = mixed

    hessian = 0.5 * (hessian + hessian.T)
    return FiniteDifferenceCurvature(
        point=center,
        steps=steps,
        function_value=function_value,
        gradient=gradient,
        hessian=hessian,
        n_function_evaluations=evaluations,
    )


def finite_difference_hessian(
    function: ScalarFunction,
    point: Any,
    *,
    relative_step: float = 1e-4,
    absolute_step: float = 1e-6,
) -> FloatArray:
    """Return only the central finite-difference Hessian."""
    return finite_difference_curvature(
        function,
        point,
        relative_step=relative_step,
        absolute_step=absolute_step,
    ).hessian


@dataclass(frozen=True, slots=True)
class LikelihoodInferenceResult:
    """Likelihood-curvature inference on the optimizer parameter scale."""

    parameter_names: tuple[str, ...]
    estimates: FloatArray
    hessian: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    z_values: FloatArray
    p_values: FloatArray
    correlation: FloatArray
    eigenvalues: FloatArray
    gradient: FloatArray
    steps: FloatArray
    n_dynamic_params: int
    covariance_type: CovarianceType
    n_locations: int
    rank: int
    condition_number: float
    positive_definite: bool
    used_pseudoinverse: bool
    n_function_evaluations: int
    objective_value: float
    stability_boundary_distance: float
    invertibility_boundary_distance: float

    def __post_init__(self) -> None:
        estimates = _freeze_float(self.estimates, name="estimates", ndim=1)
        n_params = estimates.size
        arrays: list[FloatArray] = []
        for name, value, ndim, shape in (
            ("hessian", self.hessian, 2, (n_params, n_params)),
            ("covariance", self.covariance, 2, (n_params, n_params)),
            ("standard_errors", self.standard_errors, 1, (n_params,)),
            ("z_values", self.z_values, 1, (n_params,)),
            ("p_values", self.p_values, 1, (n_params,)),
            ("correlation", self.correlation, 2, (n_params, n_params)),
            ("eigenvalues", self.eigenvalues, 1, (n_params,)),
            ("gradient", self.gradient, 1, (n_params,)),
            ("steps", self.steps, 1, (n_params,)),
        ):
            array = _freeze_float(value, name=name, ndim=ndim)
            if array.shape != shape:
                raise ValueError(f"{name} has an invalid shape")
            arrays.append(array)
        if len(self.parameter_names) != n_params:
            raise ValueError("parameter_names must match estimates")
        n_dynamic = int(self.n_dynamic_params)
        if not 0 <= n_dynamic <= n_params:
            raise ValueError("n_dynamic_params is outside the parameter range")
        if self.covariance_type not in {"scalar", "diagonal", "full"}:
            raise ValueError("covariance_type must be 'scalar', 'diagonal', or 'full'")
        n_locations = int(self.n_locations)
        if n_locations <= 0 or n_locations != self.n_locations:
            raise ValueError("n_locations must be a positive integer")
        rank = int(self.rank)
        if not 0 <= rank <= n_params:
            raise ValueError("rank is outside the parameter range")
        condition = float(self.condition_number)
        if np.isnan(condition) or condition <= 0.0:
            raise ValueError("condition_number must be positive")
        objective = float(self.objective_value)
        stability_boundary = float(self.stability_boundary_distance)
        invertibility_boundary = float(self.invertibility_boundary_distance)
        if not np.isfinite(objective):
            raise ValueError("objective_value must be finite")
        if not np.isfinite(stability_boundary) or not np.isfinite(
            invertibility_boundary
        ):
            raise ValueError("admissibility boundary distances must be finite")
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "hessian", arrays[0])
        object.__setattr__(self, "covariance", arrays[1])
        object.__setattr__(self, "standard_errors", arrays[2])
        object.__setattr__(self, "z_values", arrays[3])
        object.__setattr__(self, "p_values", arrays[4])
        object.__setattr__(self, "correlation", arrays[5])
        object.__setattr__(self, "eigenvalues", arrays[6])
        object.__setattr__(self, "gradient", arrays[7])
        object.__setattr__(self, "steps", arrays[8])
        object.__setattr__(self, "n_dynamic_params", n_dynamic)
        object.__setattr__(self, "n_locations", n_locations)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "condition_number", condition)
        object.__setattr__(self, "objective_value", objective)
        object.__setattr__(
            self,
            "stability_boundary_distance",
            stability_boundary,
        )
        object.__setattr__(
            self,
            "invertibility_boundary_distance",
            invertibility_boundary,
        )
        object.__setattr__(self, "positive_definite", bool(self.positive_definite))
        object.__setattr__(self, "used_pseudoinverse", bool(self.used_pseudoinverse))
        object.__setattr__(
            self,
            "n_function_evaluations",
            int(self.n_function_evaluations),
        )

    @property
    def max_abs_gradient(self) -> float:
        """Maximum absolute finite-difference score component."""
        return float(np.max(np.abs(self.gradient), initial=0.0))

    @property
    def minimum_admissibility_distance(self) -> float:
        """Smallest signed distance to the AR or inverse-MA boundary."""
        return float(
            min(
                self.stability_boundary_distance,
                self.invertibility_boundary_distance,
            )
        )

    @property
    def coefficient_table(self) -> pd.DataFrame:
        """Inference table for intercept and dynamic AR/MA coefficients."""
        stop = self.n_dynamic_params
        return pd.DataFrame(
            {
                "estimate": self.estimates[:stop],
                "standard_error": self.standard_errors[:stop],
                "z": self.z_values[:stop],
                "p_value": self.p_values[:stop],
            },
            index=pd.Index(self.parameter_names[:stop], name="parameter"),
        )

    @property
    def optimizer_table(self) -> pd.DataFrame:
        """Inference table for every raw optimizer parameter."""
        return pd.DataFrame(
            {
                "estimate": self.estimates,
                "standard_error": self.standard_errors,
                "z": self.z_values,
                "p_value": self.p_values,
            },
            index=pd.Index(self.parameter_names, name="parameter"),
        )

    def confidence_intervals(self, level: float = 0.95) -> pd.DataFrame:
        """Return normal-approximation intervals for dynamic coefficients."""
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("level must be between zero and one")
        critical = float(stats.norm.ppf(0.5 + level / 2.0))
        stop = self.n_dynamic_params
        estimates = self.estimates[:stop]
        margin = critical * self.standard_errors[:stop]
        return pd.DataFrame(
            {
                "estimate": estimates,
                "lower": estimates - margin,
                "upper": estimates + margin,
            },
            index=pd.Index(self.parameter_names[:stop], name="parameter"),
        )

    def innovation_covariance_inference(self) -> InnovationCovarianceInference:
        """Return natural-scale innovation covariance delta-method inference."""
        from pystarmax.covariance_inference import (
            innovation_covariance_delta_inference,
        )

        return innovation_covariance_delta_inference(self)

    def summary(self) -> str:
        """Return a compact curvature and coefficient summary."""
        header = [
            "pySTARMAx likelihood-curvature inference",
            "=" * 72,
            f"Parameters: {len(self.parameter_names)}",
            f"Hessian rank: {self.rank}",
            f"Positive definite: {self.positive_definite}",
            f"Used pseudoinverse: {self.used_pseudoinverse}",
            f"Condition number: {self.condition_number:.6g}",
            f"Maximum absolute score: {self.max_abs_gradient:.6g}",
            f"Stability-boundary distance: {self.stability_boundary_distance:.6g}",
            f"Invertibility-boundary distance: "
            f"{self.invertibility_boundary_distance:.6g}",
            "-" * 72,
        ]
        table = self.coefficient_table.to_string(
            float_format=lambda value: f"{value: .6f}"
        )
        return "\n".join(header + [table])


def _negative_log_likelihood(model: Any, raw: FloatArray) -> float:
    if model.result_ is None or model.data_ is None or model.weights_ is None:
        raise RuntimeError("fit must be called before likelihood inference")
    observations = model.data_
    weights = model.weights_
    coefficient_size = model.result_.params.size
    codec = _CovarianceCodec(model.covariance_type, observations.shape[1])
    intercept, ar_parameters, ma_parameters = model._split_coefficients(
        raw[:coefficient_size],
        n_weights=len(weights),
    )
    covariance = codec.unpack(raw[coefficient_size:])
    state_space = build_starma_state_space(
        ar_parameters,
        ma_parameters,
        weights,
        covariance,
        intercept=intercept,
    )
    spectral_radius = autoregressive_spectral_radius(ar_parameters, weights)
    ma_inverse_radius = moving_average_inverse_spectral_radius(
        ma_parameters,
        weights,
    )
    stability_limit = 1.0 - model.stability_margin
    invertibility_limit = 1.0 - model.invertibility_margin
    invalid_base = 1e12
    squared_excess = 0.0
    if model.enforce_stationarity and spectral_radius >= stability_limit:
        squared_excess += (spectral_radius - stability_limit) ** 2
    if model.enforce_invertibility and ma_inverse_radius >= invertibility_limit:
        squared_excess += (ma_inverse_radius - invertibility_limit) ** 2
    if squared_excess > 0.0:
        return float(invalid_base + invalid_base * squared_excess + 1e-8 * (raw @ raw))
    filtered = kalman_filter(
        observations,
        state_space,
        initialization=cast(Initialization, model.initialization),
        diffuse_scale=model.diffuse_scale,
    )
    return float(-filtered.log_likelihood)


def infer_kalman_starma(
    model: Any,
    *,
    relative_step: float = 1e-4,
    absolute_step: float = 1e-6,
    rcond: float = 1e-10,
    allow_singular: bool = False,
) -> LikelihoodInferenceResult:
    """Compute observed-information inference for a fitted Kalman STARMA model."""
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
    variances = np.clip(np.diag(covariance), 0.0, np.inf)
    standard_errors = np.sqrt(variances)
    z_values = np.divide(
        estimates,
        standard_errors,
        out=np.zeros_like(estimates),
        where=standard_errors > 0.0,
    )
    p_values = 2.0 * stats.norm.sf(np.abs(z_values))
    denominator = np.outer(standard_errors, standard_errors)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    np.fill_diagonal(
        correlation,
        np.where(standard_errors > 0.0, 1.0, 0.0),
    )
    retained = np.abs(eigenvalues) > threshold
    if np.any(retained):
        condition_number = float(
            np.max(np.abs(eigenvalues[retained]))
            / np.min(np.abs(eigenvalues[retained]))
        )
    else:
        condition_number = np.inf
    return LikelihoodInferenceResult(
        parameter_names=model.result_.optimizer_parameter_names,
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
        n_dynamic_params=model.result_.params.size,
        covariance_type=model.covariance_type,
        n_locations=observations.shape[1],
        rank=rank,
        condition_number=condition_number,
        positive_definite=positive_definite,
        used_pseudoinverse=used_pseudoinverse,
        n_function_evaluations=curvature.n_function_evaluations,
        objective_value=curvature.function_value,
        stability_boundary_distance=model.result_.stability_boundary_distance,
        invertibility_boundary_distance=(model.result_.invertibility_boundary_distance),
    )
