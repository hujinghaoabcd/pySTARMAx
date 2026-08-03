# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Delta-method inference for natural innovation covariance parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from pystarmax._maximum_likelihood_utils import CovarianceType, _CovarianceCodec
from pystarmax._validation import FloatArray

if TYPE_CHECKING:
    from pystarmax.likelihood_inference import LikelihoodInferenceResult


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _validate_locations(n_locations: int) -> int:
    value = int(n_locations)
    if value <= 0 or value != n_locations:
        raise ValueError("n_locations must be a positive integer")
    return value


def _element_names(
    covariance_type: CovarianceType,
    indices: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    if covariance_type == "scalar":
        return ("variance.shared",)
    names: list[str] = []
    for row, column in indices:
        if row == column:
            names.append(f"variance.location{row}")
        else:
            names.append(f"covariance.location{row}.location{column}")
    return tuple(names)


def _full_cholesky_and_jacobian(
    raw_parameters: FloatArray,
    n_locations: int,
) -> tuple[FloatArray, FloatArray]:
    lower_rows, lower_columns = np.tril_indices(n_locations)
    cholesky = np.zeros((n_locations, n_locations), dtype=float)
    derivatives = np.empty(raw_parameters.size, dtype=float)
    for index, (row, column) in enumerate(zip(lower_rows, lower_columns, strict=True)):
        raw_value = raw_parameters[index]
        if row == column:
            cholesky[row, column] = np.exp(raw_value)
            derivatives[index] = cholesky[row, column]
        else:
            cholesky[row, column] = raw_value
            derivatives[index] = 1.0

    n_elements = raw_parameters.size
    jacobian = np.zeros((n_elements, n_elements), dtype=float)
    for element_index, (row, column) in enumerate(
        zip(lower_rows, lower_columns, strict=True)
    ):
        for raw_index, (factor_row, factor_column) in enumerate(
            zip(lower_rows, lower_columns, strict=True)
        ):
            derivative = derivatives[raw_index]
            if row == factor_row:
                jacobian[element_index, raw_index] += (
                    derivative * cholesky[column, factor_column]
                )
            if column == factor_row:
                jacobian[element_index, raw_index] += (
                    derivative * cholesky[row, factor_column]
                )
    return cast(FloatArray, cholesky), cast(FloatArray, jacobian)


@dataclass(frozen=True, slots=True)
class InnovationCovarianceTransform:
    """Natural covariance elements and their Jacobian from optimizer parameters."""

    covariance_type: CovarianceType
    n_locations: int
    raw_parameter_names: tuple[str, ...]
    raw_parameters: FloatArray
    covariance_matrix: FloatArray
    element_names: tuple[str, ...]
    element_indices: tuple[tuple[int, int], ...]
    elements: FloatArray
    jacobian: FloatArray

    def __post_init__(self) -> None:
        if self.covariance_type not in {"scalar", "diagonal", "full"}:
            raise ValueError("covariance_type must be 'scalar', 'diagonal', or 'full'")
        n_locations = _validate_locations(self.n_locations)
        raw = _freeze_float(self.raw_parameters, name="raw_parameters", ndim=1)
        covariance = _freeze_float(
            self.covariance_matrix,
            name="covariance_matrix",
            ndim=2,
        )
        elements = _freeze_float(self.elements, name="elements", ndim=1)
        jacobian = _freeze_float(self.jacobian, name="jacobian", ndim=2)
        if covariance.shape != (n_locations, n_locations):
            raise ValueError("covariance_matrix has an invalid shape")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance_matrix must be symmetric")
        if len(self.raw_parameter_names) != raw.size:
            raise ValueError("raw_parameter_names must match raw_parameters")
        if len(self.element_names) != elements.size:
            raise ValueError("element_names must match elements")
        if len(self.element_indices) != elements.size:
            raise ValueError("element_indices must match elements")
        if jacobian.shape != (elements.size, raw.size):
            raise ValueError("jacobian has an invalid shape")
        for row, column in self.element_indices:
            if not (0 <= column <= row < n_locations):
                raise ValueError("element_indices must reference the lower triangle")
        object.__setattr__(self, "n_locations", n_locations)
        object.__setattr__(self, "raw_parameters", raw)
        object.__setattr__(self, "covariance_matrix", covariance)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "jacobian", jacobian)

    @property
    def n_elements(self) -> int:
        """Number of natural free covariance elements."""
        return int(self.elements.size)

    @property
    def n_raw_parameters(self) -> int:
        """Number of optimizer covariance-factor parameters."""
        return int(self.raw_parameters.size)

    @property
    def element_table(self) -> pd.DataFrame:
        """Natural covariance-element estimates."""
        return pd.DataFrame(
            {"estimate": self.elements},
            index=pd.Index(self.element_names, name="parameter"),
        )


def innovation_covariance_transform(
    raw_parameters: Any,
    *,
    covariance_type: CovarianceType,
    n_locations: int,
) -> InnovationCovarianceTransform:
    """Transform optimizer covariance parameters and return an analytic Jacobian."""
    locations = _validate_locations(n_locations)
    codec = _CovarianceCodec(covariance_type, locations)
    raw = np.asarray(raw_parameters, dtype=float)
    if raw.shape != (codec.size,):
        raise ValueError(f"raw_parameters must contain exactly {codec.size} values")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_parameters must contain finite values")
    raw = np.ascontiguousarray(raw, dtype=float)
    covariance = codec.unpack(raw)

    if covariance_type == "scalar":
        variance = float(covariance[0, 0])
        indices = ((0, 0),)
        elements = np.array([variance], dtype=float)
        jacobian = np.array([[2.0 * variance]], dtype=float)
    elif covariance_type == "diagonal":
        variances = np.diag(covariance).astype(float, copy=True)
        indices = tuple((index, index) for index in range(locations))
        elements = variances
        jacobian = np.diag(2.0 * variances)
    else:
        lower_rows, lower_columns = np.tril_indices(locations)
        indices = tuple(
            (int(row), int(column))
            for row, column in zip(lower_rows, lower_columns, strict=True)
        )
        elements = covariance[lower_rows, lower_columns].astype(float, copy=True)
        _cholesky, jacobian = _full_cholesky_and_jacobian(raw, locations)

    return InnovationCovarianceTransform(
        covariance_type=covariance_type,
        n_locations=locations,
        raw_parameter_names=codec.names,
        raw_parameters=raw,
        covariance_matrix=covariance,
        element_names=_element_names(covariance_type, indices),
        element_indices=indices,
        elements=elements,
        jacobian=jacobian,
    )


def delta_method_covariance(jacobian: Any, covariance: Any) -> FloatArray:
    """Propagate a covariance matrix through a first-order Jacobian."""
    derivative = np.asarray(jacobian, dtype=float)
    source = np.asarray(covariance, dtype=float)
    if derivative.ndim != 2:
        raise ValueError("jacobian must be two-dimensional")
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("covariance must be square")
    if derivative.shape[1] != source.shape[0]:
        raise ValueError("jacobian and covariance dimensions do not match")
    if not np.all(np.isfinite(derivative)) or not np.all(np.isfinite(source)):
        raise ValueError("jacobian and covariance must contain finite values")
    propagated = derivative @ source @ derivative.T
    propagated = 0.5 * (propagated + propagated.T)
    return cast(FloatArray, np.asarray(propagated, dtype=float))


@dataclass(frozen=True, slots=True)
class InnovationCovarianceInference:
    """First-order delta-method inference for natural covariance elements."""

    transform: InnovationCovarianceTransform
    covariance: FloatArray
    standard_errors: FloatArray
    correlation: FloatArray
    source_covariance: FloatArray
    dynamic_parameter_names: tuple[str, ...]
    dynamic_cross_covariance: FloatArray

    def __post_init__(self) -> None:
        n_elements = self.transform.n_elements
        n_raw = self.transform.n_raw_parameters
        covariance = _freeze_float(self.covariance, name="covariance", ndim=2)
        standard_errors = _freeze_float(
            self.standard_errors,
            name="standard_errors",
            ndim=1,
        )
        correlation = _freeze_float(self.correlation, name="correlation", ndim=2)
        source_covariance = _freeze_float(
            self.source_covariance,
            name="source_covariance",
            ndim=2,
        )
        dynamic_cross_covariance = _freeze_float(
            self.dynamic_cross_covariance,
            name="dynamic_cross_covariance",
            ndim=2,
        )
        if covariance.shape != (n_elements, n_elements):
            raise ValueError("covariance has an invalid shape")
        if standard_errors.shape != (n_elements,):
            raise ValueError("standard_errors has an invalid shape")
        if correlation.shape != (n_elements, n_elements):
            raise ValueError("correlation has an invalid shape")
        if source_covariance.shape != (n_raw, n_raw):
            raise ValueError("source_covariance has an invalid shape")
        if dynamic_cross_covariance.shape != (
            len(self.dynamic_parameter_names),
            n_elements,
        ):
            raise ValueError("dynamic_cross_covariance has an invalid shape")
        if np.any(standard_errors < 0.0):
            raise ValueError("standard_errors must be non-negative")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "source_covariance", source_covariance)
        object.__setattr__(
            self,
            "dynamic_cross_covariance",
            dynamic_cross_covariance,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Natural covariance-element names."""
        return self.transform.element_names

    @property
    def estimates(self) -> FloatArray:
        """Natural covariance-element estimates."""
        return self.transform.elements

    @property
    def covariance_matrix(self) -> FloatArray:
        """Fitted innovation covariance matrix."""
        return self.transform.covariance_matrix

    @property
    def table(self) -> pd.DataFrame:
        """Delta-method estimates and standard errors."""
        return pd.DataFrame(
            {
                "estimate": self.estimates,
                "standard_error": self.standard_errors,
            },
            index=pd.Index(self.parameter_names, name="parameter"),
        )

    def confidence_intervals(self, level: float = 0.95) -> pd.DataFrame:
        """Return unbounded normal delta-method confidence intervals."""
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("level must be between zero and one")
        critical = float(stats.norm.ppf(0.5 + level / 2.0))
        margin = critical * self.standard_errors
        return pd.DataFrame(
            {
                "estimate": self.estimates,
                "lower": self.estimates - margin,
                "upper": self.estimates + margin,
            },
            index=pd.Index(self.parameter_names, name="parameter"),
        )

    @property
    def standard_error_matrix(self) -> FloatArray:
        """Return standard errors arranged like the covariance matrix."""
        matrix = np.zeros(
            (self.transform.n_locations, self.transform.n_locations),
            dtype=float,
        )
        if self.transform.covariance_type == "scalar":
            np.fill_diagonal(matrix, self.standard_errors[0])
        else:
            for value, (row, column) in zip(
                self.standard_errors,
                self.transform.element_indices,
                strict=True,
            ):
                matrix[row, column] = value
                matrix[column, row] = value
        matrix.setflags(write=False)
        return cast(FloatArray, matrix)

    def summary(self) -> str:
        """Return a compact natural-scale covariance inference summary."""
        header = [
            "pySTARMAx innovation covariance delta-method inference",
            "=" * 72,
            f"Covariance type: {self.transform.covariance_type}",
            f"Locations: {self.transform.n_locations}",
            f"Natural parameters: {self.transform.n_elements}",
            "Intervals: unbounded first-order normal approximation",
            "-" * 72,
        ]
        table = self.table.to_string(float_format=lambda value: f"{value: .6f}")
        return "\n".join(header + [table])


def innovation_covariance_delta_inference(
    inference: LikelihoodInferenceResult,
) -> InnovationCovarianceInference:
    """Transform optimizer-scale covariance uncertainty to natural elements."""
    stop = inference.n_dynamic_params
    raw_parameters = inference.estimates[stop:]
    transform = innovation_covariance_transform(
        raw_parameters,
        covariance_type=inference.covariance_type,
        n_locations=inference.n_locations,
    )
    if stop + transform.n_raw_parameters != inference.estimates.size:
        raise ValueError("inference covariance parameter count is inconsistent")
    source_covariance = inference.covariance[stop:, stop:]
    propagated = delta_method_covariance(
        transform.jacobian,
        source_covariance,
    )
    variances = np.clip(np.diag(propagated), 0.0, np.inf)
    standard_errors = np.sqrt(variances)
    denominator = np.outer(standard_errors, standard_errors)
    correlation = np.divide(
        propagated,
        denominator,
        out=np.zeros_like(propagated),
        where=denominator > 0.0,
    )
    np.fill_diagonal(
        correlation,
        np.where(standard_errors > 0.0, 1.0, 0.0),
    )
    source_cross = inference.covariance[:stop, stop:]
    dynamic_cross = source_cross @ transform.jacobian.T
    return InnovationCovarianceInference(
        transform=transform,
        covariance=propagated,
        standard_errors=standard_errors,
        correlation=correlation,
        source_covariance=source_covariance,
        dynamic_parameter_names=inference.parameter_names[:stop],
        dynamic_cross_covariance=dynamic_cross,
    )


__all__ = [
    "InnovationCovarianceInference",
    "InnovationCovarianceTransform",
    "delta_method_covariance",
    "innovation_covariance_delta_inference",
    "innovation_covariance_transform",
]
