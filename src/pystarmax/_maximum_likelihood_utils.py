# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Internal parameter and covariance utilities for Kalman STARMA MLE."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

import numpy as np

from pystarmax._validation import FloatArray

CovarianceType: TypeAlias = Literal["scalar", "diagonal", "full"]


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _freeze_covariance(value: Any, *, n_locations: int) -> FloatArray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (n_locations, n_locations):
        raise ValueError("innovation_covariance has an invalid shape")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("innovation_covariance must contain finite values")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    minimum = float(np.min(eigenvalues))
    if minimum <= 0.0:
        if minimum < -1e-10 * scale:
            raise ValueError("innovation_covariance must be positive definite")
        raise ValueError("innovation_covariance must be strictly positive definite")
    frozen = np.ascontiguousarray(covariance, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _validate_incomplete_data(data: Any) -> FloatArray:
    observations = np.asarray(data, dtype=float)
    if observations.ndim != 2:
        raise ValueError("data must be a two-dimensional array")
    if observations.shape[0] < 3:
        raise ValueError("data must contain at least three time observations")
    if observations.shape[1] < 1:
        raise ValueError("data must contain at least one location")
    if np.any(np.isinf(observations)):
        raise ValueError("data must not contain infinite values")
    observed_per_location = np.sum(np.isfinite(observations), axis=0)
    if np.any(observed_per_location < 2):
        raise ValueError("each location must contain at least two observed values")
    return cast(FloatArray, np.ascontiguousarray(observations, dtype=float))


def _regularize_covariance(covariance: Any, *, n_locations: int) -> FloatArray:
    array = np.asarray(covariance, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError("start_covariance scalar must be positive and finite")
        array = np.eye(n_locations, dtype=float) * scalar
    elif array.ndim == 1:
        if array.shape != (n_locations,):
            raise ValueError("start_covariance vector must match the locations")
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError("start_covariance variances must be positive and finite")
        array = np.diag(array)
    elif array.shape != (n_locations, n_locations):
        raise ValueError("start_covariance matrix must match the locations")
    if not np.all(np.isfinite(array)):
        raise ValueError("start_covariance must contain finite values")
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.mean(np.abs(np.diag(symmetric)))))
    floor = np.finfo(float).eps * scale * 1000.0
    clipped = np.clip(eigenvalues, floor, np.inf)
    regularized = (eigenvectors * clipped) @ eigenvectors.T
    return cast(FloatArray, 0.5 * (regularized + regularized.T))


class _CovarianceCodec:
    def __init__(self, covariance_type: CovarianceType, n_locations: int) -> None:
        if covariance_type not in {"scalar", "diagonal", "full"}:
            raise ValueError("covariance_type must be 'scalar', 'diagonal', or 'full'")
        self.covariance_type = covariance_type
        self.n_locations = n_locations

    @property
    def size(self) -> int:
        if self.covariance_type == "scalar":
            return 1
        if self.covariance_type == "diagonal":
            return self.n_locations
        return self.n_locations * (self.n_locations + 1) // 2

    @property
    def names(self) -> tuple[str, ...]:
        if self.covariance_type == "scalar":
            return ("cov.log_std",)
        if self.covariance_type == "diagonal":
            return tuple(
                f"cov.log_std.location{location}"
                for location in range(self.n_locations)
            )
        names: list[str] = []
        for row in range(self.n_locations):
            for column in range(row + 1):
                if row == column:
                    names.append(f"cov.log_cholesky.location{row}")
                else:
                    names.append(f"cov.cholesky.location{row}.{column}")
        return tuple(names)

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        if self.covariance_type == "scalar":
            return [(-20.0, 20.0)]
        if self.covariance_type == "diagonal":
            return [(-20.0, 20.0)] * self.n_locations
        bounds: list[tuple[float | None, float | None]] = []
        for row in range(self.n_locations):
            for column in range(row + 1):
                if row == column:
                    bounds.append((-20.0, 20.0))
                else:
                    bounds.append((None, None))
        return bounds

    def pack(self, covariance: Any) -> FloatArray:
        regularized = _regularize_covariance(
            covariance,
            n_locations=self.n_locations,
        )
        if self.covariance_type == "scalar":
            variance = float(np.mean(np.diag(regularized)))
            return np.array([0.5 * np.log(variance)], dtype=float)
        if self.covariance_type == "diagonal":
            return cast(FloatArray, 0.5 * np.log(np.diag(regularized)))
        factor = np.linalg.cholesky(regularized)
        packed: list[float] = []
        for row in range(self.n_locations):
            for column in range(row + 1):
                value = float(factor[row, column])
                packed.append(np.log(value) if row == column else value)
        return np.asarray(packed, dtype=float)

    def unpack(self, parameters: Any) -> FloatArray:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.size,):
            raise ValueError("covariance parameter vector has an invalid shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("covariance parameters must be finite")
        if self.covariance_type == "scalar":
            standard_deviation = float(np.exp(values[0]))
            return np.eye(self.n_locations, dtype=float) * standard_deviation**2
        if self.covariance_type == "diagonal":
            standard_deviations = np.exp(values)
            return cast(FloatArray, np.diag(standard_deviations**2))
        factor = np.zeros((self.n_locations, self.n_locations), dtype=float)
        cursor = 0
        for row in range(self.n_locations):
            for column in range(row + 1):
                value = float(values[cursor])
                factor[row, column] = np.exp(value) if row == column else value
                cursor += 1
        return cast(FloatArray, factor @ factor.T)
