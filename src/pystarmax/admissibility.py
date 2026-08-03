# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Stationarity and invertibility diagnostics for STARMA matrix polynomials."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from pystarmax._validation import FloatArray
from pystarmax.weights import SpatialWeights

ComplexArray: TypeAlias = npt.NDArray[np.complex128]
PolynomialKind = Literal["autoregressive", "moving_average"]


def _freeze_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=float).copy()
    frozen.setflags(write=False)
    return cast(FloatArray, frozen)


def _freeze_complex(value: Any, *, name: str, ndim: int) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.ascontiguousarray(array, dtype=np.complex128).copy()
    frozen.setflags(write=False)
    return cast(ComplexArray, frozen)


def _resolve_weights(weights: Any) -> SpatialWeights:
    if isinstance(weights, SpatialWeights):
        return weights
    if isinstance(weights, np.ndarray):
        return SpatialWeights.from_matrices([weights], prepend_identity=True)
    if isinstance(weights, Iterable):
        matrices = list(weights)
        if not matrices:
            raise ValueError("weights must not be empty")
        return SpatialWeights.from_matrices(matrices)
    raise TypeError("weights must be SpatialWeights, a matrix, or a matrix sequence")


def _parameter_rows(value: Any, *, width: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return cast(FloatArray, np.empty((0, width), dtype=float))
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (order, {width})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, np.ascontiguousarray(array, dtype=float))


def _validate_margin(margin: float) -> float:
    value = float(margin)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("margin must be between zero and one")
    return value


def compose_lag_operators(parameters: Any, weights: Any) -> FloatArray:
    """Compose temporal-lag matrices from coefficients and spatial weights.

    Parameters use shape ``(temporal order, number of spatial weights)``. The
    returned array has shape ``(temporal order, location, location)`` and retains
    the supplied matrix orientation exactly.
    """
    resolved = _resolve_weights(weights)
    rows = _parameter_rows(
        parameters,
        width=len(resolved),
        name="parameters",
    )
    operators = np.empty(
        (rows.shape[0], resolved.n_locations, resolved.n_locations),
        dtype=float,
    )
    for temporal_lag, row in enumerate(rows):
        operator = np.zeros(
            (resolved.n_locations, resolved.n_locations),
            dtype=float,
        )
        for coefficient, matrix in zip(row, resolved, strict=True):
            operator += coefficient * matrix
        operators[temporal_lag] = operator
    return cast(FloatArray, np.ascontiguousarray(operators, dtype=float))


def _companion_matrix(operators: FloatArray, *, sign: float) -> FloatArray:
    if operators.ndim != 3:
        raise ValueError("operators must be three-dimensional")
    order, n_locations, width = operators.shape
    if n_locations != width:
        raise ValueError("operators must contain square lag matrices")
    if order == 0:
        return cast(FloatArray, np.empty((0, 0), dtype=float))
    companion = np.zeros(
        (order * n_locations, order * n_locations),
        dtype=float,
    )
    companion[:n_locations, :] = sign * np.concatenate(
        [operators[lag] for lag in range(order)],
        axis=1,
    )
    identity = np.eye(n_locations, dtype=float)
    for lag in range(1, order):
        row = lag * n_locations
        column = (lag - 1) * n_locations
        companion[row : row + n_locations, column : column + n_locations] = identity
    return cast(FloatArray, companion)


@dataclass(frozen=True, slots=True)
class PolynomialAdmissibility:
    """Spectral diagnostic for one AR or inverse-MA matrix polynomial."""

    kind: PolynomialKind
    operators: FloatArray
    companion_matrix: FloatArray
    eigenvalues: ComplexArray
    spectral_radius: float
    limit: float
    distance: float
    admissible: bool

    def __post_init__(self) -> None:
        if self.kind not in {"autoregressive", "moving_average"}:
            raise ValueError("kind must be 'autoregressive' or 'moving_average'")
        operators = _freeze_float(self.operators, name="operators", ndim=3)
        if operators.shape[1] != operators.shape[2]:
            raise ValueError("operators must contain square lag matrices")
        order, n_locations, _width = operators.shape
        companion = _freeze_float(
            self.companion_matrix,
            name="companion_matrix",
            ndim=2,
        )
        expected_size = order * n_locations
        if companion.shape != (expected_size, expected_size):
            raise ValueError("companion_matrix has an invalid shape")
        eigenvalues = _freeze_complex(self.eigenvalues, name="eigenvalues", ndim=1)
        if eigenvalues.shape != (expected_size,):
            raise ValueError("eigenvalues must match the companion dimension")
        radius = float(self.spectral_radius)
        limit = float(self.limit)
        distance = float(self.distance)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("spectral_radius must be non-negative and finite")
        if not np.isfinite(limit) or not 0.0 < limit < 1.0:
            raise ValueError("limit must be between zero and one")
        if not np.isfinite(distance):
            raise ValueError("distance must be finite")
        expected_radius = (
            float(np.max(np.abs(eigenvalues), initial=0.0)) if eigenvalues.size else 0.0
        )
        if not np.isclose(radius, expected_radius, rtol=1e-10, atol=1e-12):
            raise ValueError("spectral_radius must match eigenvalues")
        if not np.isclose(distance, limit - radius, rtol=1e-10, atol=1e-12):
            raise ValueError("distance must equal limit minus spectral_radius")
        if bool(self.admissible) != bool(radius < limit):
            raise ValueError("admissible must match the spectral-radius criterion")
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "companion_matrix", companion)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "spectral_radius", radius)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "admissible", bool(self.admissible))

    @property
    def order(self) -> int:
        """Temporal polynomial order."""
        return int(self.operators.shape[0])

    @property
    def n_locations(self) -> int:
        """Number of locations represented by each lag matrix."""
        return int(self.operators.shape[1])

    @property
    def margin(self) -> float:
        """Required distance below the unit spectral-radius boundary."""
        return float(1.0 - self.limit)

    def summary(self) -> str:
        """Return a compact text summary."""
        label = (
            "AR stationarity" if self.kind == "autoregressive" else "MA invertibility"
        )
        return "\n".join(
            [
                f"pySTARMAx {label} diagnostic",
                "=" * 56,
                f"Order: {self.order}",
                f"Locations: {self.n_locations}",
                f"Spectral radius: {self.spectral_radius:.6f}",
                f"Required limit: {self.limit:.6f}",
                f"Boundary distance: {self.distance:.6f}",
                f"Admissible: {self.admissible}",
            ]
        )


@dataclass(frozen=True, slots=True)
class STARMAAdmissibility:
    """Joint AR stationarity and MA invertibility diagnostics."""

    autoregressive: PolynomialAdmissibility
    moving_average: PolynomialAdmissibility

    def __post_init__(self) -> None:
        if self.autoregressive.kind != "autoregressive":
            raise ValueError("autoregressive must contain an AR diagnostic")
        if self.moving_average.kind != "moving_average":
            raise ValueError("moving_average must contain an MA diagnostic")
        if self.autoregressive.n_locations != self.moving_average.n_locations:
            raise ValueError("AR and MA diagnostics must use the same locations")

    @property
    def stationary(self) -> bool:
        """Whether the AR matrix polynomial meets its spectral limit."""
        return self.autoregressive.admissible

    @property
    def invertible(self) -> bool:
        """Whether the inverse-MA recursion meets its spectral limit."""
        return self.moving_average.admissible

    @property
    def admissible(self) -> bool:
        """Whether both stationarity and invertibility criteria hold."""
        return self.stationary and self.invertible

    @property
    def minimum_distance(self) -> float:
        """Smallest signed distance to either admissibility boundary."""
        return float(
            min(
                self.autoregressive.distance,
                self.moving_average.distance,
            )
        )

    def summary(self) -> str:
        """Return a compact joint summary."""
        return "\n".join(
            [
                "pySTARMAx STARMA admissibility diagnostic",
                "=" * 64,
                f"AR spectral radius: {self.autoregressive.spectral_radius:.6f}",
                f"AR boundary distance: {self.autoregressive.distance:.6f}",
                f"Stationary: {self.stationary}",
                f"Inverse-MA spectral radius: "
                f"{self.moving_average.spectral_radius:.6f}",
                f"MA boundary distance: {self.moving_average.distance:.6f}",
                f"Invertible: {self.invertible}",
                f"Jointly admissible: {self.admissible}",
            ]
        )


def _diagnose(
    parameters: Any,
    weights: Any,
    *,
    kind: PolynomialKind,
    margin: float,
) -> PolynomialAdmissibility:
    resolved_margin = _validate_margin(margin)
    operators = compose_lag_operators(parameters, weights)
    sign = 1.0 if kind == "autoregressive" else -1.0
    companion = _companion_matrix(operators, sign=sign)
    eigenvalues = (
        np.asarray(np.linalg.eigvals(companion), dtype=np.complex128)
        if companion.size
        else np.empty(0, dtype=np.complex128)
    )
    radius = float(np.max(np.abs(eigenvalues), initial=0.0))
    limit = 1.0 - resolved_margin
    return PolynomialAdmissibility(
        kind=kind,
        operators=operators,
        companion_matrix=companion,
        eigenvalues=eigenvalues,
        spectral_radius=radius,
        limit=limit,
        distance=limit - radius,
        admissible=radius < limit,
    )


def autoregressive_diagnostics(
    ar_parameters: Any,
    weights: Any,
    *,
    margin: float = 1e-6,
) -> PolynomialAdmissibility:
    """Evaluate STARMA autoregressive stationarity by companion radius."""
    return _diagnose(
        ar_parameters,
        weights,
        kind="autoregressive",
        margin=margin,
    )


def moving_average_diagnostics(
    ma_parameters: Any,
    weights: Any,
    *,
    margin: float = 1e-6,
) -> PolynomialAdmissibility:
    """Evaluate STARMA moving-average invertibility.

    pySTARMAx uses ``epsilon_t + sum(B_k epsilon_(t-k))``. The inverse recursion
    therefore has companion top row ``[-B_1, ..., -B_q]``. Invertibility requires
    the spectral radius of that inverse-recursion companion to be below one.
    """
    return _diagnose(
        ma_parameters,
        weights,
        kind="moving_average",
        margin=margin,
    )


def starma_admissibility(
    ar_parameters: Any,
    ma_parameters: Any,
    weights: Any,
    *,
    stability_margin: float = 1e-6,
    invertibility_margin: float = 1e-6,
) -> STARMAAdmissibility:
    """Return joint AR stationarity and MA invertibility diagnostics."""
    return STARMAAdmissibility(
        autoregressive=autoregressive_diagnostics(
            ar_parameters,
            weights,
            margin=stability_margin,
        ),
        moving_average=moving_average_diagnostics(
            ma_parameters,
            weights,
            margin=invertibility_margin,
        ),
    )


def autoregressive_spectral_radius(ar_parameters: Any, weights: Any) -> float:
    """Return the AR companion spectral radius without applying a margin."""
    return autoregressive_diagnostics(
        ar_parameters,
        weights,
        margin=np.finfo(float).eps,
    ).spectral_radius


def moving_average_inverse_spectral_radius(
    ma_parameters: Any,
    weights: Any,
) -> float:
    """Return the inverse-MA companion spectral radius without a margin."""
    return moving_average_diagnostics(
        ma_parameters,
        weights,
        margin=np.finfo(float).eps,
    ).spectral_radius


__all__ = [
    "PolynomialAdmissibility",
    "PolynomialKind",
    "STARMAAdmissibility",
    "autoregressive_diagnostics",
    "autoregressive_spectral_radius",
    "compose_lag_operators",
    "moving_average_diagnostics",
    "moving_average_inverse_spectral_radius",
    "starma_admissibility",
]
