# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Auditable multiplicative seasonal matrix-polynomial expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from pystarmax._validation import FloatArray, as_float_matrix, validate_nonnegative_int
from pystarmax.weights import SpatialWeights


@dataclass(frozen=True, slots=True)
class LagOperator:
    """One temporal lag and its fixed location-to-location operator matrix."""

    lag: int
    matrix: FloatArray
    label: str

    def __post_init__(self) -> None:
        lag = validate_nonnegative_int(self.lag, name="lag")
        if lag == 0:
            raise ValueError("lag must be positive")
        values = as_float_matrix(self.matrix, name="matrix")
        if values.shape[0] != values.shape[1]:
            raise ValueError("matrix must be square")
        frozen = values.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "lag", lag)
        object.__setattr__(self, "matrix", cast(FloatArray, frozen))
        object.__setattr__(self, "label", str(self.label))


def _coefficient_array(
    value: Any,
    *,
    name: str,
    n_spatial: int,
) -> FloatArray:
    array = as_float_matrix(value, name=name)
    if array.shape[1] != n_spatial:
        raise ValueError(
            f"{name} has {array.shape[1]} spatial columns; expected {n_spatial}"
        )
    return array


def factor_matrices(
    coefficients: Any,
    weights: SpatialWeights,
    *,
    name: str = "coefficients",
) -> tuple[FloatArray, ...]:
    """Convert coefficient rows into matrix-valued temporal factors."""
    values = _coefficient_array(
        coefficients,
        name=name,
        n_spatial=len(weights),
    )
    matrices: list[FloatArray] = []
    for row in values:
        matrix = np.zeros(
            (weights.n_locations, weights.n_locations), dtype=float
        )
        for coefficient, spatial_weight in zip(row, weights, strict=True):
            matrix += float(coefficient) * spatial_weight
        matrices.append(cast(FloatArray, matrix))
    return tuple(matrices)


def expand_multiplicative_operators(
    nonseasonal_coefficients: Any,
    seasonal_coefficients: Any,
    weights: SpatialWeights,
    *,
    seasonal_period: int,
    kind: Literal["ar", "ma"],
) -> tuple[LagOperator, ...]:
    """Expand factorized ordinary and seasonal matrix polynomials.

    The convention is seasonal factor multiplied on the left of the ordinary
    factor. For autoregression this is

    ``(I - S(B**s)) (I - A(B))``

    and the prediction equation therefore contains ``+A``, ``+S``, and
    ``-S@A`` terms. For moving averages the package sign convention gives
    ``(I + N(B**s)) (I + M(B))`` and all three term groups are positive.

    Matrix cross-products are applied directly. They are never projected back
    onto the supplied spatial-weight basis, which need not be closed under
    multiplication.
    """
    period = validate_nonnegative_int(seasonal_period, name="seasonal_period")
    if period == 0:
        raise ValueError("seasonal_period must be positive")
    if kind not in {"ar", "ma"}:
        raise ValueError("kind must be 'ar' or 'ma'")

    ordinary = factor_matrices(
        nonseasonal_coefficients,
        weights,
        name="nonseasonal_coefficients",
    )
    seasonal = factor_matrices(
        seasonal_coefficients,
        weights,
        name="seasonal_coefficients",
    )
    terms: list[LagOperator] = []
    prefix = "ar" if kind == "ar" else "ma"
    for index, matrix in enumerate(ordinary, start=1):
        terms.append(
            LagOperator(
                lag=index,
                matrix=matrix,
                label=f"{prefix}.ordinary.t{index}",
            )
        )
    for index, matrix in enumerate(seasonal, start=1):
        lag = index * period
        terms.append(
            LagOperator(
                lag=lag,
                matrix=matrix,
                label=f"{prefix}.seasonal.t{lag}",
            )
        )
    cross_sign = -1.0 if kind == "ar" else 1.0
    for seasonal_index, seasonal_matrix in enumerate(seasonal, start=1):
        for ordinary_index, ordinary_matrix in enumerate(ordinary, start=1):
            lag = seasonal_index * period + ordinary_index
            terms.append(
                LagOperator(
                    lag=lag,
                    matrix=cross_sign * (seasonal_matrix @ ordinary_matrix),
                    label=(
                        f"{prefix}.cross.t{lag}."
                        f"s{seasonal_index * period}.o{ordinary_index}"
                    ),
                )
            )
    return tuple(terms)


def maximum_operator_lag(operators: tuple[LagOperator, ...]) -> int:
    """Largest lag in an expanded operator collection, or zero when empty."""
    return max((operator.lag for operator in operators), default=0)
