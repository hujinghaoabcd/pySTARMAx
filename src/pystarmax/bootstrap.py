# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Bootstrap utilities for parameter-aware predictive inference."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)
from pystarmax.differencing import differencing_coefficients
from pystarmax.forecasting import draw_innovations, random_generator

BootstrapMethod = Literal["residual", "parametric"]


def validate_bootstrap_method(method: str) -> BootstrapMethod:
    """Normalize and validate a bootstrap innovation method."""
    normalized = str(method).strip().lower()
    if normalized not in {"residual", "parametric"}:
        raise ValueError("bootstrap_method must be 'residual' or 'parametric'")
    return cast(BootstrapMethod, normalized)


def validate_bootstrap_arguments(
    *,
    n_bootstrap: int,
    max_attempts: int | None,
    bootstrap_method: str,
) -> tuple[int, int, BootstrapMethod]:
    """Validate replication controls and return normalized values."""
    resolved_bootstrap = validate_nonnegative_int(n_bootstrap, name="n_bootstrap")
    if resolved_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")

    if max_attempts is None:
        resolved_attempts = max(
            resolved_bootstrap + 10,
            3 * resolved_bootstrap,
        )
    else:
        resolved_attempts = validate_nonnegative_int(max_attempts, name="max_attempts")
        if resolved_attempts < resolved_bootstrap:
            raise ValueError("max_attempts must be at least n_bootstrap")
    return (
        resolved_bootstrap,
        resolved_attempts,
        validate_bootstrap_method(bootstrap_method),
    )


def finite_centered_residuals(residuals: Any) -> FloatArray:
    """Return complete finite residual rows centered by location."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("residuals must have shape (time, locations)")
    complete = values[np.all(np.isfinite(values), axis=1)]
    if complete.shape[0] < 2:
        raise ValueError("residuals must contain at least two fully finite rows")
    centered = complete - complete.mean(axis=0, keepdims=True)
    return cast(
        FloatArray,
        np.ascontiguousarray(centered, dtype=float),
    )


def draw_bootstrap_innovations(
    *,
    bootstrap_method: str,
    residuals: Any,
    covariance: Any,
    n_simulations: int,
    steps: int,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    """Draw joint residual or Gaussian parametric innovation paths."""
    method = validate_bootstrap_method(bootstrap_method)
    simulations = validate_nonnegative_int(n_simulations, name="n_simulations")
    horizon = validate_nonnegative_int(steps, name="steps")
    if simulations == 0 or horizon == 0:
        raise ValueError("n_simulations and steps must be positive")

    if method == "parametric":
        return draw_innovations(
            covariance,
            n_simulations=simulations,
            steps=horizon,
            random_state=random_state,
        )

    centered = finite_centered_residuals(residuals)
    generator = random_generator(random_state)
    indices = generator.integers(
        0,
        centered.shape[0],
        size=(simulations, horizon),
    )
    draws = centered[indices]
    return cast(
        FloatArray,
        np.ascontiguousarray(draws, dtype=float),
    )


def restore_bootstrap_series(
    transformed: Any,
    observed: Any,
    *,
    ordinary_order: int = 0,
    seasonal_order: int = 0,
    seasonal_period: int = 1,
) -> FloatArray:
    """Recursively reconstruct a complete pseudo-series.

    Initial rows required by the combined differencing polynomial are copied
    from the observed sample. All later rows are reconstructed from the
    pseudo-history rather than repeatedly using observed lags.
    """
    observations = validate_time_space(observed)
    values = np.asarray(transformed, dtype=float)
    if values.ndim != 2 or values.shape[1] != observations.shape[1]:
        raise ValueError("transformed must have shape (time, observed locations)")
    if not np.all(np.isfinite(values)):
        raise ValueError("transformed must contain only finite values")

    coefficients = differencing_coefficients(
        ordinary_order=ordinary_order,
        seasonal_order=seasonal_order,
        seasonal_period=seasonal_period,
    )
    offset = int(coefficients.size - 1)
    if values.shape[0] != observations.shape[0] - offset:
        raise ValueError("transformed has an incompatible number of time rows")
    if offset == 0:
        return cast(
            FloatArray,
            np.ascontiguousarray(values, dtype=float),
        )

    restored = np.empty_like(observations, dtype=float)
    restored[:offset] = observations[:offset]
    for transformed_index, row in enumerate(values):
        time_index = transformed_index + offset
        current = np.asarray(row, dtype=float).copy()
        for lag, coefficient in enumerate(coefficients[1:], start=1):
            current -= float(coefficient) * restored[time_index - lag]
        restored[time_index] = current
    return cast(
        FloatArray,
        np.ascontiguousarray(restored, dtype=float),
    )
