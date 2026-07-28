# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Forecast distributions and conditional innovation intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import FloatArray, validate_nonnegative_int


@dataclass(frozen=True, slots=True)
class ForecastInterval:
    """Immutable point forecast and simulation-based interval.

    Parameters
    ----------
    mean, lower, upper:
        Arrays with shape ``(steps, locations)``.
    level:
        Central interval probability in ``(0, 1)``.
    n_simulations:
        Number of conditional innovation paths used for the quantiles.
    method:
        Human-readable description of the uncertainty calculation.
    """

    mean: FloatArray
    lower: FloatArray
    upper: FloatArray
    level: float
    n_simulations: int
    method: str = "conditional innovation simulation"

    def __post_init__(self) -> None:
        arrays: list[FloatArray] = []
        reference_shape: tuple[int, int] | None = None
        for name, value in (
            ("mean", self.mean),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            array = np.asarray(value, dtype=float)
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
                raise ValueError(f"{name} must have shape (steps, locations)")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            if reference_shape is None:
                reference_shape = array.shape
            elif array.shape != reference_shape:
                raise ValueError("mean, lower, and upper must share one shape")
            frozen = np.ascontiguousarray(array, dtype=float).copy()
            frozen.setflags(write=False)
            arrays.append(cast(FloatArray, frozen))

        level = float(self.level)
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between zero and one")
        n_simulations = validate_nonnegative_int(
            self.n_simulations, name="n_simulations"
        )
        if n_simulations < 2:
            raise ValueError("n_simulations must be at least two")
        if np.any(arrays[1] > arrays[2]):
            raise ValueError("lower must not exceed upper")

        object.__setattr__(self, "mean", arrays[0])
        object.__setattr__(self, "lower", arrays[1])
        object.__setattr__(self, "upper", arrays[2])
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "n_simulations", n_simulations)
        object.__setattr__(self, "method", str(self.method))

    @property
    def shape(self) -> tuple[int, int]:
        """Forecast array shape ``(steps, locations)``."""
        return self.mean.shape


def validate_interval_arguments(
    *,
    steps: int,
    level: float,
    n_simulations: int,
) -> tuple[int, float, int]:
    """Validate common interval arguments and return normalized values."""
    resolved_steps = validate_nonnegative_int(steps, name="steps")
    if resolved_steps == 0:
        raise ValueError("steps must be positive")
    resolved_level = float(level)
    if not 0.0 < resolved_level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    resolved_simulations = validate_nonnegative_int(n_simulations, name="n_simulations")
    if resolved_simulations < 2:
        raise ValueError("n_simulations must be at least two")
    return resolved_steps, resolved_level, resolved_simulations


def random_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Resolve an integer seed, generator, or ``None`` to one generator."""
    return (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )


def draw_innovations(
    covariance: Any,
    *,
    n_simulations: int,
    steps: int,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    """Draw stable Gaussian innovation paths from a fitted covariance matrix.

    The covariance is symmetrized and tiny negative eigenvalues caused by
    floating-point estimation are clipped to zero. This permits singular fitted
    covariance matrices while retaining their estimated location dependence.
    """
    values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("covariance must contain only finite values")
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -100.0 * tolerance:
        raise ValueError("covariance must be positive semidefinite")
    clipped = np.clip(eigenvalues, 0.0, np.inf)
    factor = eigenvectors @ np.diag(np.sqrt(clipped))
    standard = random_generator(random_state).standard_normal(
        (n_simulations, steps, values.shape[0])
    )
    draws = standard @ factor.T
    return cast(FloatArray, np.ascontiguousarray(draws, dtype=float))


def interval_from_paths(
    *,
    mean: Any,
    paths: Any,
    level: float,
    method: str = "conditional innovation simulation",
) -> ForecastInterval:
    """Construct a central empirical interval from forecast paths."""
    mean_values = np.asarray(mean, dtype=float)
    path_values = np.asarray(paths, dtype=float)
    if path_values.ndim != 3:
        raise ValueError("paths must have shape (simulations, steps, locations)")
    if mean_values.shape != path_values.shape[1:]:
        raise ValueError("mean shape must match the path step/location dimensions")
    if not np.all(np.isfinite(path_values)):
        raise ValueError("paths must contain only finite values")
    alpha = 0.5 * (1.0 - float(level))
    lower = np.quantile(path_values, alpha, axis=0)
    upper = np.quantile(path_values, 1.0 - alpha, axis=0)
    return ForecastInterval(
        mean=np.asarray(mean_values, dtype=float),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        level=float(level),
        n_simulations=int(path_values.shape[0]),
        method=method,
    )
