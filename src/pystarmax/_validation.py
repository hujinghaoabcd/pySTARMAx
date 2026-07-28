# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Internal validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def as_float_matrix(value: Any, *, name: str) -> FloatArray:
    """Return a finite two-dimensional float array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)


def validate_time_space(value: Any, *, name: str = "data") -> FloatArray:
    """Validate an observation matrix with shape ``(time, location)``."""
    array = as_float_matrix(value, name=name)
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two time observations")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one location")
    return array


def validate_nonnegative_int(value: int, *, name: str) -> int:
    """Validate a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
