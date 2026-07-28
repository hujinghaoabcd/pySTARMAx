# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Ordinary temporal differencing and forecast inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)


@dataclass(frozen=True, slots=True)
class DifferencingState:
    """Immutable end-of-sample state for ordinary forecast integration.

    ``anchors[level]`` stores the final observed value at difference level
    ``level``. Level zero is the original series, level one is its first
    difference, and so on through ``order - 1``.
    """

    order: int
    n_locations: int
    anchors: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        order = validate_nonnegative_int(self.order, name="order")
        n_locations = validate_nonnegative_int(self.n_locations, name="n_locations")
        if n_locations == 0:
            raise ValueError("n_locations must be positive")
        if len(self.anchors) != order:
            raise ValueError("anchors must contain one vector per difference level")

        checked: list[FloatArray] = []
        for index, anchor in enumerate(self.anchors):
            values = np.asarray(anchor, dtype=float)
            if values.ndim != 1 or values.shape[0] != n_locations:
                raise ValueError(f"anchors[{index}] must have shape ({n_locations},)")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"anchors[{index}] must contain only finite values")
            frozen = np.asarray(values, dtype=float).copy()
            frozen.setflags(write=False)
            checked.append(cast(FloatArray, frozen))

        object.__setattr__(self, "order", order)
        object.__setattr__(self, "n_locations", n_locations)
        object.__setattr__(self, "anchors", tuple(checked))

    def inverse_forecast(self, differenced_forecast: Any) -> FloatArray:
        """Reconstruct future observations on the original data scale.

        The supplied rows are forecasts at difference order ``self.order``.
        The stored end-of-sample anchors are copied, so repeated calls are
        deterministic and do not mutate the state.
        """
        values = np.asarray(differenced_forecast, dtype=float)
        if values.ndim != 2:
            raise ValueError("differenced_forecast must be a two-dimensional array")
        if values.shape[0] < 1:
            raise ValueError("differenced_forecast must contain at least one row")
        if values.shape[1] != self.n_locations:
            raise ValueError(
                "differenced_forecast contains "
                f"{values.shape[1]} locations; expected {self.n_locations}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("differenced_forecast must contain only finite values")
        if self.order == 0:
            return cast(FloatArray, np.ascontiguousarray(values, dtype=float).copy())

        state = [anchor.copy() for anchor in self.anchors]
        restored = np.empty_like(values, dtype=float)
        for time_index, row in enumerate(values):
            increment = np.asarray(row, dtype=float).copy()
            for level in range(self.order - 1, -1, -1):
                state[level] = state[level] + increment
                increment = state[level]
            restored[time_index] = increment
        return cast(FloatArray, np.ascontiguousarray(restored, dtype=float))


def ordinary_difference(
    data: Any,
    *,
    order: int = 1,
) -> tuple[FloatArray, DifferencingState]:
    """Apply ``(1 - B)^order`` along time and capture forecast anchors.

    Parameters
    ----------
    data:
        Finite observation matrix with shape ``(time, location)``.
    order:
        Non-negative ordinary integration order.

    Returns
    -------
    differenced, state:
        The differenced observations and an immutable end-of-sample state that
        can invert future forecasts back to the original scale.
    """
    observations = validate_time_space(data)
    order = validate_nonnegative_int(order, name="order")
    if order >= observations.shape[0]:
        raise ValueError("order must be smaller than the number of time rows")

    levels: list[FloatArray] = [observations.copy()]
    for _ in range(order):
        levels.append(cast(FloatArray, np.diff(levels[-1], axis=0)))
    state = DifferencingState(
        order=order,
        n_locations=observations.shape[1],
        anchors=tuple(level[-1] for level in levels[:-1]),
    )
    differenced = np.ascontiguousarray(levels[-1], dtype=float)
    return cast(FloatArray, differenced), state
