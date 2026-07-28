# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Ordinary and seasonal temporal differencing with forecast inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)


def _validate_positive_int(value: int, *, name: str) -> int:
    result = validate_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_forecast(
    value: Any,
    *,
    n_locations: int,
    name: str,
) -> FloatArray:
    values = np.asarray(value, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if values.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if values.shape[1] != n_locations:
        raise ValueError(
            f"{name} contains {values.shape[1]} locations; expected {n_locations}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, np.ascontiguousarray(values, dtype=float))


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
        n_locations = _validate_positive_int(
            self.n_locations, name="n_locations"
        )
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
        """Reconstruct future observations on the original data scale."""
        values = _validate_forecast(
            differenced_forecast,
            n_locations=self.n_locations,
            name="differenced_forecast",
        )
        if self.order == 0:
            return cast(FloatArray, values.copy())

        state = [anchor.copy() for anchor in self.anchors]
        restored = np.empty_like(values, dtype=float)
        for time_index, row in enumerate(values):
            increment = np.asarray(row, dtype=float).copy()
            for level in range(self.order - 1, -1, -1):
                state[level] = state[level] + increment
                increment = state[level]
            restored[time_index] = increment
        return cast(FloatArray, np.ascontiguousarray(restored, dtype=float))


@dataclass(frozen=True, slots=True)
class SeasonalDifferencingState:
    """Immutable state for inverting ``(1 - B**period)^order`` forecasts.

    ``histories[level]`` stores the final ``period`` rows at seasonal difference
    level ``level``. During inversion each new value is added to the value one
    full seasonal period earlier and appended to a private rolling history.
    """

    order: int
    period: int
    n_locations: int
    histories: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        order = validate_nonnegative_int(self.order, name="order")
        period = _validate_positive_int(self.period, name="period")
        n_locations = _validate_positive_int(
            self.n_locations, name="n_locations"
        )
        if len(self.histories) != order:
            raise ValueError("histories must contain one matrix per seasonal level")

        checked: list[FloatArray] = []
        expected_shape = (period, n_locations)
        for index, history in enumerate(self.histories):
            values = np.asarray(history, dtype=float)
            if values.shape != expected_shape:
                raise ValueError(
                    f"histories[{index}] must have shape {expected_shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"histories[{index}] must contain only finite values"
                )
            frozen = np.asarray(values, dtype=float).copy()
            frozen.setflags(write=False)
            checked.append(cast(FloatArray, frozen))

        object.__setattr__(self, "order", order)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "n_locations", n_locations)
        object.__setattr__(self, "histories", tuple(checked))

    def inverse_forecast(self, differenced_forecast: Any) -> FloatArray:
        """Reconstruct forecasts before seasonal differencing."""
        values = _validate_forecast(
            differenced_forecast,
            n_locations=self.n_locations,
            name="differenced_forecast",
        )
        if self.order == 0:
            return cast(FloatArray, values.copy())

        queues = [history.copy() for history in self.histories]
        restored = np.empty_like(values, dtype=float)
        for time_index, row in enumerate(values):
            increment = np.asarray(row, dtype=float).copy()
            for level in range(self.order - 1, -1, -1):
                value = queues[level][0] + increment
                queues[level][:-1] = queues[level][1:]
                queues[level][-1] = value
                increment = value
            restored[time_index] = increment
        return cast(FloatArray, np.ascontiguousarray(restored, dtype=float))


@dataclass(frozen=True, slots=True)
class CombinedDifferencingState:
    """State for ordinary and seasonal differencing applied in sequence."""

    ordinary: DifferencingState
    seasonal: SeasonalDifferencingState

    def __post_init__(self) -> None:
        if self.ordinary.n_locations != self.seasonal.n_locations:
            raise ValueError("ordinary and seasonal states must share locations")

    @property
    def n_locations(self) -> int:
        """Number of locations represented by both component states."""
        return self.ordinary.n_locations

    @property
    def ordinary_order(self) -> int:
        """Ordinary differencing order ``d``."""
        return self.ordinary.order

    @property
    def seasonal_order(self) -> int:
        """Seasonal differencing order ``D``."""
        return self.seasonal.order

    @property
    def seasonal_period(self) -> int:
        """Seasonal period ``s``."""
        return self.seasonal.period

    def inverse_forecast(self, differenced_forecast: Any) -> FloatArray:
        """Undo seasonal differencing first, then ordinary differencing."""
        ordinary_scale = self.seasonal.inverse_forecast(differenced_forecast)
        return self.ordinary.inverse_forecast(ordinary_scale)


def ordinary_difference(
    data: Any,
    *,
    order: int = 1,
) -> tuple[FloatArray, DifferencingState]:
    """Apply ``(1 - B)^order`` along time and capture forecast anchors."""
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


def seasonal_difference(
    data: Any,
    *,
    order: int = 1,
    period: int,
) -> tuple[FloatArray, SeasonalDifferencingState]:
    """Apply ``(1 - B**period)^order`` along time.

    Each seasonal difference reduces the available sample by ``period`` rows.
    The returned state retains the final seasonal cycle at every lower-order
    level so that future forecasts can be reconstructed exactly.
    """
    observations = validate_time_space(data)
    order = validate_nonnegative_int(order, name="order")
    period = _validate_positive_int(period, name="period")
    if order * period >= observations.shape[0]:
        raise ValueError(
            "seasonal order times period must be smaller than the number of time rows"
        )

    levels: list[FloatArray] = [observations.copy()]
    for _ in range(order):
        previous = levels[-1]
        levels.append(
            cast(FloatArray, previous[period:] - previous[:-period])
        )
    state = SeasonalDifferencingState(
        order=order,
        period=period,
        n_locations=observations.shape[1],
        histories=tuple(level[-period:] for level in levels[:-1]),
    )
    differenced = np.ascontiguousarray(levels[-1], dtype=float)
    return cast(FloatArray, differenced), state


def combined_difference(
    data: Any,
    *,
    ordinary_order: int = 0,
    seasonal_order: int = 0,
    seasonal_period: int = 1,
) -> tuple[FloatArray, CombinedDifferencingState]:
    """Apply ordinary then seasonal differencing and retain both states.

    The scalar temporal operators commute, but fixing this execution order makes
    state capture and inverse reconstruction deterministic. Inversion applies
    the component states in reverse order.
    """
    observations = validate_time_space(data)
    ordinary, ordinary_state = ordinary_difference(
        observations, order=ordinary_order
    )
    transformed, seasonal_state = seasonal_difference(
        ordinary,
        order=seasonal_order,
        period=seasonal_period,
    )
    return transformed, CombinedDifferencingState(
        ordinary=ordinary_state,
        seasonal=seasonal_state,
    )
