# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Rolling-origin evaluation for point and interval forecasts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from pystarmax._validation import (
    FloatArray,
    validate_nonnegative_int,
    validate_time_space,
)
from pystarmax.forecasting import ForecastInterval, random_generator

IntervalMethod = Literal["conditional", "bootstrap"]


def _positive_int(value: int, *, name: str) -> int:
    resolved = validate_nonnegative_int(value, name=name)
    if resolved == 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _validate_level(level: float) -> float:
    resolved = float(level)
    if not 0.0 < resolved < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    return resolved


def interval_score(
    observed: Any,
    lower: Any,
    upper: Any,
    *,
    level: float = 0.95,
) -> FloatArray:
    """Return the central Winkler interval score elementwise.

    Lower values are better. The score equals interval width for observations
    inside the interval and adds a level-dependent penalty for misses.
    """
    actual = np.asarray(observed, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if actual.shape != lower_values.shape or actual.shape != upper_values.shape:
        raise ValueError("observed, lower, and upper must share one shape")
    if actual.size == 0:
        raise ValueError("interval arrays must not be empty")
    if not (
        np.all(np.isfinite(actual))
        and np.all(np.isfinite(lower_values))
        and np.all(np.isfinite(upper_values))
    ):
        raise ValueError("interval arrays must contain only finite values")
    if np.any(lower_values > upper_values):
        raise ValueError("lower must not exceed upper")

    alpha = 1.0 - _validate_level(level)
    score = upper_values - lower_values
    below = actual < lower_values
    above = actual > upper_values
    score = score + (2.0 / alpha) * (lower_values - actual) * below
    score = score + (2.0 / alpha) * (actual - upper_values) * above
    return cast(FloatArray, np.ascontiguousarray(score, dtype=float))


@dataclass(frozen=True, slots=True)
class IntervalMetrics:
    """Aggregate point and interval forecast metrics."""

    nominal_coverage: float
    empirical_coverage: float
    coverage_gap: float
    average_width: float
    mean_interval_score: float
    mae: float
    rmse: float
    n_forecasts: int

    def __post_init__(self) -> None:
        nominal = _validate_level(self.nominal_coverage)
        empirical = float(self.empirical_coverage)
        if not 0.0 <= empirical <= 1.0:
            raise ValueError("empirical_coverage must lie between zero and one")
        n_forecasts = _positive_int(self.n_forecasts, name="n_forecasts")
        finite_values = (
            self.coverage_gap,
            self.average_width,
            self.mean_interval_score,
            self.mae,
            self.rmse,
        )
        if not np.all(np.isfinite(np.asarray(finite_values, dtype=float))):
            raise ValueError("metric values must be finite")
        if (
            min(
                self.average_width,
                self.mean_interval_score,
                self.mae,
                self.rmse,
            )
            < 0.0
        ):
            raise ValueError("width, score, MAE, and RMSE must be non-negative")
        object.__setattr__(self, "nominal_coverage", nominal)
        object.__setattr__(self, "empirical_coverage", empirical)
        object.__setattr__(self, "coverage_gap", float(self.coverage_gap))
        object.__setattr__(self, "average_width", float(self.average_width))
        object.__setattr__(self, "mean_interval_score", float(self.mean_interval_score))
        object.__setattr__(self, "mae", float(self.mae))
        object.__setattr__(self, "rmse", float(self.rmse))
        object.__setattr__(self, "n_forecasts", n_forecasts)

    @property
    def absolute_coverage_error(self) -> float:
        """Absolute difference between empirical and nominal coverage."""
        return abs(self.coverage_gap)


@dataclass(frozen=True, slots=True)
class RollingOriginResult:
    """Immutable rolling-origin point and interval forecast evaluation."""

    origins: np.ndarray[Any, np.dtype[np.integer[Any]]]
    observed: FloatArray
    mean: FloatArray
    lower: FloatArray
    upper: FloatArray
    level: float
    interval_method: str

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=int)
        if origins.ndim != 1 or origins.size == 0:
            raise ValueError("origins must be a non-empty one-dimensional array")
        if np.any(origins < 1) or np.any(np.diff(origins) <= 0):
            raise ValueError("origins must be positive and strictly increasing")

        arrays: list[FloatArray] = []
        expected_shape: tuple[int, int, int] | None = None
        for name, value in (
            ("observed", self.observed),
            ("mean", self.mean),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            array = np.asarray(value, dtype=float)
            if array.ndim != 3 or min(array.shape) < 1:
                raise ValueError(
                    f"{name} must have shape (origins, horizon, locations)"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            if expected_shape is None:
                expected_shape = (
                    int(array.shape[0]),
                    int(array.shape[1]),
                    int(array.shape[2]),
                )
            elif array.shape != expected_shape:
                raise ValueError("all forecast arrays must share one shape")
            frozen = np.ascontiguousarray(array, dtype=float).copy()
            frozen.setflags(write=False)
            arrays.append(cast(FloatArray, frozen))

        if expected_shape is None or expected_shape[0] != origins.size:
            raise ValueError("the first forecast dimension must match origins")
        if np.any(arrays[2] > arrays[3]):
            raise ValueError("lower must not exceed upper")
        frozen_origins = np.asarray(origins, dtype=int).copy()
        frozen_origins.setflags(write=False)

        object.__setattr__(self, "origins", frozen_origins)
        object.__setattr__(self, "observed", arrays[0])
        object.__setattr__(self, "mean", arrays[1])
        object.__setattr__(self, "lower", arrays[2])
        object.__setattr__(self, "upper", arrays[3])
        object.__setattr__(self, "level", _validate_level(self.level))
        object.__setattr__(self, "interval_method", str(self.interval_method))

    @property
    def shape(self) -> tuple[int, int, int]:
        """Result shape ``(origins, horizon, locations)``."""
        return (
            int(self.observed.shape[0]),
            int(self.observed.shape[1]),
            int(self.observed.shape[2]),
        )

    @property
    def covered(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Boolean indicator of interval coverage for every forecast."""
        return (self.observed >= self.lower) & (self.observed <= self.upper)

    @property
    def widths(self) -> FloatArray:
        """Interval widths for every forecast."""
        return cast(FloatArray, self.upper - self.lower)

    @property
    def scores(self) -> FloatArray:
        """Elementwise central interval scores."""
        return interval_score(
            self.observed,
            self.lower,
            self.upper,
            level=self.level,
        )

    def metrics(self) -> IntervalMetrics:
        """Aggregate metrics across origins, horizons, and locations."""
        return _aggregate_metrics(
            observed=self.observed,
            mean=self.mean,
            lower=self.lower,
            upper=self.upper,
            level=self.level,
        )

    def metrics_by_horizon(self) -> tuple[IntervalMetrics, ...]:
        """Return aggregate metrics separately for each forecast horizon."""
        return tuple(
            _aggregate_metrics(
                observed=self.observed[:, horizon_index, :],
                mean=self.mean[:, horizon_index, :],
                lower=self.lower[:, horizon_index, :],
                upper=self.upper[:, horizon_index, :],
                level=self.level,
            )
            for horizon_index in range(self.shape[1])
        )


def _aggregate_metrics(
    *,
    observed: Any,
    mean: Any,
    lower: Any,
    upper: Any,
    level: float,
) -> IntervalMetrics:
    actual = np.asarray(observed, dtype=float)
    center = np.asarray(mean, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    scores = interval_score(actual, lower_values, upper_values, level=level)
    covered = (actual >= lower_values) & (actual <= upper_values)
    errors = center - actual
    empirical = float(np.mean(covered))
    nominal = _validate_level(level)
    return IntervalMetrics(
        nominal_coverage=nominal,
        empirical_coverage=empirical,
        coverage_gap=empirical - nominal,
        average_width=float(np.mean(upper_values - lower_values)),
        mean_interval_score=float(np.mean(scores)),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        n_forecasts=int(actual.size),
    )


def rolling_origin_evaluate(
    model_factory: Callable[[], Any],
    data: Any,
    weights: Any,
    *,
    initial_window: int,
    horizon: int = 1,
    step: int = 1,
    window_size: int | None = None,
    interval_method: IntervalMethod = "conditional",
    level: float = 0.95,
    interval_kwargs: Mapping[str, Any] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> RollingOriginResult:
    """Evaluate prediction intervals over expanding or rolling origins.

    ``initial_window`` is the first forecast origin. By default every model is
    fitted to all observations before that origin. Supplying ``window_size``
    instead uses only the most recent rows at every origin.
    """
    observations = validate_time_space(data)
    initial_window = _positive_int(initial_window, name="initial_window")
    horizon = _positive_int(horizon, name="horizon")
    step = _positive_int(step, name="step")
    if initial_window >= observations.shape[0]:
        raise ValueError("initial_window must be smaller than the number of rows")
    if initial_window + horizon > observations.shape[0]:
        raise ValueError("initial_window leaves no complete forecast horizon")
    resolved_window: int | None = None
    if window_size is not None:
        resolved_window = _positive_int(window_size, name="window_size")
    if interval_method not in {"conditional", "bootstrap"}:
        raise ValueError("interval_method must be 'conditional' or 'bootstrap'")
    level = _validate_level(level)

    kwargs = dict(interval_kwargs or {})
    reserved = {"steps", "level", "random_state"}.intersection(kwargs)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"interval_kwargs must not override: {names}")

    origins = np.arange(
        initial_window,
        observations.shape[0] - horizon + 1,
        step,
        dtype=int,
    )
    if origins.size == 0:
        raise ValueError("configuration produces no rolling forecast origins")

    generator = random_generator(random_state)
    actual_values: list[FloatArray] = []
    means: list[FloatArray] = []
    lowers: list[FloatArray] = []
    uppers: list[FloatArray] = []
    for origin in origins:
        start = 0 if resolved_window is None else max(0, int(origin) - resolved_window)
        training = observations[start : int(origin)]
        model = model_factory()
        model.fit(training, weights)
        seed = int(generator.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        method_name = (
            "predict_interval"
            if interval_method == "conditional"
            else "predict_bootstrap_interval"
        )
        method = getattr(model, method_name, None)
        if method is None or not callable(method):
            raise TypeError(f"model does not provide {method_name}()")
        interval = method(
            steps=horizon,
            level=level,
            random_state=seed,
            **kwargs,
        )
        if not isinstance(interval, ForecastInterval):
            raise TypeError(f"{method_name}() must return ForecastInterval")
        actual_values.append(observations[int(origin) : int(origin) + horizon])
        means.append(interval.mean)
        lowers.append(interval.lower)
        uppers.append(interval.upper)

    return RollingOriginResult(
        origins=origins,
        observed=np.stack(actual_values, axis=0),
        mean=np.stack(means, axis=0),
        lower=np.stack(lowers, axis=0),
        upper=np.stack(uppers, axis=0),
        level=level,
        interval_method=interval_method,
    )
