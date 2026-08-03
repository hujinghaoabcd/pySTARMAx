from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pystarmax.evaluation import (
    IntervalMetrics,
    RollingOriginResult,
    interval_score,
    rolling_origin_evaluate,
)
from pystarmax.forecasting import ForecastInterval


class DummyIntervalModel:
    def __init__(self, fitted_lengths: list[int], calls: list[str]) -> None:
        self.fitted_lengths = fitted_lengths
        self.calls = calls
        self.last: np.ndarray[Any, np.dtype[np.float64]] | None = None

    def fit(self, data: Any, weights: Any) -> None:
        values = np.asarray(data, dtype=float)
        self.fitted_lengths.append(int(values.shape[0]))
        self.last = values[-1]

    def _interval(self, steps: int, level: float, method: str) -> ForecastInterval:
        if self.last is None:
            raise RuntimeError("fit first")
        mean = np.repeat(self.last[None, :], steps, axis=0)
        return ForecastInterval(
            mean=mean,
            lower=mean - 1.0,
            upper=mean + 1.0,
            level=level,
            n_simulations=4,
            method=method,
        )

    def predict_interval(
        self,
        steps: int,
        *,
        level: float,
        random_state: int,
        **kwargs: Any,
    ) -> ForecastInterval:
        self.calls.append(f"conditional:{random_state}")
        return self._interval(steps, level, "conditional")

    def predict_bootstrap_interval(
        self,
        steps: int,
        *,
        level: float,
        random_state: int,
        **kwargs: Any,
    ) -> ForecastInterval:
        self.calls.append(f"bootstrap:{random_state}")
        return self._interval(steps, level, "bootstrap")


def test_interval_score_matches_winkler_formula() -> None:
    observed = np.array([0.0, -2.0, 3.0])
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])

    score = interval_score(observed, lower, upper, level=0.8)

    np.testing.assert_allclose(score, np.array([2.0, 12.0, 22.0]))


def test_interval_score_rejects_invalid_arrays() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        interval_score(np.zeros(2), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        interval_score(np.zeros(2), np.ones(2), np.zeros(2))
    with pytest.raises(ValueError, match="strictly between"):
        interval_score(np.zeros(2), np.zeros(2), np.ones(2), level=1.0)


def test_rolling_result_metrics_and_horizon_metrics() -> None:
    observed = np.array([[[0.0], [3.0]], [[0.5], [-2.0]]])
    mean = np.zeros_like(observed)
    lower = np.full_like(observed, -1.0)
    upper = np.full_like(observed, 1.0)
    result = RollingOriginResult(
        origins=np.array([4, 6]),
        observed=observed,
        mean=mean,
        lower=lower,
        upper=upper,
        level=0.5,
        interval_method="conditional",
    )

    metrics = result.metrics()

    assert result.shape == (2, 2, 1)
    np.testing.assert_array_equal(
        result.covered,
        (observed >= -1.0) & (observed <= 1.0),
    )
    np.testing.assert_allclose(result.widths, 2.0)
    assert metrics.nominal_coverage == 0.5
    assert metrics.empirical_coverage == 0.5
    assert metrics.coverage_gap == 0.0
    assert metrics.average_width == 2.0
    assert metrics.n_forecasts == 4
    assert len(result.metrics_by_horizon()) == 2
    assert result.metrics_by_horizon()[0].empirical_coverage == 1.0
    assert result.metrics_by_horizon()[1].empirical_coverage == 0.0


def test_interval_metrics_are_immutable_and_validate_values() -> None:
    metrics = IntervalMetrics(
        nominal_coverage=0.9,
        empirical_coverage=0.8,
        coverage_gap=-0.1,
        average_width=2.0,
        mean_interval_score=2.5,
        mae=0.5,
        rmse=0.7,
        n_forecasts=10,
    )

    assert metrics.absolute_coverage_error == pytest.approx(0.1)
    with pytest.raises(AttributeError):
        metrics.mae = 2.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-negative"):
        IntervalMetrics(0.9, 0.8, -0.1, -1.0, 2.0, 1.0, 1.0, 10)


def test_rolling_origin_expanding_window_and_seed_reproducibility() -> None:
    data = np.arange(24, dtype=float).reshape(12, 2)
    lengths_a: list[int] = []
    calls_a: list[str] = []
    lengths_b: list[int] = []
    calls_b: list[str] = []

    result_a = rolling_origin_evaluate(
        lambda: DummyIntervalModel(lengths_a, calls_a),
        data,
        weights=None,
        initial_window=6,
        horizon=2,
        step=2,
        random_state=42,
    )
    result_b = rolling_origin_evaluate(
        lambda: DummyIntervalModel(lengths_b, calls_b),
        data,
        weights=None,
        initial_window=6,
        horizon=2,
        step=2,
        random_state=42,
    )

    np.testing.assert_array_equal(result_a.origins, np.array([6, 8, 10]))
    assert result_a.shape == (3, 2, 2)
    assert lengths_a == [6, 8, 10]
    assert calls_a == calls_b
    np.testing.assert_allclose(result_a.observed[0], data[6:8])
    np.testing.assert_allclose(result_a.mean[0], np.repeat(data[5:6], 2, axis=0))


def test_rolling_origin_fixed_window_and_bootstrap_dispatch() -> None:
    data = np.arange(20, dtype=float).reshape(10, 2)
    lengths: list[int] = []
    calls: list[str] = []

    result = rolling_origin_evaluate(
        lambda: DummyIntervalModel(lengths, calls),
        data,
        weights=None,
        initial_window=5,
        horizon=1,
        step=2,
        window_size=3,
        interval_method="bootstrap",
        interval_kwargs={"n_bootstrap": 4},
        random_state=7,
    )

    np.testing.assert_array_equal(result.origins, np.array([5, 7, 9]))
    assert lengths == [3, 3, 3]
    assert all(call.startswith("bootstrap:") for call in calls)
    assert result.interval_method == "bootstrap"


def test_rolling_origin_rejects_invalid_configuration() -> None:
    data = np.arange(16, dtype=float).reshape(8, 2)
    factory = lambda: DummyIntervalModel([], [])

    with pytest.raises(ValueError, match="complete forecast horizon"):
        rolling_origin_evaluate(
            factory,
            data,
            None,
            initial_window=7,
            horizon=2,
        )
    with pytest.raises(ValueError, match="conditional.*bootstrap"):
        rolling_origin_evaluate(
            factory,
            data,
            None,
            initial_window=4,
            interval_method="other",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must not override"):
        rolling_origin_evaluate(
            factory,
            data,
            None,
            initial_window=4,
            interval_kwargs={"steps": 2},
        )


def test_rolling_origin_requires_interval_method() -> None:
    class NoIntervalModel:
        def fit(self, data: Any, weights: Any) -> None:
            return None

    data = np.arange(16, dtype=float).reshape(8, 2)
    with pytest.raises(TypeError, match="predict_interval"):
        rolling_origin_evaluate(
            NoIntervalModel,
            data,
            None,
            initial_window=4,
        )
