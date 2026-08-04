from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    DifferencingState,
    KalmanSTARIMA,
    KalmanSTARMA,
    SeasonalKalmanSTARIMA,
    SpatialWeights,
)
from pystarmax.kalman_forecasting import (
    inverse_forecast_paths,
    kalman_forecast_interval,
    simulate_kalman_forecast_paths,
)
from pystarmax.state_space import KalmanFilterResult, StateSpaceModel


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def analytic_filter_result() -> KalmanFilterResult:
    model = StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.5]),
        innovation_covariance=np.array([[1.0]]),
        ar_order=1,
        ma_order=0,
    )
    return KalmanFilterResult(
        model=model,
        predicted_state=np.array([[1.0]]),
        filtered_state=np.array([[1.0]]),
        predicted_covariance=np.array([[[4.0]]]),
        filtered_covariance=np.array([[[4.0]]]),
        innovations=np.array([[0.0]]),
        innovation_covariance=np.array([[[1.0]]]),
        observed_mask=np.array([[True]]),
        log_likelihood_contributions=np.array([-1.0]),
        jitter=np.array([0.0]),
        log_likelihood=-1.0,
        n_observations=1,
        initialization="known",
    )


def test_paths_propagate_filtered_state_and_future_innovation_uncertainty() -> None:
    paths = simulate_kalman_forecast_paths(
        analytic_filter_result(),
        steps=2,
        n_simulations=30_000,
        random_state=2026,
    )

    np.testing.assert_allclose(paths.mean(axis=0)[:, 0], [1.5, 2.0], atol=0.04)
    np.testing.assert_allclose(paths.var(axis=0)[:, 0], [5.0, 6.0], rtol=0.04)


def test_stationary_interval_is_reproducible_and_matches_recursive_mean() -> None:
    rng = np.random.default_rng(17)
    data = rng.normal(loc=0.3, scale=0.7, size=(120, 1))
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(data, identity_weights())

    first = model.predict_interval(
        steps=3,
        level=0.9,
        n_simulations=4000,
        random_state=11,
    )
    second = kalman_forecast_interval(
        model.filter(),
        steps=3,
        level=0.9,
        n_simulations=4000,
        random_state=11,
    )

    np.testing.assert_allclose(first.mean, model.predict(steps=3))
    np.testing.assert_allclose(first.lower, second.lower)
    np.testing.assert_allclose(first.upper, second.upper)
    assert first.method.startswith("fixed-parameter Gaussian")


def test_inverse_forecast_paths_applies_the_recursion_per_simulation() -> None:
    state = DifferencingState(
        order=1,
        n_locations=1,
        anchors=(np.array([10.0]),),
    )
    paths = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[-1.0], [0.0], [1.0]],
        ]
    )

    restored = inverse_forecast_paths(state, paths)

    np.testing.assert_allclose(restored[0, :, 0], [11.0, 13.0, 16.0])
    np.testing.assert_allclose(restored[1, :, 0], [9.0, 9.0, 10.0])


def test_ordinary_integrated_interval_is_inverted_pathwise() -> None:
    rng = np.random.default_rng(27)
    increments = rng.normal(loc=0.2, scale=0.45, size=180)
    levels = np.cumsum(increments)[:, None]
    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(levels, identity_weights())

    transformed = model.predict_differenced_interval(
        steps=6,
        level=0.9,
        n_simulations=6000,
        random_state=37,
    )
    original = model.predict_interval(
        steps=6,
        level=0.9,
        n_simulations=6000,
        random_state=37,
    )

    np.testing.assert_allclose(transformed.mean, model.predict_differenced(steps=6))
    np.testing.assert_allclose(original.mean, model.predict(steps=6))
    np.testing.assert_allclose(
        original.lower[0],
        levels[-1] + transformed.lower[0],
    )
    np.testing.assert_allclose(
        original.upper[0],
        levels[-1] + transformed.upper[0],
    )
    assert float(original.upper[-1] - original.lower[-1]) > float(
        original.upper[0] - original.lower[0]
    )


def test_seasonal_interval_uses_rolling_cycle_pathwise() -> None:
    rng = np.random.default_rng(47)
    period = 4
    values = np.zeros(220, dtype=float)
    innovations = rng.normal(scale=0.4, size=values.size)
    for time_index in range(period, values.size):
        values[time_index] = values[time_index - period] + innovations[time_index]
    levels = values[:, None]
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=period,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=300,
    )
    model.fit(levels, identity_weights())

    transformed = model.predict_differenced_interval(
        steps=period + 1,
        level=0.9,
        n_simulations=6000,
        random_state=57,
    )
    original = model.predict_interval(
        steps=period + 1,
        level=0.9,
        n_simulations=6000,
        random_state=57,
    )

    np.testing.assert_allclose(original.mean, model.predict(steps=period + 1))
    for horizon in range(period):
        np.testing.assert_allclose(
            original.lower[horizon],
            levels[-period + horizon] + transformed.lower[horizon],
        )
        np.testing.assert_allclose(
            original.upper[horizon],
            levels[-period + horizon] + transformed.upper[horizon],
        )
    assert float(original.upper[-1] - original.lower[-1]) > float(
        original.upper[0] - original.lower[0]
    )


def test_original_interval_requires_finite_terminal_anchors() -> None:
    rng = np.random.default_rng(67)
    levels = np.cumsum(rng.normal(size=100))[:, None]
    levels[-1, 0] = np.nan
    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        include_intercept=False,
        max_iter=300,
    )
    model.fit(levels, identity_weights())

    transformed = model.predict_differenced_interval(
        steps=2,
        n_simulations=200,
        random_state=1,
    )

    assert transformed.shape == (2, 1)
    with pytest.raises(RuntimeError, match="finite terminal"):
        model.predict_interval(steps=2, n_simulations=200, random_state=1)


def test_interval_argument_validation() -> None:
    result = analytic_filter_result()
    with pytest.raises(ValueError, match="steps"):
        kalman_forecast_interval(result, steps=0)
    with pytest.raises(ValueError, match="level"):
        kalman_forecast_interval(result, level=1.0)
    with pytest.raises(ValueError, match="n_simulations"):
        kalman_forecast_interval(result, n_simulations=1)
