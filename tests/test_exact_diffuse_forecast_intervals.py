from __future__ import annotations

import numpy as np
import pytest

from pystarmax import ExactDiffuseKalmanSTARIMA, SpatialWeights
from pystarmax.exact_diffuse import exact_diffuse_filter
from pystarmax.exact_diffuse_forecasting import (
    exact_diffuse_forecast_interval,
    simulate_exact_diffuse_forecast_paths,
)
from pystarmax.state_space import StateSpaceModel


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def analytic_filter_result():
    model = StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.5]),
        innovation_covariance=np.array([[1.0]]),
        ar_order=1,
        ma_order=0,
    )
    return exact_diffuse_filter(
        np.array([[np.nan]]),
        model,
        initial_state=np.array([0.5]),
        initial_covariance=np.array([[3.0]]),
        initial_diffuse_covariance=np.zeros((1, 1), dtype=float),
    )


def test_paths_propagate_terminal_state_and_future_innovations() -> None:
    paths = simulate_exact_diffuse_forecast_paths(
        analytic_filter_result(),
        steps=2,
        n_simulations=30_000,
        random_state=2026,
    )

    np.testing.assert_allclose(paths.mean(axis=0)[:, 0], [1.5, 2.0], atol=0.04)
    np.testing.assert_allclose(paths.var(axis=0)[:, 0], [5.0, 6.0], rtol=0.04)


def test_exact_interval_is_reproducible_and_matches_recursive_mean() -> None:
    rng = np.random.default_rng(24)
    increments = rng.normal(loc=0.2, scale=0.45, size=140)
    levels = np.cumsum(increments)[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(levels, identity_weights())

    first = model.predict_interval(
        steps=5,
        level=0.9,
        n_simulations=5000,
        random_state=31,
    )
    second = exact_diffuse_forecast_interval(
        model.filter(),
        steps=5,
        level=0.9,
        n_simulations=5000,
        random_state=31,
    )

    np.testing.assert_allclose(first.mean, model.predict(steps=5))
    np.testing.assert_allclose(first.lower, second.lower)
    np.testing.assert_allclose(first.upper, second.upper)
    assert first.method.startswith("fixed-parameter exact diffuse")
    assert float((first.upper[-1] - first.lower[-1])[0]) > float(
        (first.upper[0] - first.lower[0])[0]
    )


def test_differenced_projection_matches_point_forecast() -> None:
    rng = np.random.default_rng(44)
    second_differences = rng.normal(loc=0.1, scale=0.35, size=160)
    first_differences = np.cumsum(second_differences)
    levels = np.cumsum(first_differences)[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=2,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(levels, identity_weights())

    interval = model.predict_differenced_interval(
        steps=4,
        level=0.9,
        n_simulations=4000,
        random_state=51,
    )

    np.testing.assert_allclose(interval.mean, model.predict_differenced(steps=4))
    assert interval.shape == (4, 1)


def test_unresolved_terminal_diffuse_rank_is_rejected() -> None:
    model = StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[1.0]]),
        ar_order=1,
        ma_order=0,
    )
    filtered = exact_diffuse_filter(
        np.array([[np.nan], [np.nan]]),
        model,
    )

    assert filtered.final_diffuse_rank == 1
    with pytest.raises(RuntimeError, match="diffuse rank"):
        simulate_exact_diffuse_forecast_paths(
            filtered,
            steps=2,
            n_simulations=100,
            random_state=1,
        )


def test_exact_interval_argument_validation() -> None:
    result = analytic_filter_result()
    with pytest.raises(ValueError, match="steps"):
        exact_diffuse_forecast_interval(result, steps=0)
    with pytest.raises(ValueError, match="level"):
        exact_diffuse_forecast_interval(result, level=1.0)
    with pytest.raises(ValueError, match="n_simulations"):
        exact_diffuse_forecast_interval(result, n_simulations=1)
    with pytest.raises(TypeError, match="ExactDiffuseFilterResult"):
        simulate_exact_diffuse_forecast_paths(  # type: ignore[arg-type]
            object(),
            steps=1,
            n_simulations=2,
        )
