from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    ExactDiffuseKalmanSTARIMA,
    SpatialWeights,
    StateSpaceModel,
    build_exact_integrated_state_space,
    exact_diffuse_filter,
    exact_diffuse_forecast_interval,
    kalman_filter,
    simulate_exact_diffuse_forecast_paths,
    simulate_kalman_forecast_paths,
)


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def white_noise_state_space(variance: float = 0.4) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[0.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[variance]]),
        ar_order=0,
        ma_order=0,
    )


def stable_ar_state_space(
    phi: float = 0.45,
    variance: float = 0.3,
) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[phi]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.1]),
        innovation_covariance=np.array([[variance]]),
        ar_order=1,
        ma_order=0,
    )


def test_zero_diffuse_paths_match_ordinary_kalman_paths() -> None:
    rng = np.random.default_rng(2026)
    model = stable_ar_state_space()
    data = rng.normal(size=(80, 1))
    data[25:29, 0] = np.nan

    exact_specification = build_exact_integrated_state_space(model, 0)
    exact_filter = exact_specification.filter(data)
    ordinary_filter = kalman_filter(data, model, initialization="stationary")

    assert exact_filter.final_diffuse_rank == 0
    np.testing.assert_allclose(
        exact_filter.filtered_state,
        ordinary_filter.filtered_state,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        exact_filter.filtered_covariance,
        ordinary_filter.filtered_covariance,
        rtol=1e-10,
        atol=1e-10,
    )

    exact_paths = simulate_exact_diffuse_forecast_paths(
        exact_filter,
        steps=5,
        n_simulations=2000,
        random_state=123,
    )
    ordinary_paths = simulate_kalman_forecast_paths(
        ordinary_filter,
        steps=5,
        n_simulations=2000,
        random_state=123,
    )
    np.testing.assert_allclose(exact_paths, ordinary_paths, rtol=1e-12, atol=1e-12)


def test_random_walk_forecast_variance_grows_linearly() -> None:
    variance = 0.5
    transformed = white_noise_state_space(variance)
    specification = build_exact_integrated_state_space(transformed, 1)
    levels = np.arange(20, dtype=float)[:, None]
    filtered = specification.filter(levels)

    paths = simulate_exact_diffuse_forecast_paths(
        filtered,
        steps=3,
        n_simulations=30000,
        random_state=77,
    )
    expected_mean = np.repeat(levels[-1:], 3, axis=0)
    expected_variance = variance * np.arange(1, 4, dtype=float)

    np.testing.assert_allclose(paths.mean(axis=0), expected_mean, atol=0.025)
    np.testing.assert_allclose(
        paths[:, :, 0].var(axis=0),
        expected_variance,
        rtol=0.035,
        atol=0.015,
    )

    interval = exact_diffuse_forecast_interval(
        filtered,
        steps=3,
        level=0.9,
        n_simulations=5000,
        random_state=77,
    )
    np.testing.assert_allclose(interval.mean, expected_mean)
    assert np.all(interval.lower < interval.mean)
    assert np.all(interval.upper > interval.mean)
    assert interval.shape == (3, 1)
    assert "exact diffuse" in interval.method
    assert not interval.mean.flags.writeable


def test_fitted_model_intervals_match_point_forecasts_and_are_reproducible() -> None:
    rng = np.random.default_rng(91)
    increments = 0.15 + rng.normal(scale=0.35, size=120)
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
        steps=6,
        level=0.9,
        n_simulations=3000,
        random_state=12,
    )
    second = model.predict_interval(
        steps=6,
        level=0.9,
        n_simulations=3000,
        random_state=12,
    )
    differenced = model.predict_differenced_interval(
        steps=6,
        level=0.9,
        n_simulations=3000,
        random_state=12,
    )

    np.testing.assert_allclose(first.mean, model.predict(steps=6))
    np.testing.assert_allclose(
        differenced.mean,
        model.predict_differenced(steps=6),
    )
    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)
    assert np.all(first.upper - first.lower > 0.0)
    assert np.all(differenced.upper - differenced.lower > 0.0)
    assert "highest ordinary-difference scale" in differenced.method


def test_forecast_simulation_rejects_unresolved_terminal_diffuse_rank() -> None:
    model = StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[0.5]]),
        ar_order=1,
        ma_order=0,
    )
    filtered = exact_diffuse_filter(
        np.full((4, 1), np.nan),
        model,
    )

    assert filtered.final_diffuse_rank == 1
    with pytest.raises(RuntimeError, match="zero remaining diffuse rank"):
        simulate_exact_diffuse_forecast_paths(
            filtered,
            steps=2,
            n_simulations=100,
        )


def test_exact_diffuse_forecast_validation() -> None:
    filtered = build_exact_integrated_state_space(
        white_noise_state_space(),
        0,
    ).filter(np.array([[0.0], [0.2], [-0.1]]))

    with pytest.raises(TypeError, match="ExactDiffuseFilterResult"):
        simulate_exact_diffuse_forecast_paths(  # type: ignore[arg-type]
            object(),
            steps=2,
            n_simulations=100,
        )
    with pytest.raises(ValueError, match="steps"):
        simulate_exact_diffuse_forecast_paths(
            filtered,
            steps=0,
            n_simulations=100,
        )
    with pytest.raises(ValueError, match="at least two"):
        simulate_exact_diffuse_forecast_paths(
            filtered,
            steps=2,
            n_simulations=1,
        )
    with pytest.raises(ValueError, match="strictly between"):
        exact_diffuse_forecast_interval(
            filtered,
            steps=2,
            level=1.0,
            n_simulations=100,
        )
