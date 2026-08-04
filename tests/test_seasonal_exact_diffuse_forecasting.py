from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    ExactSeasonalIntegratedStateSpace,
    SeasonalExactDiffuseKalmanSTARIMA,
    SpatialWeights,
    StateSpaceModel,
    build_exact_integrated_state_space,
    build_exact_seasonal_integrated_state_space,
    exact_diffuse_forecast_interval,
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    simulate_exact_diffuse_forecast_paths,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def white_noise_state_space(
    *,
    mean: float = 0.0,
    variance: float = 0.4,
) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[0.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([mean]),
        innovation_covariance=np.array([[variance]]),
        ar_order=0,
        ma_order=0,
    )


def stable_ar_state_space(
    *,
    phi: float = 0.35,
    intercept: float = 0.1,
    variance: float = 0.3,
) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[phi]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([intercept]),
        innovation_covariance=np.array([[variance]]),
        ar_order=1,
        ma_order=0,
    )


def seasonal_random_walk(
    *,
    n_rows: int,
    period: int,
    drift: float,
    scale: float,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = np.zeros(n_rows, dtype=float)
    values[:period] = np.linspace(-0.4, 0.6, period)
    innovations = generator.normal(scale=scale, size=n_rows - period)
    for index in range(period, n_rows):
        values[index] = values[index - period] + drift + innovations[index - period]
    return values[:, None]


def test_period_two_paths_restore_levels_pathwise() -> None:
    variance = 0.5
    drift = 0.2
    data = seasonal_random_walk(
        n_rows=60,
        period=2,
        drift=drift,
        scale=np.sqrt(variance),
        seed=4,
    )
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_state_space(mean=drift, variance=variance),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    filtered = specification.filter(data)

    assert filtered.final_diffuse_rank == 0
    original_paths = simulate_seasonal_exact_diffuse_forecast_paths(
        filtered,
        specification,
        steps=6,
        n_simulations=30000,
        random_state=17,
    )
    differenced_paths = simulate_seasonal_exact_diffuse_differenced_forecast_paths(
        filtered,
        specification,
        steps=6,
        n_simulations=30000,
        random_state=17,
    )

    restored_differences = np.empty_like(original_paths)
    restored_differences[:, 0, 0] = original_paths[:, 0, 0] - data[-2, 0]
    restored_differences[:, 1, 0] = original_paths[:, 1, 0] - data[-1, 0]
    restored_differences[:, 2:, 0] = (
        original_paths[:, 2:, 0] - original_paths[:, :-2, 0]
    )
    np.testing.assert_allclose(
        differenced_paths,
        restored_differences,
        rtol=1e-12,
        atol=1e-12,
    )

    expected_mean = np.empty((6, 1), dtype=float)
    expected_mean[0, 0] = data[-2, 0] + drift
    expected_mean[1, 0] = data[-1, 0] + drift
    for step in range(2, 6):
        expected_mean[step, 0] = expected_mean[step - 2, 0] + drift
    expected_variance = variance * np.repeat(np.arange(1, 4), 2)

    np.testing.assert_allclose(original_paths.mean(axis=0), expected_mean, atol=0.025)
    np.testing.assert_allclose(
        original_paths[:, :, 0].var(axis=0),
        expected_variance,
        rtol=0.035,
        atol=0.02,
    )
    np.testing.assert_allclose(
        differenced_paths[:, :, 0].mean(axis=0),
        np.full(6, drift),
        atol=0.02,
    )
    np.testing.assert_allclose(
        differenced_paths[:, :, 0].var(axis=0),
        np.full(6, variance),
        rtol=0.035,
        atol=0.015,
    )


def test_original_interval_uses_pathwise_restored_quantiles() -> None:
    data = seasonal_random_walk(
        n_rows=48,
        period=3,
        drift=0.12,
        scale=0.45,
        seed=8,
    )
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_state_space(mean=0.12, variance=0.45**2),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=3,
    )
    filtered = specification.filter(data)
    paths = simulate_seasonal_exact_diffuse_forecast_paths(
        filtered,
        specification,
        steps=7,
        n_simulations=5000,
        random_state=91,
    )
    interval = seasonal_exact_diffuse_forecast_interval(
        filtered,
        specification,
        steps=7,
        level=0.9,
        n_simulations=5000,
        random_state=91,
    )

    np.testing.assert_allclose(interval.lower, np.quantile(paths, 0.05, axis=0))
    np.testing.assert_allclose(interval.upper, np.quantile(paths, 0.95, axis=0))
    assert "pathwise ordinary-seasonal level restoration" in interval.method
    assert interval.shape == (7, 1)
    assert not interval.mean.flags.writeable
    assert not interval.lower.flags.writeable
    assert not interval.upper.flags.writeable


def test_transformed_interval_matches_transformed_paths() -> None:
    data = seasonal_random_walk(
        n_rows=50,
        period=2,
        drift=-0.08,
        scale=0.3,
        seed=11,
    )
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_state_space(mean=-0.08, variance=0.09),
        ordinary_integration_order=1,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    filtered = specification.filter(data)
    paths = simulate_seasonal_exact_diffuse_differenced_forecast_paths(
        filtered,
        specification,
        steps=5,
        n_simulations=4000,
        random_state=103,
    )
    interval = seasonal_exact_diffuse_differenced_forecast_interval(
        filtered,
        specification,
        steps=5,
        level=0.8,
        n_simulations=4000,
        random_state=103,
    )

    np.testing.assert_allclose(interval.lower, np.quantile(paths, 0.1, axis=0))
    np.testing.assert_allclose(interval.upper, np.quantile(paths, 0.9, axis=0))
    assert "combined ordinary-seasonal difference scale" in interval.method


def test_zero_seasonal_order_reduces_to_ordinary_exact_diffuse_forecasting() -> None:
    rng = np.random.default_rng(13)
    data = np.cumsum(rng.normal(size=70))[:, None]
    transformed = stable_ar_state_space()
    seasonal_specification = build_exact_seasonal_integrated_state_space(
        transformed,
        ordinary_integration_order=1,
        seasonal_integration_order=0,
        seasonal_period=4,
    )
    ordinary_specification = build_exact_integrated_state_space(transformed, 1)
    seasonal_filter = seasonal_specification.filter(data)
    ordinary_filter = ordinary_specification.filter(data)

    seasonal_paths = simulate_seasonal_exact_diffuse_forecast_paths(
        seasonal_filter,
        seasonal_specification,
        steps=5,
        n_simulations=2500,
        random_state=44,
    )
    ordinary_paths = simulate_exact_diffuse_forecast_paths(
        ordinary_filter,
        steps=5,
        n_simulations=2500,
        random_state=44,
    )
    np.testing.assert_allclose(seasonal_paths, ordinary_paths, rtol=1e-12, atol=1e-12)

    seasonal_interval = seasonal_exact_diffuse_forecast_interval(
        seasonal_filter,
        seasonal_specification,
        steps=5,
        level=0.9,
        n_simulations=2500,
        random_state=44,
    )
    ordinary_interval = exact_diffuse_forecast_interval(
        ordinary_filter,
        steps=5,
        level=0.9,
        n_simulations=2500,
        random_state=44,
    )
    np.testing.assert_allclose(seasonal_interval.mean, ordinary_interval.mean)
    np.testing.assert_allclose(seasonal_interval.lower, ordinary_interval.lower)
    np.testing.assert_allclose(seasonal_interval.upper, ordinary_interval.upper)


def test_fitted_facade_paths_and_intervals_match_point_forecasts() -> None:
    data = seasonal_random_walk(
        n_rows=100,
        period=4,
        drift=0.15,
        scale=0.25,
        seed=27,
    )
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(
        data,
        identity_weights(),
        start_params=np.array([0.1]),
        start_covariance=np.array([[0.08]]),
    )

    original_paths = model.simulate_forecast_paths(
        steps=6,
        n_simulations=3000,
        random_state=66,
    )
    differenced_paths = model.simulate_differenced_forecast_paths(
        steps=6,
        n_simulations=3000,
        random_state=66,
    )
    original_interval = model.predict_interval(
        steps=6,
        level=0.9,
        n_simulations=3000,
        random_state=66,
    )
    differenced_interval = model.predict_differenced_interval(
        steps=6,
        level=0.9,
        n_simulations=3000,
        random_state=66,
    )

    np.testing.assert_allclose(original_interval.mean, model.predict(steps=6))
    np.testing.assert_allclose(
        differenced_interval.mean,
        model.predict_differenced(steps=6),
    )
    np.testing.assert_allclose(
        original_interval.lower,
        np.quantile(original_paths, 0.05, axis=0),
    )
    np.testing.assert_allclose(
        original_interval.upper,
        np.quantile(original_paths, 0.95, axis=0),
    )
    np.testing.assert_allclose(
        differenced_interval.lower,
        np.quantile(differenced_paths, 0.05, axis=0),
    )
    np.testing.assert_allclose(
        differenced_interval.upper,
        np.quantile(differenced_paths, 0.95, axis=0),
    )


def test_unresolved_terminal_diffuse_rank_is_rejected() -> None:
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_state_space(),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=4,
    )
    filtered = specification.filter(np.full((3, 1), np.nan))

    assert filtered.final_diffuse_rank > 0
    with pytest.raises(RuntimeError, match="zero remaining diffuse rank"):
        simulate_seasonal_exact_diffuse_forecast_paths(
            filtered,
            specification,
            steps=2,
            n_simulations=100,
        )
    with pytest.raises(RuntimeError, match="zero remaining diffuse rank"):
        seasonal_exact_diffuse_differenced_forecast_interval(
            filtered,
            specification,
            steps=2,
            n_simulations=100,
        )


def test_seasonal_forecast_input_validation() -> None:
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_state_space(),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    filtered = specification.filter(np.arange(12, dtype=float)[:, None])
    other = build_exact_seasonal_integrated_state_space(
        white_noise_state_space(),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )

    with pytest.raises(TypeError, match="ExactDiffuseFilterResult"):
        simulate_seasonal_exact_diffuse_forecast_paths(  # type: ignore[arg-type]
            object(),
            specification,
            steps=2,
            n_simulations=100,
        )
    with pytest.raises(TypeError, match="ExactSeasonalIntegratedStateSpace"):
        simulate_seasonal_exact_diffuse_forecast_paths(  # type: ignore[arg-type]
            filtered,
            object(),
            steps=2,
            n_simulations=100,
        )
    with pytest.raises(ValueError, match="state space used"):
        simulate_seasonal_exact_diffuse_forecast_paths(
            filtered,
            other,
            steps=2,
            n_simulations=100,
        )
    with pytest.raises(ValueError, match="steps"):
        simulate_seasonal_exact_diffuse_forecast_paths(
            filtered,
            specification,
            steps=0,
            n_simulations=100,
        )
    with pytest.raises(ValueError, match="at least two"):
        simulate_seasonal_exact_diffuse_differenced_forecast_paths(
            filtered,
            specification,
            steps=2,
            n_simulations=1,
        )
    with pytest.raises(ValueError, match="strictly between"):
        seasonal_exact_diffuse_forecast_interval(
            filtered,
            specification,
            steps=2,
            level=1.0,
            n_simulations=100,
        )


def test_specification_type_is_public() -> None:
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_state_space(),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    assert isinstance(specification, ExactSeasonalIntegratedStateSpace)
