import numpy as np
import pytest

from pystarmax import (
    STARIMA,
    STARMA,
    ForecastInterval,
    SeasonalSTARIMA,
    SpatialWeights,
    combined_difference,
    differencing_coefficients,
    restore_fitted_values,
    simulate_seasonal_starima,
    simulate_seasonal_starma,
    simulate_starima,
    simulate_starma,
)


def _identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights.from_matrices(
        [np.eye(n_locations, dtype=float)], names=["W0"]
    )


def test_forecast_interval_is_validated_and_immutable() -> None:
    mean = np.array([[1.0], [2.0]])
    interval = ForecastInterval(
        mean=mean,
        lower=mean - 0.5,
        upper=mean + 0.5,
        level=0.9,
        n_simulations=100,
    )
    assert interval.shape == (2, 1)
    assert interval.level == 0.9
    assert not interval.mean.flags.writeable
    with pytest.raises(ValueError, match="between"):
        ForecastInterval(mean, mean, mean, level=1.0, n_simulations=10)
    with pytest.raises(ValueError, match="at least two"):
        ForecastInterval(mean, mean, mean, level=0.9, n_simulations=1)
    with pytest.raises(ValueError, match="share one shape"):
        ForecastInterval(mean, mean[:1], mean, level=0.9, n_simulations=10)
    with pytest.raises(ValueError, match="must not exceed"):
        ForecastInterval(mean, mean + 1.0, mean, level=0.9, n_simulations=10)


def test_combined_differencing_coefficients_and_fitted_restoration() -> None:
    coefficients = differencing_coefficients(
        ordinary_order=1,
        seasonal_order=1,
        seasonal_period=3,
    )
    np.testing.assert_allclose(coefficients, [1.0, -1.0, 0.0, -1.0, 1.0])

    time = np.arange(20.0)
    data = np.column_stack((0.2 * time**2 + 2.0, np.sin(time / 3.0)))
    transformed, _ = combined_difference(
        data,
        ordinary_order=1,
        seasonal_order=1,
        seasonal_period=3,
    )
    fitted = transformed.copy()
    fitted[:2] = np.nan
    restored = restore_fitted_values(
        data,
        fitted,
        ordinary_order=1,
        seasonal_order=1,
        seasonal_period=3,
    )
    assert np.isnan(restored[:6]).all()
    np.testing.assert_allclose(restored[6:], data[6:])

    with pytest.raises(ValueError, match="incompatible number"):
        restore_fitted_values(data, fitted[:-1], ordinary_order=1)
    partial = fitted.copy()
    partial[2, 0] = np.nan
    with pytest.raises(ValueError, match="entirely finite"):
        restore_fitted_values(
            data,
            partial,
            ordinary_order=1,
            seasonal_order=1,
            seasonal_period=3,
        )


def test_starma_interval_is_reproducible_and_matches_point_forecast() -> None:
    weights = _identity_weights(2)
    data = simulate_starma(
        phi=np.array([[0.45]]),
        theta=np.array([[0.20]]),
        weights=weights,
        n_steps=350,
        burnin=150,
        innovation_covariance=np.array([[1.0, 0.4], [0.4, 0.8]]),
        random_state=14,
    )
    model = STARMA(
        ar_order=1,
        ma_order=1,
        include_intercept=False,
        max_iter=100,
    )
    model.fit(data, weights)
    first = model.predict_interval(
        steps=5, level=0.9, n_simulations=1200, random_state=55
    )
    second = model.predict_interval(
        steps=5, level=0.9, n_simulations=1200, random_state=55
    )
    np.testing.assert_allclose(first.mean, model.predict(5))
    np.testing.assert_allclose(first.lower, second.lower)
    np.testing.assert_allclose(first.upper, second.upper)
    assert np.all(first.upper > first.lower)


def test_starima_original_fitted_values_and_interval_path_inversion() -> None:
    weights = _identity_weights(2)
    data = simulate_starima(
        phi=np.array([[0.35]]),
        theta=np.array([[0.15]]),
        weights=weights,
        n_steps=280,
        integration_order=1,
        burnin=120,
        random_state=17,
    )
    model = STARIMA(
        ar_order=1,
        integration_order=1,
        ma_order=1,
        include_intercept=False,
        max_iter=100,
    )
    result = model.fit(data, weights)
    fitted = model.fitted_original()
    assert fitted.shape == data.shape
    assert np.isnan(fitted[:2]).all()
    np.testing.assert_allclose(
        data[2:] - fitted[2:],
        result.residuals[1:],
    )

    interval = model.predict_interval(
        steps=6,
        level=0.8,
        n_simulations=800,
        random_state=91,
    )
    transformed_paths = model.core_model._simulate_forecast_paths(
        steps=6,
        n_simulations=800,
        random_state=91,
    )
    assert model.differencing_state_ is not None
    original_paths = np.stack(
        [
            model.differencing_state_.inverse_forecast(path)
            for path in transformed_paths
        ],
        axis=0,
    )
    np.testing.assert_allclose(interval.lower, np.quantile(original_paths, 0.1, axis=0))
    np.testing.assert_allclose(interval.upper, np.quantile(original_paths, 0.9, axis=0))
    np.testing.assert_allclose(interval.mean, model.predict(6))
    width = np.mean(interval.upper - interval.lower, axis=1)
    assert width[-1] > width[0]


def test_seasonal_original_fitted_values_and_interval_propagation() -> None:
    weights = _identity_weights(1)
    data = simulate_seasonal_starima(
        phi=np.array([[0.25]]),
        seasonal_phi=np.zeros((0, 1)),
        theta=np.zeros((0, 1)),
        seasonal_theta=np.zeros((0, 1)),
        weights=weights,
        seasonal_period=4,
        n_steps=260,
        integration_order=1,
        seasonal_integration_order=1,
        burnin=100,
        random_state=23,
    )
    model = SeasonalSTARIMA(
        ar_order=1,
        integration_order=1,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_period=4,
        include_intercept=False,
    )
    result = model.fit(data, weights)
    fitted = model.fitted_original()
    expected = restore_fitted_values(
        data,
        result.fitted_values,
        ordinary_order=1,
        seasonal_order=1,
        seasonal_period=4,
    )
    np.testing.assert_allclose(fitted, expected, equal_nan=True)

    interval = model.predict_interval(
        steps=8,
        level=0.9,
        n_simulations=600,
        random_state=101,
    )
    transformed_paths = model._simulate_forecast_paths(
        steps=8,
        n_simulations=600,
        random_state=101,
    )
    assert model.differencing_state_ is not None
    original_paths = np.stack(
        [
            model.differencing_state_.inverse_forecast(path)
            for path in transformed_paths
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        interval.lower, np.quantile(original_paths, 0.05, axis=0)
    )
    np.testing.assert_allclose(
        interval.upper, np.quantile(original_paths, 0.95, axis=0)
    )


def test_multiplicative_seasonal_interval_supports_nonlinear_core() -> None:
    weights = _identity_weights(1)
    data = simulate_seasonal_starma(
        phi=np.array([[0.20]]),
        seasonal_phi=np.array([[0.30]]),
        theta=np.zeros((0, 1)),
        seasonal_theta=np.zeros((0, 1)),
        weights=weights,
        seasonal_period=4,
        n_steps=450,
        burnin=160,
        random_state=31,
    )
    model = SeasonalSTARIMA(
        ar_order=1,
        seasonal_ar_order=1,
        seasonal_period=4,
        seasonal_integration_order=0,
        include_intercept=False,
        max_iter=1500,
    )
    model.fit(data, weights)
    interval = model.predict_interval(
        steps=5,
        level=0.9,
        n_simulations=500,
        random_state=8,
    )
    assert model.core_model_ is None
    assert interval.shape == (5, 1)
    np.testing.assert_allclose(interval.mean, model.predict(5))
    assert np.all(interval.upper > interval.lower)


def test_interval_runtime_and_argument_errors() -> None:
    model = STARMA(ar_order=1)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_interval()

    weights = _identity_weights(1)
    data = simulate_starma(
        phi=np.array([[0.2]]),
        theta=np.zeros((0, 1)),
        weights=weights,
        n_steps=40,
        burnin=20,
        random_state=4,
    )
    model.fit(data, weights)
    with pytest.raises(ValueError, match="between"):
        model.predict_interval(level=1.0)
    with pytest.raises(ValueError, match="at least two"):
        model.predict_interval(n_simulations=1)
    with pytest.raises(ValueError, match="positive"):
        model.predict_interval(steps=0)
