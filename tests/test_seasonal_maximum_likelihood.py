from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    KalmanSTARMA,
    SeasonalKalmanSTARIMA,
    SeasonalKalmanSTARIMAResult,
    SpatialWeights,
)


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_zero_seasonal_orders_match_stationary_kalman_starma() -> None:
    rng = np.random.default_rng(2026)
    data = rng.normal(loc=0.2, scale=0.7, size=(100, 1))
    weights = identity_weights()
    start_params = np.array([0.1, 0.25])
    start_covariance = np.array([[0.6]])

    stationary = KalmanSTARMA(
        ar_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    seasonal = SeasonalKalmanSTARIMA(
        ar_order=1,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )

    stationary_result = stationary.fit(
        data,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )
    seasonal_result = seasonal.fit(
        data,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )

    np.testing.assert_allclose(seasonal_result.params, stationary_result.params)
    np.testing.assert_allclose(
        seasonal_result.innovation_covariance,
        stationary_result.innovation_covariance,
    )
    assert seasonal_result.log_likelihood == pytest.approx(
        stationary_result.log_likelihood
    )
    np.testing.assert_allclose(seasonal.predict(5), stationary.predict(5))
    np.testing.assert_allclose(
        seasonal.filter().filtered_state,
        stationary.filter().filtered_state,
    )
    assert seasonal_result.order == (1, 0, 0)
    assert seasonal_result.seasonal_order == (0, 0, 0, 4)


def test_multiplicative_operator_expansion_keeps_matrix_cross_products() -> None:
    model = SeasonalKalmanSTARIMA(
        ar_order=1,
        integration_order=0,
        ma_order=1,
        seasonal_ar_order=1,
        seasonal_integration_order=0,
        seasonal_ma_order=1,
        seasonal_period=4,
        include_intercept=False,
    )
    weights = identity_weights()
    expanded = model._expanded(  # noqa: SLF001
        np.array([0.2, 0.3, 0.4, 0.5]),
        weights,
    )

    assert expanded[5] == (1, 4, 5)
    assert expanded[7] == (1, 4, 5)
    np.testing.assert_allclose(expanded[6][:, 0, 0], [0.2, 0.3, -0.06])
    np.testing.assert_allclose(expanded[8][:, 0, 0], [0.4, 0.5, 0.20])


def test_pure_seasonal_ar_estimation_and_admissibility() -> None:
    rng = np.random.default_rng(17)
    seasonal_period = 4
    coefficient = 0.55
    data = np.zeros(180, dtype=float)
    innovations = rng.normal(scale=0.6, size=data.size)
    for time_index in range(seasonal_period, data.size):
        data[time_index] = (
            coefficient * data[time_index - seasonal_period] + innovations[time_index]
        )
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=seasonal_period,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=400,
    )

    result = model.fit(
        data[:, None],
        identity_weights(),
        start_params=np.array([0.25]),
        start_covariance=np.array([[0.5]]),
    )
    admissibility = model.admissibility()

    assert isinstance(result, SeasonalKalmanSTARIMAResult)
    assert result.seasonal_ar_parameters[0, 0] == pytest.approx(
        coefficient,
        abs=0.12,
    )
    assert result.ar_lags == (seasonal_period,)
    assert result.ar_spectral_radius < result.stability_limit
    assert admissibility.stationary
    assert admissibility.invertible
    assert admissibility.admissible
    assert "Jointly admissible" in admissibility.summary()


def test_seasonal_random_walk_forecast_restores_cycle_history() -> None:
    rng = np.random.default_rng(27)
    seasonal_period = 4
    seasonal_differences = rng.normal(loc=0.3, scale=0.4, size=120)
    initial_cycle = np.array([10.0, 11.0, 9.0, 12.0])
    values = list(initial_cycle)
    for difference in seasonal_differences:
        values.append(values[-seasonal_period] + difference)
    data = np.asarray(values, dtype=float)[:, None]
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=seasonal_period,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )

    result = model.fit(data, identity_weights())
    transformed_forecast = model.predict_differenced(6)[:, 0]
    original_forecast = model.predict(6)[:, 0]
    queue = list(data[-seasonal_period:, 0])
    expected: list[float] = []
    for difference in transformed_forecast:
        value = queue.pop(0) + float(difference)
        queue.append(value)
        expected.append(value)

    np.testing.assert_allclose(original_forecast, expected)
    assert result.n_transformed_rows == data.shape[0] - seasonal_period
    assert result.original_scale_forecast_available
    assert "transformed conditional ordinary-seasonal history" in result.summary()


def test_missing_level_propagates_through_seasonal_difference() -> None:
    rng = np.random.default_rng(37)
    data = rng.normal(size=(80, 2))
    data[20, 0] = np.nan
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="diagonal",
        include_intercept=True,
        max_iter=300,
    )

    result = model.fit(data, identity_weights(2))
    transformed = model.transformed_data_

    assert transformed is not None
    assert np.isnan(transformed[16, 0])
    assert np.isnan(transformed[20, 0])
    assert result.original_missing_cells == 1
    assert result.transformed_missing_cells == 2
    assert result.filter_result.n_observations == int(
        np.count_nonzero(np.isfinite(transformed))
    )


def test_missing_terminal_seasonal_history_blocks_only_original_scale() -> None:
    rng = np.random.default_rng(47)
    data = rng.normal(size=(90, 1))
    data[-2, 0] = np.nan
    model = SeasonalKalmanSTARIMA(
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

    result = model.fit(data, identity_weights())

    assert not result.original_scale_forecast_available
    assert np.all(np.isfinite(model.predict_differenced(4)))
    with pytest.raises(RuntimeError, match="ordinary and seasonal terminal anchors"):
        model.predict(4)


def test_new_data_filter_smooth_and_innovation_lengths_follow_combined_offset() -> None:
    rng = np.random.default_rng(57)
    training = np.cumsum(rng.normal(size=(150, 1)), axis=0)
    model = SeasonalKalmanSTARIMA(
        ar_order=1,
        integration_order=1,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=350,
    )
    model.fit(training, identity_weights())
    new_data = training[-40:].copy()
    new_data[18:20, 0] = np.nan

    filtered = model.filter(new_data)
    smoothed = model.smooth(new_data)
    innovations = model.smooth_innovation_disturbances(new_data)

    expected_rows = new_data.shape[0] - 1 - 4
    assert filtered.filtered_state.shape[0] == expected_rows
    assert smoothed.smoothed_state.shape[0] == expected_rows
    assert innovations.n_transitions == expected_rows - 1


def test_fitted_alignment_uses_observed_combined_lag_history() -> None:
    rng = np.random.default_rng(67)
    data = rng.normal(size=(100, 1))
    data[30, 0] = np.nan
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(data, identity_weights())

    transformed_fitted = model.fitted_differenced()
    original_fitted = model.fitted_original()

    assert transformed_fitted.shape == (data.shape[0] - 5, 1)
    assert original_fitted.shape == data.shape
    assert np.isnan(original_fitted[:5, 0]).all()
    assert np.isnan(original_fitted[31, 0])
    assert np.isnan(original_fitted[34, 0])


def test_validation_and_not_fitted_errors() -> None:
    with pytest.raises(ValueError, match="seasonal_period"):
        SeasonalKalmanSTARIMA(seasonal_period=0)
    with pytest.raises(ValueError, match="integration_order"):
        SeasonalKalmanSTARIMA(integration_order=-1)
    model = SeasonalKalmanSTARIMA(
        seasonal_integration_order=1,
        seasonal_period=4,
    )
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict_differenced()
    with pytest.raises(ValueError, match="must be smaller"):
        model.fit(np.ones((4, 1)), identity_weights())
    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(np.ones(20), identity_weights())
    with pytest.raises(ValueError, match="infinite"):
        model.fit(
            np.array([[1.0], [2.0], [np.inf], [4.0], [5.0], [6.0]]),
            identity_weights(),
        )
