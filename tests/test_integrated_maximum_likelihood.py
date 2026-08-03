from __future__ import annotations

import numpy as np
import pytest

from pystarmax import KalmanSTARIMA, KalmanSTARIMAResult, KalmanSTARMA, SpatialWeights


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_zero_integration_matches_stationary_kalman_model() -> None:
    rng = np.random.default_rng(2026)
    data = rng.normal(loc=0.25, scale=0.8, size=(100, 1))
    weights = identity_weights()
    start_params = np.array([0.1])
    start_covariance = np.array([[0.7]])

    stationary = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    integrated = KalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
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
    integrated_result = integrated.fit(
        data,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )

    np.testing.assert_allclose(integrated_result.params, stationary_result.params)
    np.testing.assert_allclose(
        integrated_result.core_result.innovation_covariance,
        stationary_result.innovation_covariance,
    )
    assert integrated_result.log_likelihood == pytest.approx(
        stationary_result.log_likelihood
    )
    np.testing.assert_allclose(integrated.predict(5), stationary.predict(5))
    np.testing.assert_allclose(
        integrated.filter().filtered_state,
        stationary.filter().filtered_state,
    )
    assert integrated_result.order == (0, 0, 0)
    assert integrated_result.original_scale_forecast_available


def test_random_walk_with_drift_is_fitted_on_first_differences() -> None:
    rng = np.random.default_rng(17)
    drift = 0.35
    increments = rng.normal(loc=drift, scale=0.5, size=180)
    data = np.concatenate([[4.0], 4.0 + np.cumsum(increments)])[:, None]
    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=400,
    )

    result = model.fit(data, identity_weights())
    differenced_forecast = model.predict_differenced(4)
    original_forecast = model.predict(4)

    assert isinstance(result, KalmanSTARIMAResult)
    assert result.order == (0, 1, 0)
    assert result.params[0] == pytest.approx(np.mean(increments), abs=2e-3)
    np.testing.assert_allclose(
        differenced_forecast[:, 0],
        np.repeat(result.params[0], 4),
        atol=2e-3,
    )
    np.testing.assert_allclose(
        original_forecast[:, 0],
        data[-1, 0] + np.cumsum(differenced_forecast[:, 0]),
    )
    assert result.n_original_rows == data.shape[0]
    assert result.n_differenced_rows == data.shape[0] - 1
    assert "conditional on initial history" in result.summary()


def test_second_order_inverse_differencing_uses_terminal_level_and_slope() -> None:
    second_differences = np.full(80, 0.2, dtype=float)
    first_differences = np.concatenate([[1.0], 1.0 + np.cumsum(second_differences)])
    levels = np.concatenate([[5.0], 5.0 + np.cumsum(first_differences)])[:, None]
    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=2,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(levels, identity_weights())

    differenced_forecast = model.predict_differenced(3)[:, 0]
    original_forecast = model.predict(3)[:, 0]
    last_level = levels[-1, 0]
    last_slope = levels[-1, 0] - levels[-2, 0]
    expected_slope = last_slope + np.cumsum(differenced_forecast)
    expected_level = last_level + np.cumsum(expected_slope)

    np.testing.assert_allclose(original_forecast, expected_level)
    assert model.order == (0, 2, 0)


def test_internal_missing_values_propagate_to_differences_without_imputation() -> None:
    rng = np.random.default_rng(31)
    data = np.cumsum(rng.normal(size=(90, 2)), axis=0)
    data[20, 0] = np.nan
    data[40:42, 1] = np.nan
    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="diagonal",
        include_intercept=True,
        max_iter=300,
    )

    result = model.fit(data, identity_weights(2))
    transformed = model.differenced_data_

    assert transformed is not None
    assert np.isnan(transformed[19, 0])
    assert np.isnan(transformed[20, 0])
    assert np.isnan(transformed[39:42, 1]).all()
    assert result.original_missing_cells == 3
    assert result.differenced_missing_cells == 5
    assert result.filter_result.observed_mask.shape == (data.shape[0] - 1, 2)
    assert result.filter_result.n_observations == int(
        np.count_nonzero(np.isfinite(transformed))
    )
    assert result.original_scale_forecast_available


def test_trailing_missing_anchor_blocks_only_original_scale_prediction() -> None:
    rng = np.random.default_rng(41)
    data = np.cumsum(rng.normal(size=(100, 1)), axis=0)
    data[-1, 0] = np.nan
    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )

    result = model.fit(data, identity_weights())

    assert not result.original_scale_forecast_available
    assert np.all(np.isfinite(model.predict_differenced(3)))
    with pytest.raises(RuntimeError, match="terminal differencing anchors"):
        model.predict(3)


def test_new_data_filter_smooth_and_innovation_results_use_differenced_scale() -> None:
    rng = np.random.default_rng(51)
    training = np.cumsum(rng.normal(scale=0.6, size=(130, 1)), axis=0)
    model = KalmanSTARIMA(
        ar_order=1,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=400,
    )
    model.fit(training, identity_weights())
    new_data = training[-30:].copy()
    new_data[10:13, 0] = np.nan

    filtered = model.filter(new_data)
    smoothed = model.smooth(new_data)
    innovations = model.smooth_innovation_disturbances(new_data)

    assert filtered.filtered_state.shape[0] == new_data.shape[0] - 1
    assert smoothed.smoothed_state.shape[0] == new_data.shape[0] - 1
    assert innovations.n_transitions == new_data.shape[0] - 2
    assert np.count_nonzero(~filtered.observed_mask[:, 0]) == 4


def test_fitted_original_alignment_and_missing_history() -> None:
    rng = np.random.default_rng(61)
    data = np.cumsum(rng.normal(size=(90, 1)), axis=0)
    data[30, 0] = np.nan
    model = KalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(data, identity_weights())

    fitted_differenced = model.fitted_differenced()
    fitted_original = model.fitted_original()

    assert fitted_differenced.shape == (data.shape[0] - 1, 1)
    assert fitted_original.shape == data.shape
    assert np.isnan(fitted_original[0, 0])
    assert np.isnan(fitted_original[31, 0])
    finite_index = 10
    assert fitted_original[finite_index, 0] == pytest.approx(
        data[finite_index - 1, 0] + fitted_differenced[finite_index - 1, 0]
    )


def test_validation_and_not_fitted_errors() -> None:
    with pytest.raises(ValueError, match="integration_order"):
        KalmanSTARIMA(integration_order=-1)
    model = KalmanSTARIMA(integration_order=1)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict_differenced()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.filter()
    with pytest.raises(ValueError, match="at least three"):
        model.fit(np.ones((2, 1)), identity_weights())
    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(np.ones(10), identity_weights())
    with pytest.raises(ValueError, match="infinite"):
        model.fit(np.array([[1.0], [2.0], [np.inf], [4.0]]), identity_weights())
