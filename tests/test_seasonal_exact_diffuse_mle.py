from __future__ import annotations

import numpy as np
import pytest

from pystarmax import SpatialWeights
from pystarmax.exact_diffuse_mle import ExactDiffuseKalmanSTARIMA
from pystarmax.seasonal_exact_diffuse_mle import (
    SeasonalExactDiffuseKalmanSTARIMA,
    SeasonalExactDiffuseKalmanSTARIMAResult,
)


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def seasonal_integrate(values: np.ndarray, *, period: int) -> np.ndarray:
    history = list(np.linspace(4.0, 5.0, period))
    for value in values:
        history.append(history[-period] + float(value))
    return np.asarray(history, dtype=float)[:, None]


def test_seasonal_random_walk_mle_matches_increment_closed_form() -> None:
    rng = np.random.default_rng(2027)
    period = 2
    drift = 0.16
    scale = 0.38
    transformed = drift + rng.normal(scale=scale, size=220)
    levels = seasonal_integrate(transformed, period=period)
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=period,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=400,
    )

    result = model.fit(levels, identity_weights())
    observed = levels[period:, 0] - levels[:-period, 0]
    expected_drift = float(np.mean(observed))
    expected_variance = float(np.mean((observed - expected_drift) ** 2))
    expected_log_likelihood = -0.5 * period * np.log(2.0 * np.pi)
    expected_log_likelihood -= 0.5 * observed.size * (
        np.log(2.0 * np.pi) + np.log(expected_variance) + 1.0
    )

    assert isinstance(result, SeasonalExactDiffuseKalmanSTARIMAResult)
    assert result.order == (0, 0, 0)
    assert result.seasonal_order == (0, 1, 0, period)
    assert result.intercept == pytest.approx(expected_drift, abs=3e-5)
    assert result.innovation_covariance[0, 0] == pytest.approx(
        expected_variance,
        rel=3e-4,
        abs=3e-5,
    )
    assert result.log_likelihood == pytest.approx(
        expected_log_likelihood,
        rel=3e-5,
        abs=3e-5,
    )
    assert result.n_diffuse_observations == period
    assert result.filter_result.diffuse_end_time == period - 1
    assert result.n_params == 2
    assert result.aic == pytest.approx(-2.0 * result.log_likelihood + 4.0)
    assert result.converged


def test_combined_ordinary_and_seasonal_integration_mle() -> None:
    rng = np.random.default_rng(37)
    drift = -0.07
    scale = 0.29
    transformed = drift + rng.normal(scale=scale, size=200)
    levels = [3.0, 3.4, 4.1]
    for value in transformed:
        levels.append(levels[-1] + levels[-2] - levels[-3] + float(value))
    observations = np.asarray(levels, dtype=float)[:, None]
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        include_intercept=True,
        max_iter=400,
    )

    result = model.fit(observations, identity_weights())
    observed = (
        observations[3:, 0]
        - observations[2:-1, 0]
        - observations[1:-2, 0]
        + observations[:-3, 0]
    )

    assert result.intercept == pytest.approx(np.mean(observed), abs=4e-5)
    assert result.innovation_covariance[0, 0] == pytest.approx(
        np.mean((observed - np.mean(observed)) ** 2),
        rel=4e-4,
        abs=4e-5,
    )
    assert result.integrated_state_space.integration_degree == 3
    assert result.filter_result.initial_diffuse_rank == 3
    assert result.n_diffuse_observations == 3
    assert result.filter_result.final_diffuse_rank == 0


def test_factor_parameter_count_and_cross_lag_expansion() -> None:
    rng = np.random.default_rng(47)
    phi = 0.22
    seasonal_phi = 0.18
    transformed = np.zeros(260, dtype=float)
    innovations = rng.normal(scale=0.35, size=transformed.size)
    for time_index in range(3, transformed.size):
        transformed[time_index] = (
            phi * transformed[time_index - 1]
            + seasonal_phi * transformed[time_index - 2]
            - seasonal_phi * phi * transformed[time_index - 3]
            + innovations[time_index]
        )
    levels = seasonal_integrate(transformed, period=2)
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=1,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=500,
    )

    result = model.fit(
        levels,
        identity_weights(),
        start_params=np.array([0.15, 0.12]),
        start_covariance=np.array([[0.2]]),
    )

    assert result.parameter_names == ("ar.t1.W0", "sar.t2.W0")
    assert result.ar_lags == (1, 2, 3)
    assert result.n_params == 3
    assert result.raw_optimizer_params.size == 3
    np.testing.assert_allclose(
        result.ar_matrices[2, 0, 0],
        -result.ar_parameters[0, 0]
        * result.seasonal_ar_parameters[0, 0],
    )
    assert result.stationary
    assert result.admissible


def test_zero_seasonal_order_matches_ordinary_exact_diffuse_mle() -> None:
    rng = np.random.default_rng(57)
    increments = 0.12 + rng.normal(scale=0.31, size=190)
    levels = np.cumsum(increments)[:, None]
    weights = identity_weights()
    start_params = np.array([0.08])
    start_covariance = np.array([[0.15]])
    ordinary = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        include_intercept=True,
        max_iter=400,
    )
    seasonal = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=12,
        include_intercept=True,
        max_iter=400,
    )

    ordinary_result = ordinary.fit(
        levels,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )
    seasonal_result = seasonal.fit(
        levels,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )

    np.testing.assert_allclose(
        seasonal_result.raw_optimizer_params,
        ordinary_result.raw_optimizer_params,
        rtol=3e-5,
        atol=3e-5,
    )
    assert seasonal_result.log_likelihood == pytest.approx(
        ordinary_result.log_likelihood,
        rel=3e-6,
        abs=3e-6,
    )
    assert seasonal_result.n_diffuse_observations == 1


def test_missing_initial_levels_delay_diffuse_completion() -> None:
    rng = np.random.default_rng(67)
    transformed = 0.08 + rng.normal(scale=0.25, size=150)
    levels = seasonal_integrate(transformed, period=2)
    levels[0, 0] = np.nan
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
        include_intercept=True,
        max_iter=300,
    )

    result = model.fit(levels, identity_weights())

    assert result.filter_result.diffuse_end_time == 2
    assert result.n_observations == levels.size - 1
    assert result.n_diffuse_observations == 2
    assert result.original_missing_cells == 1


def test_state_spaces_predictions_and_result_immutability() -> None:
    rng = np.random.default_rng(77)
    period = 2
    transformed = 0.2 + rng.normal(scale=0.18, size=170)
    levels = seasonal_integrate(transformed, period=period)
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        seasonal_integration_order=1,
        seasonal_period=period,
        include_intercept=True,
        max_iter=300,
    )
    result = model.fit(levels, identity_weights())

    assert model.filter() is result.filter_result
    assert model.to_state_space() is result.integrated_state_space.model
    assert model.to_transformed_state_space() is result.transformed_state_space
    np.testing.assert_allclose(
        model.predict_differenced(steps=4),
        np.full((4, 1), result.intercept),
    )
    expected = list(levels[-period:, 0])
    for _ in range(4):
        expected.append(expected[-period] + result.intercept)
    np.testing.assert_allclose(
        model.predict(steps=4)[:, 0],
        expected[period:],
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(model.fitted_original(), result.filter_result.predicted_observations)
    assert model.admissibility().admissible
    assert not result.params.flags.writeable
    assert not result.ar_matrices.flags.writeable
    assert not result.innovation_covariance.flags.writeable
    with pytest.raises(ValueError):
        result.innovation_covariance[0, 0] = 1.0
    assert "seasonal exact diffuse" in result.summary().lower()


def test_validation_and_not_fitted_errors() -> None:
    with pytest.raises(ValueError, match="diffuse_tolerance"):
        SeasonalExactDiffuseKalmanSTARIMA(diffuse_tolerance=0.0)
    with pytest.raises(ValueError, match="seasonal_period"):
        SeasonalExactDiffuseKalmanSTARIMA(seasonal_period=0)

    model = SeasonalExactDiffuseKalmanSTARIMA()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.filter()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.admissibility()
    with pytest.raises(ValueError, match="ordinary order plus seasonal order"):
        model.fit(np.ones((4, 1)), identity_weights())
    with pytest.raises(ValueError, match="start_params"):
        model.fit(
            np.arange(30.0)[:, None],
            identity_weights(),
            start_params=np.zeros(20),
        )
