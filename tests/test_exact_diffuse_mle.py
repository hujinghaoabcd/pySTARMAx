from __future__ import annotations

import numpy as np
import pytest

from pystarmax import KalmanSTARMA, SpatialWeights
from pystarmax.exact_diffuse_mle import (
    ExactDiffuseKalmanSTARIMA,
    ExactDiffuseKalmanSTARIMAResult,
)


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def test_random_walk_mle_matches_increment_closed_form() -> None:
    rng = np.random.default_rng(2026)
    drift = 0.18
    scale = 0.45
    increments = drift + rng.normal(scale=scale, size=240)
    levels = np.cumsum(increments)[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=400,
    )

    result = model.fit(levels, identity_weights())
    observed_increments = np.diff(levels[:, 0])
    expected_drift = float(np.mean(observed_increments))
    expected_variance = float(np.mean((observed_increments - expected_drift) ** 2))

    assert isinstance(result, ExactDiffuseKalmanSTARIMAResult)
    assert result.order == (0, 1, 0)
    assert result.intercept == pytest.approx(expected_drift, abs=2e-5)
    assert result.innovation_covariance[0, 0] == pytest.approx(
        expected_variance,
        rel=2e-4,
        abs=2e-5,
    )
    assert result.n_diffuse_observations == 1
    assert result.filter_result.diffuse_end_time == 0
    assert result.log_likelihood == pytest.approx(result.filter_result.log_likelihood)
    assert result.aic == pytest.approx(-2.0 * result.log_likelihood + 4.0)
    assert result.bic == pytest.approx(
        -2.0 * result.log_likelihood + np.log(levels.size) * 2.0
    )
    assert result.converged


def test_second_order_integrated_mle_matches_second_difference_closed_form() -> None:
    rng = np.random.default_rng(17)
    drift = -0.08
    scale = 0.35
    second_differences = drift + rng.normal(scale=scale, size=220)
    first_difference = 0.5
    level = 8.0
    values = []
    for value in second_differences:
        first_difference += value
        level += first_difference
        values.append(level)
    levels = np.asarray(values, dtype=float)[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=2,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=400,
    )

    result = model.fit(levels, identity_weights())
    observed = np.diff(levels[:, 0], n=2)
    expected_drift = float(np.mean(observed))
    expected_variance = float(np.mean((observed - expected_drift) ** 2))

    assert result.intercept == pytest.approx(expected_drift, abs=3e-5)
    assert result.innovation_covariance[0, 0] == pytest.approx(
        expected_variance,
        rel=3e-4,
        abs=3e-5,
    )
    assert result.n_diffuse_observations == 2
    assert result.filter_result.initial_diffuse_rank == 2
    assert result.filter_result.final_diffuse_rank == 0


def test_zero_integration_order_matches_stationary_kalman_mle() -> None:
    rng = np.random.default_rng(27)
    phi = 0.45
    data = np.zeros(260, dtype=float)
    innovations = rng.normal(scale=0.5, size=data.size)
    for time_index in range(1, data.size):
        data[time_index] = phi * data[time_index - 1] + innovations[time_index]
    observations = data[:, None]
    weights = identity_weights()
    start_params = np.array([0.25])
    start_covariance = np.array([[0.3]])

    stationary = KalmanSTARMA(
        ar_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
        initialization="stationary",
        max_iter=400,
    )
    exact = ExactDiffuseKalmanSTARIMA(
        ar_order=1,
        integration_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=400,
    )
    stationary_result = stationary.fit(
        observations,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )
    exact_result = exact.fit(
        observations,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )

    np.testing.assert_allclose(
        exact_result.raw_optimizer_params,
        stationary_result.raw_optimizer_params,
        rtol=3e-5,
        atol=3e-5,
    )
    assert exact_result.log_likelihood == pytest.approx(
        stationary_result.log_likelihood,
        rel=3e-6,
        abs=3e-6,
    )
    assert exact_result.n_diffuse_observations == 0


def test_missing_initial_levels_delay_diffuse_completion() -> None:
    rng = np.random.default_rng(37)
    increments = 0.1 + rng.normal(scale=0.4, size=140)
    levels = np.cumsum(increments)[:, None]
    levels[:2, 0] = np.nan
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        include_intercept=True,
        max_iter=300,
    )

    result = model.fit(levels, identity_weights())

    assert result.filter_result.diffuse_end_time == 2
    assert result.n_observations == levels.shape[0] - 2
    assert result.n_diffuse_observations == 1
    assert result.filter_result.log_likelihood_contributions[0] == 0.0
    assert result.filter_result.log_likelihood_contributions[1] == 0.0


def test_fitted_filter_state_spaces_and_predictions_are_consistent() -> None:
    rng = np.random.default_rng(47)
    increments = 0.25 + rng.normal(scale=0.2, size=160)
    levels = np.cumsum(increments)[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        include_intercept=True,
        max_iter=300,
    )
    result = model.fit(levels, identity_weights())

    assert model.filter() is result.filter_result
    assert model.to_state_space() is result.integrated_state_space.model
    assert model.to_transformed_state_space() is result.transformed_state_space
    new_filter = model.filter(levels[-20:])
    assert new_filter.n_observations == 20
    np.testing.assert_allclose(
        model.predict_differenced(steps=4),
        np.full((4, 1), result.intercept),
    )
    expected_levels = levels[-1, 0] + result.intercept * np.arange(1, 5)
    np.testing.assert_allclose(
        model.predict(steps=4)[:, 0],
        expected_levels,
        rtol=2e-5,
        atol=2e-5,
    )
    assert "exact diffuse" in result.summary().lower()


def test_admissibility_and_immutable_result_arrays() -> None:
    rng = np.random.default_rng(57)
    data = np.zeros(180, dtype=float)
    for time_index in range(1, data.size):
        data[time_index] = 0.35 * data[time_index - 1] + rng.normal(scale=0.5)
    levels = np.cumsum(data)[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=1,
        integration_order=1,
        ma_order=0,
        include_intercept=False,
        max_iter=400,
    )
    result = model.fit(levels, identity_weights())
    diagnostics = model.admissibility()

    assert diagnostics.stationary
    assert result.stationary
    assert result.invertible
    assert result.admissible
    assert not result.params.flags.writeable
    assert not result.innovation_covariance.flags.writeable
    with pytest.raises(ValueError):
        result.params[0] = 0.0


def test_validation_and_not_fitted_errors() -> None:
    with pytest.raises(ValueError, match="diffuse_tolerance"):
        ExactDiffuseKalmanSTARIMA(diffuse_tolerance=0.0)
    with pytest.raises(ValueError, match="integration_order"):
        ExactDiffuseKalmanSTARIMA(integration_order=-1)

    model = ExactDiffuseKalmanSTARIMA()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.filter()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.admissibility()

    with pytest.raises(ValueError, match="at least three"):
        model.fit(np.ones((1, 1)), identity_weights())
    with pytest.raises(ValueError, match="start_params"):
        model.fit(
            np.arange(20.0)[:, None],
            identity_weights(),
            start_params=np.zeros(20),
        )
