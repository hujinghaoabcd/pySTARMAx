from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    ExactDiffuseKalmanSTARIMA,
    ExactDiffuseLagOneCovarianceResult,
    SeasonalExactDiffuseKalmanSTARIMA,
    SpatialWeights,
    exact_diffuse_lag_one_covariance,
)
from pystarmax.exact_diffuse import exact_diffuse_filter
from pystarmax.exact_diffuse_simulation_smoothing import (
    exact_diffuse_simulation_smoother,
)
from pystarmax.exact_seasonal_integrated import (
    build_exact_seasonal_integrated_state_space,
)
from pystarmax.seasonal_exact_diffuse_simulation_smoothing import (
    seasonal_exact_diffuse_simulation_smoother,
)
from pystarmax.smoothing import kalman_smoother
from pystarmax.state_space import StateSpaceModel, kalman_filter


def local_level_model(variance: float = 0.5) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[variance]]),
        ar_order=1,
        ma_order=0,
    )


def non_symmetric_stable_model() -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[0.40, 0.25], [0.00, 0.30]]),
        design=np.eye(2, dtype=float),
        selection=np.eye(2, dtype=float),
        state_intercept=np.array([0.10, -0.05]),
        innovation_covariance=np.array([[0.30, 0.08], [0.08, 0.40]]),
        ar_order=1,
        ma_order=0,
    )


def white_noise_model(variance: float = 0.5) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.zeros((1, 1), dtype=float),
        design=np.eye(1, dtype=float),
        selection=np.eye(1, dtype=float),
        state_intercept=np.zeros(1, dtype=float),
        innovation_covariance=np.array([[variance]]),
        ar_order=0,
        ma_order=0,
    )


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def empirical_cross_covariance(
    paths: np.ndarray,
    left_time: int,
    right_time: int,
) -> np.ndarray:
    left = paths[:, left_time] - paths[:, left_time].mean(axis=0)
    right = paths[:, right_time] - paths[:, right_time].mean(axis=0)
    return left.T @ right / paths.shape[0]


def test_diffuse_random_walk_final_anchor_has_closed_form_lag_covariance() -> None:
    variance = 0.4
    data = np.array([[np.nan], [np.nan], [2.0]])
    result = exact_diffuse_lag_one_covariance(
        exact_diffuse_filter(data, local_level_model(variance))
    )

    np.testing.assert_allclose(
        result.smoother_result.smoothed_state[:, 0],
        np.array([2.0, 2.0, 2.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.smoother_result.smoothed_covariance[:, 0, 0],
        np.array([2.0 * variance, variance, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.lag_one_covariance[:, 0, 0],
        np.array([variance, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.observation_lag_one_covariance,
        result.lag_one_covariance,
        atol=0.0,
    )
    np.testing.assert_allclose(result.state_disturbance_mean, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.state_disturbance_covariance[:, 0, 0],
        variance,
        atol=1e-12,
    )
    assert result.diffuse_rank == 1
    assert result.maximum_mean_discrepancy < 1e-12
    assert result.maximum_marginal_covariance_discrepancy < 1e-12
    assert result.maximum_state_disturbance_discrepancy < 1e-12


def test_zero_diffuse_case_reduces_to_rts_with_non_symmetric_transition() -> None:
    model = non_symmetric_stable_model()
    data = np.array(
        [
            [0.4, -0.1],
            [np.nan, 0.2],
            [0.8, np.nan],
            [0.5, -0.4],
            [np.nan, np.nan],
            [0.2, -0.2],
            [0.1, np.nan],
        ]
    )
    initial_state = np.array([0.2, -0.1])
    initial_covariance = np.array([[0.8, 0.2], [0.2, 0.6]])
    exact_filtered = exact_diffuse_filter(
        data,
        model,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        initial_diffuse_covariance=np.zeros((2, 2), dtype=float),
    )
    ordinary_filtered = kalman_filter(
        data,
        model,
        initialization="known",
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )
    exact = exact_diffuse_lag_one_covariance(exact_filtered)
    ordinary = kalman_smoother(ordinary_filtered)

    np.testing.assert_allclose(
        exact.smoother_result.smoothed_state,
        ordinary.smoothed_state,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        exact.smoother_result.smoothed_covariance,
        ordinary.smoothed_covariance,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        exact.lag_one_covariance,
        ordinary.lag_one_covariance,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        exact.state_disturbance_mean,
        ordinary.state_disturbance_mean,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        exact.state_disturbance_covariance,
        ordinary.state_disturbance_covariance,
        atol=2e-10,
    )


def test_conditional_paths_independently_recover_diffuse_lag_covariance() -> None:
    variance = 0.5
    filtered = exact_diffuse_filter(
        np.array([[np.nan], [np.nan], [3.0]]),
        local_level_model(variance),
    )
    exact = exact_diffuse_lag_one_covariance(filtered)
    simulated = exact_diffuse_simulation_smoother(
        filtered,
        n_simulations=16000,
        random_state=2026,
    )

    for time_index in range(exact.n_transitions):
        empirical = empirical_cross_covariance(
            simulated.state_paths,
            time_index,
            time_index + 1,
        )
        np.testing.assert_allclose(
            empirical,
            exact.lag_one_covariance[time_index],
            atol=0.012,
            rtol=0.04,
        )


def test_seasonal_augmented_state_matches_conditional_path_cross_covariance() -> None:
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_model(0.4),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    data = np.array([[0.0], [10.0], [np.nan], [np.nan], [4.0], [14.0]])
    filtered = specification.filter(data)
    exact = exact_diffuse_lag_one_covariance(filtered)
    simulated = seasonal_exact_diffuse_simulation_smoother(
        filtered,
        specification,
        n_simulations=12000,
        random_state=71,
    )

    for time_index in range(exact.n_transitions):
        empirical = empirical_cross_covariance(
            simulated.state_paths,
            time_index,
            time_index + 1,
        )
        np.testing.assert_allclose(
            empirical,
            exact.lag_one_covariance[time_index],
            atol=0.015,
            rtol=0.06,
        )


def test_fitted_facades_reuse_training_filter_and_refilter_new_data() -> None:
    rng = np.random.default_rng(77)
    levels = np.cumsum(0.1 + rng.normal(scale=0.25, size=70))[:, None]
    ordinary = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    ordinary_fit = ordinary.fit(levels, identity_weights())
    training = ordinary.smooth_lag_one_covariance()
    assert isinstance(training, ExactDiffuseLagOneCovarianceResult)
    assert training.filter_result is ordinary_fit.filter_result

    new_levels = levels[-12:].copy()
    new_levels[5, 0] = np.nan
    new_result = ordinary.smooth_lag_one_covariance(new_levels)
    assert new_result.filter_result is not ordinary_fit.filter_result
    assert new_result.lag_one_covariance.shape[0] == new_levels.shape[0] - 1

    seasonal_data = np.empty((70, 1), dtype=float)
    seasonal_data[:2, 0] = (0.0, 10.0)
    seasonal_innovations = rng.normal(scale=0.25, size=70)
    for time_index in range(2, 70):
        seasonal_data[time_index, 0] = (
            seasonal_data[time_index - 2, 0]
            + seasonal_innovations[time_index]
        )
    seasonal = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=300,
    )
    seasonal_fit = seasonal.fit(seasonal_data, identity_weights())
    seasonal_result = seasonal.smooth_lag_one_covariance()
    assert seasonal_result.filter_result is seasonal_fit.filter_result
    assert seasonal_result.lag_one_covariance.shape[0] == seasonal_data.shape[0] - 1


def test_single_observation_empty_transition_contract_and_immutability() -> None:
    result = exact_diffuse_lag_one_covariance(
        exact_diffuse_filter(np.array([[1.0]]), local_level_model())
    )

    assert result.n_transitions == 0
    assert result.lag_one_covariance.shape == (0, 1, 1)
    assert result.state_disturbance_covariance.shape == (0, 1, 1)
    assert not result.lag_one_covariance.flags.writeable
    assert not result.state_disturbance_covariance.flags.writeable
    with pytest.raises(ValueError):
        result.lag_one_covariance.setflags(write=True)


def test_unresolved_diffuse_rank_and_invalid_arguments_are_rejected() -> None:
    unresolved = exact_diffuse_filter(
        np.full((4, 1), np.nan),
        local_level_model(),
    )
    with pytest.raises(RuntimeError, match="every diffuse direction"):
        exact_diffuse_lag_one_covariance(unresolved)
    resolved = exact_diffuse_filter(
        np.array([[1.0], [np.nan], [2.0]]),
        local_level_model(),
    )
    with pytest.raises(TypeError, match="ExactDiffuseFilterResult"):
        exact_diffuse_lag_one_covariance(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rcond"):
        exact_diffuse_lag_one_covariance(resolved, rcond=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        exact_diffuse_lag_one_covariance(resolved, tolerance=0.0)
