from __future__ import annotations

import numpy as np
import pytest

from pystarmax import SpatialWeights
from pystarmax.exact_diffuse_disturbance_smoothing import (
    ExactDiffuseDisturbanceResult,
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_model import ExactDiffuseKalmanSTARIMA
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.exact_seasonal_integrated import (
    build_exact_seasonal_integrated_state_space,
)
from pystarmax.seasonal_exact_diffuse_model import (
    SeasonalExactDiffuseKalmanSTARIMA,
)
from pystarmax.state_space import StateSpaceModel


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def white_noise_model(
    variance: float = 0.5,
    *,
    n_locations: int = 1,
) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.zeros((n_locations, n_locations), dtype=float),
        design=np.eye(n_locations, dtype=float),
        selection=np.eye(n_locations, dtype=float),
        state_intercept=np.zeros(n_locations, dtype=float),
        innovation_covariance=variance * np.eye(n_locations, dtype=float),
        ar_order=0,
        ma_order=0,
    )


def seasonal_bridge_data() -> np.ndarray:
    return np.array([[0.0], [10.0], [np.nan], [np.nan], [4.0], [14.0]])


def test_seasonal_random_walk_bridge_matches_two_closed_form_bridges() -> None:
    variance = 0.5
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_model(variance),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    result = exact_diffuse_smoother(specification.filter(seasonal_bridge_data()))

    np.testing.assert_allclose(
        result.smoothed_observations[:, 0],
        [0.0, 10.0, 2.0, 12.0, 4.0, 14.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.smoothed_observation_covariance[:, 0, 0],
        [0.0, 0.0, variance / 2.0, variance / 2.0, 0.0, 0.0],
        atol=1e-12,
    )
    assert result.filter_result.initial_diffuse_rank == 2
    assert result.filter_result.final_diffuse_rank == 0
    assert result.maximum_filter_reconstruction_error < 1e-12


def test_seasonal_bridge_primitive_innovations_match_closed_form() -> None:
    variance = 0.5
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_model(variance),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    smoother = exact_diffuse_smoother(specification.filter(seasonal_bridge_data()))
    result = exact_diffuse_disturbance_smoother(smoother)

    np.testing.assert_allclose(
        result.innovation_mean[:, 0],
        [0.0, 2.0, 2.0, 2.0, 2.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.innovation_covariance[:, 0, 0],
        [variance, variance / 2.0, variance / 2.0, variance / 2.0, variance / 2.0],
        atol=1e-12,
    )
    selection = specification.model.selection
    np.testing.assert_allclose(
        result.state_disturbance_mean,
        result.innovation_mean @ selection.T,
        atol=1e-12,
    )
    for index in range(result.n_transitions):
        np.testing.assert_allclose(
            result.state_disturbance_covariance[index],
            selection @ result.innovation_covariance[index] @ selection.T,
            atol=1e-12,
        )


def test_partial_locations_are_smoothed_without_imputation() -> None:
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_model(variance=0.3, n_locations=2),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    data = np.array(
        [
            [0.0, np.nan],
            [10.0, 20.0],
            [np.nan, 22.0],
            [12.0, np.nan],
            [4.0, 24.0],
            [14.0, 26.0],
        ]
    )
    result = exact_diffuse_smoother(specification.filter(data))

    np.testing.assert_allclose(
        result.smoothed_observations[result.filter_result.observed_mask],
        data[result.filter_result.observed_mask],
        atol=1e-12,
    )
    for covariance in result.smoothed_covariance:
        assert float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-11
    assert result.maximum_filter_reconstruction_error < 1e-11


def test_fitted_facade_returns_training_state_and_disturbance_posteriors() -> None:
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=200,
    )
    fitted = model.fit(
        seasonal_bridge_data(),
        identity_weights(),
        start_covariance=np.array([[0.5]]),
    )

    smoothed = model.smooth()
    disturbances = model.smooth_innovation_disturbances()

    assert isinstance(smoothed, ExactDiffuseSmootherResult)
    assert isinstance(disturbances, ExactDiffuseDisturbanceResult)
    assert smoothed.filter_result is fitted.filter_result
    assert disturbances.smoother_result.filter_result is fitted.filter_result
    np.testing.assert_allclose(
        smoothed.smoothed_observations[:, 0],
        [0.0, 10.0, 2.0, 12.0, 4.0, 14.0],
        atol=2e-8,
    )
    assert disturbances.innovation_mean.shape == (5, 1)
    assert disturbances.state_disturbance_mean.shape == (
        5,
        fitted.integrated_state_space.model.state_dim,
    )


def test_fitted_facade_new_data_uses_fresh_diffuse_initialization() -> None:
    training = np.array([[0.0], [10.0], [2.0], [12.0], [4.0], [14.0]])
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        include_intercept=False,
        max_iter=200,
    )
    model.fit(training, identity_weights(), start_covariance=np.array([[0.5]]))
    new_data = np.array([[100.0], [200.0], [np.nan], [204.0], [108.0]])

    smoothed = model.smooth(new_data)

    assert smoothed.filter_result is not model.filter()
    assert smoothed.filter_result.initial_diffuse_rank == 2
    np.testing.assert_allclose(
        smoothed.smoothed_observations[smoothed.filter_result.observed_mask],
        new_data[smoothed.filter_result.observed_mask],
        atol=1e-10,
    )


def test_zero_seasonal_order_matches_ordinary_exact_diffuse_facade() -> None:
    rng = np.random.default_rng(2801)
    increments = 0.08 + rng.normal(scale=0.25, size=100)
    levels = np.cumsum(increments)[:, None]
    weights = identity_weights()
    start_params = np.array([0.05])
    start_covariance = np.array([[0.1]])
    ordinary = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        include_intercept=True,
        max_iter=300,
    )
    seasonal = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=2,
        include_intercept=True,
        max_iter=300,
    )
    ordinary.fit(
        levels,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )
    seasonal.fit(
        levels,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )

    ordinary_smoother = ordinary.smooth()
    seasonal_smoother = seasonal.smooth()
    ordinary_disturbance = ordinary.smooth_innovation_disturbances()
    seasonal_disturbance = seasonal.smooth_innovation_disturbances()

    np.testing.assert_allclose(
        seasonal_smoother.smoothed_state,
        ordinary_smoother.smoothed_state,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        seasonal_smoother.smoothed_covariance,
        ordinary_smoother.smoothed_covariance,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        seasonal_disturbance.innovation_mean,
        ordinary_disturbance.innovation_mean,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        seasonal_disturbance.innovation_covariance,
        ordinary_disturbance.innovation_covariance,
        rtol=2e-8,
        atol=2e-8,
    )


def test_facade_requires_fit_and_preserves_immutable_results() -> None:
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
        include_intercept=False,
    )
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.smooth()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.smooth_innovation_disturbances()

    model.fit(
        seasonal_bridge_data(),
        identity_weights(),
        start_covariance=np.array([[0.5]]),
    )
    smoothed = model.smooth()
    disturbances = model.smooth_innovation_disturbances()

    assert not smoothed.smoothed_state.flags.writeable
    assert not disturbances.innovation_mean.flags.writeable
    with pytest.raises(ValueError):
        smoothed.smoothed_state[0, 0] = 0.0
    with pytest.raises(ValueError, match="tolerance"):
        model.smooth(tolerance=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        model.smooth_innovation_disturbances(tolerance=0.0)
