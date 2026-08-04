from __future__ import annotations

import numpy as np
import pytest

from pystarmax.exact_diffuse import exact_diffuse_filter
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.exact_integrated import build_exact_integrated_state_space
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


def stable_ar_model(phi: float = 0.45, variance: float = 0.3) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[phi]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.1]),
        innovation_covariance=np.array([[variance]]),
        ar_order=1,
        ma_order=0,
    )


def local_linear_trend_model(variance: float = 0.2) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[1.0, 1.0], [0.0, 1.0]]),
        design=np.array([[1.0, 0.0]]),
        selection=np.array([[1.0], [0.0]]),
        state_intercept=np.zeros(2),
        innovation_covariance=np.array([[variance]]),
        ar_order=2,
        ma_order=0,
    )


def test_random_walk_missing_bridge_matches_closed_form() -> None:
    variance = 0.5
    data = np.array([[1.0], [np.nan], [3.0]])
    filtered = exact_diffuse_filter(data, local_level_model(variance))
    result = exact_diffuse_smoother(filtered)

    np.testing.assert_allclose(result.smoothed_state[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        result.smoothed_covariance[:, 0, 0],
        [0.0, variance / 2.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.smoothed_observations[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        result.smoothed_observation_covariance[:, 0, 0],
        [0.0, variance / 2.0, 0.0],
        atol=1e-12,
    )
    assert result.diffuse_end_time == 0
    assert result.final_diffuse_rank == 0
    assert result.maximum_filter_reconstruction_error < 1e-12


def test_leading_missing_random_walk_is_smoothed_from_later_level() -> None:
    variance = 0.4
    data = np.array([[np.nan], [np.nan], [3.0]])
    result = exact_diffuse_smoother(
        exact_diffuse_filter(data, local_level_model(variance))
    )

    np.testing.assert_allclose(result.smoothed_state[:, 0], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(
        result.smoothed_covariance[:, 0, 0],
        [2.0 * variance, variance, 0.0],
        atol=1e-12,
    )
    assert result.filter_result.diffuse_end_time == 2


def test_zero_diffuse_covariance_matches_stationary_rts_smoother() -> None:
    rng = np.random.default_rng(2026)
    model = stable_ar_model()
    data = rng.normal(size=(80, 1))
    data[20:24, 0] = np.nan
    specification = build_exact_integrated_state_space(model, 0)
    exact_filter = specification.filter(data)
    exact = exact_diffuse_smoother(exact_filter)
    ordinary_filter = kalman_filter(data, model, initialization="stationary")
    ordinary = kalman_smoother(ordinary_filter)

    assert exact.filter_result.n_diffuse_observations == 0
    np.testing.assert_allclose(
        exact.smoothed_state,
        ordinary.smoothed_state,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        exact.smoothed_covariance,
        ordinary.smoothed_covariance,
        rtol=2e-9,
        atol=2e-9,
    )


def test_exact_diffuse_smoother_matches_large_variance_limit() -> None:
    model = local_linear_trend_model()
    data = np.array([[1.0], [1.4], [np.nan], [2.6], [3.1], [3.9]])
    finite = np.array([[0.2, 0.03], [0.03, 0.15]])
    diffuse = np.eye(2)
    exact_filter = exact_diffuse_filter(
        data,
        model,
        initial_covariance=finite,
        initial_diffuse_covariance=diffuse,
    )
    exact = exact_diffuse_smoother(exact_filter)

    scale = 1e8
    approximate_filter = kalman_filter(
        data,
        model,
        initialization="known",
        initial_state=np.zeros(2),
        initial_covariance=finite + scale * diffuse,
    )
    approximate = kalman_smoother(approximate_filter)

    np.testing.assert_allclose(
        exact.smoothed_state,
        approximate.smoothed_state,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        exact.smoothed_covariance,
        approximate.smoothed_covariance,
        rtol=3e-4,
        atol=3e-4,
    )


def test_partial_locations_and_observed_cells() -> None:
    model = StateSpaceModel(
        transition=np.eye(2),
        design=np.eye(2),
        selection=np.eye(2),
        state_intercept=np.zeros(2),
        innovation_covariance=np.diag([0.2, 0.3]),
        ar_order=1,
        ma_order=0,
    )
    data = np.array(
        [
            [1.0, np.nan],
            [np.nan, 2.0],
            [2.0, 2.5],
        ]
    )
    result = exact_diffuse_smoother(exact_diffuse_filter(data, model))

    np.testing.assert_allclose(
        result.smoothed_observations[result.filter_result.observed_mask],
        data[result.filter_result.observed_mask],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.smoothed_observation_covariance[-1],
        np.zeros((2, 2)),
        atol=1e-12,
    )
    assert result.maximum_filter_reconstruction_error < 1e-12


def test_result_arrays_are_immutable_and_covariances_are_psd() -> None:
    result = exact_diffuse_smoother(
        exact_diffuse_filter(
            np.array([[1.0], [np.nan], [2.0], [2.4]]),
            local_level_model(),
        )
    )

    assert isinstance(result, ExactDiffuseSmootherResult)
    assert not result.smoothed_state.flags.writeable
    assert not result.smoothed_covariance.flags.writeable
    assert not result.scaled_smoothed_diffuse_estimator.flags.writeable
    for covariance in result.smoothed_covariance:
        assert float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-12
    with pytest.raises(ValueError):
        result.smoothed_state[0, 0] = 0.0


def test_validation_errors() -> None:
    filtered = exact_diffuse_filter(np.array([[1.0], [2.0]]), local_level_model())
    with pytest.raises(TypeError, match="ExactDiffuseFilterResult"):
        exact_diffuse_smoother(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        exact_diffuse_smoother(filtered, tolerance=0.0)
