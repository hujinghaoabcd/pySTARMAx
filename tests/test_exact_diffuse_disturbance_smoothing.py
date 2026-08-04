from __future__ import annotations

import numpy as np
import pytest

from pystarmax.exact_diffuse import exact_diffuse_filter
from pystarmax.exact_diffuse_disturbance_smoothing import (
    ExactDiffuseDisturbanceResult,
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_smoothing import exact_diffuse_smoother
from pystarmax.exact_integrated import build_exact_integrated_state_space
from pystarmax.innovation_smoothing import innovation_disturbance_smoother
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


def test_fully_observed_random_walk_recovers_exact_innovations() -> None:
    data = np.array([[1.0], [2.0], [4.0]])
    smoother = exact_diffuse_smoother(
        exact_diffuse_filter(data, local_level_model(variance=0.5))
    )
    result = exact_diffuse_disturbance_smoother(smoother)

    np.testing.assert_allclose(result.innovation_mean[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(
        result.innovation_covariance[:, 0, 0],
        [0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.state_disturbance_mean[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(
        result.state_disturbance_covariance[:, 0, 0],
        [0.0, 0.0],
        atol=1e-12,
    )
    assert result.n_transitions == 2
    assert result.innovation_dim == 1


def test_random_walk_bridge_has_conditional_increment_moments() -> None:
    variance = 0.5
    data = np.array([[1.0], [np.nan], [3.0]])
    result = exact_diffuse_disturbance_smoother(
        exact_diffuse_smoother(
            exact_diffuse_filter(data, local_level_model(variance))
        )
    )

    np.testing.assert_allclose(result.innovation_mean[:, 0], [1.0, 1.0])
    np.testing.assert_allclose(
        result.innovation_covariance[:, 0, 0],
        [variance / 2.0, variance / 2.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.state_disturbance_covariance,
        result.innovation_covariance,
    )


def test_leading_diffuse_level_leaves_future_innovations_unresolved() -> None:
    variance = 0.4
    data = np.array([[np.nan], [np.nan], [3.0]])
    result = exact_diffuse_disturbance_smoother(
        exact_diffuse_smoother(
            exact_diffuse_filter(data, local_level_model(variance))
        )
    )

    np.testing.assert_allclose(result.innovation_mean[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(
        result.innovation_covariance[:, 0, 0],
        [variance, variance],
        atol=1e-12,
    )


def test_zero_diffuse_case_matches_ordinary_innovation_smoother() -> None:
    rng = np.random.default_rng(2026)
    model = stable_ar_model()
    data = rng.normal(size=(100, 1))
    data[30:35, 0] = np.nan

    exact_specification = build_exact_integrated_state_space(model, 0)
    exact_smoother = exact_diffuse_smoother(exact_specification.filter(data))
    exact = exact_diffuse_disturbance_smoother(exact_smoother)

    ordinary_filter = kalman_filter(data, model, initialization="stationary")
    ordinary_smoother = kalman_smoother(ordinary_filter)
    ordinary = innovation_disturbance_smoother(ordinary_smoother)

    np.testing.assert_allclose(
        exact.innovation_mean,
        ordinary.innovation_mean,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        exact.innovation_covariance,
        ordinary.innovation_covariance,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        exact.state_disturbance_mean,
        ordinary_smoother.state_disturbance_mean,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        exact.state_disturbance_covariance,
        ordinary_smoother.state_disturbance_covariance,
        rtol=2e-8,
        atol=2e-8,
    )


def test_selection_null_space_retains_primitive_innovation_variance() -> None:
    first_variance = 0.6
    unresolved_variance = 2.5
    model = StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0], [0.0]]),
        selection=np.array([[1.0, 0.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.diag([first_variance, unresolved_variance]),
        ar_order=1,
        ma_order=0,
    )
    data = np.array(
        [
            [0.0, np.nan],
            [1.0, np.nan],
            [3.0, np.nan],
        ]
    )
    result = exact_diffuse_disturbance_smoother(
        exact_diffuse_smoother(exact_diffuse_filter(data, model))
    )

    np.testing.assert_allclose(result.innovation_mean[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(result.innovation_mean[:, 1], [0.0, 0.0])
    np.testing.assert_allclose(
        result.innovation_covariance[:, 0, 0],
        [0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.innovation_covariance[:, 1, 1],
        [unresolved_variance, unresolved_variance],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.innovation_covariance[:, 0, 1],
        [0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.state_disturbance_mean[:, 0],
        result.innovation_mean[:, 0],
    )
    np.testing.assert_allclose(
        result.state_disturbance_covariance[:, 0, 0],
        [0.0, 0.0],
        atol=1e-12,
    )
    assert result.innovation_dim == 2


def test_single_observation_has_empty_disturbance_arrays() -> None:
    result = exact_diffuse_disturbance_smoother(
        exact_diffuse_smoother(
            exact_diffuse_filter(np.array([[1.0]]), local_level_model())
        )
    )

    assert result.innovation_mean.shape == (0, 1)
    assert result.innovation_covariance.shape == (0, 1, 1)
    assert result.state_disturbance_mean.shape == (0, 1)
    assert result.state_disturbance_covariance.shape == (0, 1, 1)
    assert result.n_transitions == 0


def test_result_arrays_are_immutable_and_covariances_are_psd() -> None:
    result = exact_diffuse_disturbance_smoother(
        exact_diffuse_smoother(
            exact_diffuse_filter(
                np.array([[1.0], [np.nan], [2.5], [3.0]]),
                local_level_model(),
            )
        )
    )

    assert isinstance(result, ExactDiffuseDisturbanceResult)
    assert not result.innovation_mean.flags.writeable
    assert not result.innovation_covariance.flags.writeable
    assert not result.state_disturbance_covariance.flags.writeable
    for covariance in result.innovation_covariance:
        assert float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-12
    with pytest.raises(ValueError):
        result.innovation_mean[0, 0] = 0.0


def test_validation_errors() -> None:
    smoother = exact_diffuse_smoother(
        exact_diffuse_filter(np.array([[1.0], [2.0]]), local_level_model())
    )
    with pytest.raises(TypeError, match="ExactDiffuseSmootherResult"):
        exact_diffuse_disturbance_smoother(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        exact_diffuse_disturbance_smoother(smoother, tolerance=0.0)
