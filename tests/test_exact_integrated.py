from __future__ import annotations

import numpy as np
import pytest

from pystarmax.exact_integrated import (
    ExactIntegratedStateSpace,
    build_exact_integrated_state_space,
    exact_integrated_filter,
    exact_integrated_loglikelihood,
)
from pystarmax.state_space import StateSpaceModel, kalman_filter


def white_noise_model(*, drift: float = 0.2, variance: float = 0.25) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[0.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([drift]),
        innovation_covariance=np.array([[variance]]),
        ar_order=0,
        ma_order=0,
    )


def test_first_order_integrated_likelihood_matches_increment_likelihood() -> None:
    drift = 0.2
    variance = 0.25
    levels = np.array([[10.0], [10.4], [10.3], [10.8]])
    result = exact_integrated_filter(
        levels,
        white_noise_model(drift=drift, variance=variance),
        integration_order=1,
    )

    expected = -0.5 * np.log(2.0 * np.pi)
    for increment in np.diff(levels[:, 0]):
        residual = increment - drift
        expected -= 0.5 * (
            np.log(2.0 * np.pi) + np.log(variance) + residual * residual / variance
        )

    assert result.log_likelihood == pytest.approx(expected)
    assert exact_integrated_loglikelihood(
        levels,
        white_noise_model(drift=drift, variance=variance),
        1,
    ) == pytest.approx(expected)
    np.testing.assert_allclose(result.filtered_observations, levels)
    assert result.initial_diffuse_rank == 1
    assert result.n_diffuse_observations == 1
    assert result.diffuse_end_time == 0


def test_second_order_integration_recovers_first_and_second_differences() -> None:
    drift = -0.1
    variance = 0.16
    second_differences = np.array([0.3, -0.2, 0.4, 0.1, -0.1])
    levels = []
    level = 5.0
    first_difference = 0.7
    for value in second_differences:
        first_difference += drift + value
        level += first_difference
        levels.append(level)
    observations = np.asarray(levels, dtype=float)[:, None]

    result = exact_integrated_filter(
        observations,
        white_noise_model(drift=drift, variance=variance),
        integration_order=2,
    )

    expected = -np.log(2.0 * np.pi)
    observed_second_differences = np.diff(observations[:, 0], n=2)
    for value in observed_second_differences:
        residual = value - drift
        expected -= 0.5 * (
            np.log(2.0 * np.pi) + np.log(variance) + residual * residual / variance
        )

    assert result.log_likelihood == pytest.approx(expected)
    np.testing.assert_array_equal(result.filtered_diffuse_rank[:3], [1, 0, 0])
    np.testing.assert_allclose(
        result.filtered_state[1:, 1],
        np.diff(observations[:, 0]),
    )
    np.testing.assert_allclose(result.filtered_observations, observations)
    assert result.initial_diffuse_rank == 2
    assert result.n_diffuse_observations == 2


def test_augmented_matrices_preserve_transformed_output_orientation() -> None:
    transformed = StateSpaceModel(
        transition=np.array([[0.5, -0.2], [1.0, 0.0]]),
        design=np.array([[1.0, 0.0]]),
        selection=np.array([[1.0], [0.0]]),
        state_intercept=np.array([0.3, 0.0]),
        innovation_covariance=np.array([[0.4]]),
        ar_order=2,
        ma_order=0,
    )

    specification = build_exact_integrated_state_space(transformed, 2)
    model = specification.model

    np.testing.assert_allclose(
        model.transition,
        np.array(
            [
                [1.0, 1.0, 0.5, -0.2],
                [0.0, 1.0, 0.5, -0.2],
                [0.0, 0.0, 0.5, -0.2],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(model.state_intercept, [0.3, 0.3, 0.3, 0.0])
    np.testing.assert_allclose(model.selection[:, 0], [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(model.design, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        specification.initial_diffuse_covariance,
        np.diag([1.0, 1.0, 0.0, 0.0]),
    )
    assert specification.n_diffuse_directions == 2
    assert model.ar_order == 4


def test_zero_integration_order_matches_stationary_filter() -> None:
    transformed = StateSpaceModel(
        transition=np.array([[0.4]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.1]),
        innovation_covariance=np.array([[0.3]]),
        ar_order=1,
        ma_order=0,
    )
    data = np.array([[0.2], [0.4], [0.1], [0.5]])
    specification = build_exact_integrated_state_space(transformed, 0)
    exact = specification.filter(data)
    ordinary = kalman_filter(data, transformed, initialization="stationary")

    assert isinstance(specification, ExactIntegratedStateSpace)
    assert specification.model is transformed
    assert specification.n_diffuse_directions == 0
    assert exact.n_diffuse_observations == 0
    np.testing.assert_allclose(exact.filtered_state, ordinary.filtered_state)
    np.testing.assert_allclose(
        exact.filtered_covariance,
        ordinary.filtered_covariance,
    )
    assert exact.log_likelihood == pytest.approx(ordinary.log_likelihood)


def test_missing_levels_delay_integrated_diffuse_completion() -> None:
    levels = np.array([[np.nan], [3.0], [3.4], [3.7]])
    specification = build_exact_integrated_state_space(white_noise_model(), 1)
    result = specification.filter(levels)

    np.testing.assert_array_equal(result.filtered_diffuse_rank, [1, 0, 0, 0])
    assert result.diffuse_end_time == 1
    assert result.n_observations == 3
    assert result.n_diffuse_observations == 1


def test_nonstationary_transformed_state_is_rejected() -> None:
    nonstationary = StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[0.2]]),
        ar_order=1,
        ma_order=0,
    )

    with pytest.raises(ValueError, match="must be stationary"):
        build_exact_integrated_state_space(nonstationary, 1)


def test_integrated_validation_errors() -> None:
    transformed = white_noise_model()
    with pytest.raises(TypeError, match="StateSpaceModel"):
        build_exact_integrated_state_space(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integration_order"):
        build_exact_integrated_state_space(transformed, -1)
