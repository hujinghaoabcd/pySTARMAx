from __future__ import annotations

import numpy as np
import pytest

from pystarmax.exact_integrated import build_exact_integrated_state_space
from pystarmax.exact_seasonal_integrated import (
    ExactSeasonalIntegratedStateSpace,
    build_exact_seasonal_integrated_state_space,
    exact_seasonal_integrated_filter,
    exact_seasonal_integrated_loglikelihood,
)
from pystarmax.state_space import StateSpaceModel


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


def test_seasonal_random_walk_likelihood_matches_seasonal_increments() -> None:
    drift = 0.2
    variance = 0.25
    period = 2
    levels = np.array([[10.0], [20.0], [10.4], [19.8], [10.9], [20.3]])

    result = exact_seasonal_integrated_filter(
        levels,
        white_noise_model(drift=drift, variance=variance),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=period,
    )

    expected = -0.5 * period * np.log(2.0 * np.pi)
    for increment in levels[period:, 0] - levels[:-period, 0]:
        residual = increment - drift
        expected -= 0.5 * (
            np.log(2.0 * np.pi) + np.log(variance) + residual * residual / variance
        )

    assert result.log_likelihood == pytest.approx(expected)
    assert exact_seasonal_integrated_loglikelihood(
        levels,
        white_noise_model(drift=drift, variance=variance),
        0,
        1,
        period,
    ) == pytest.approx(expected)
    np.testing.assert_allclose(result.filtered_observations, levels)
    np.testing.assert_array_equal(result.filtered_diffuse_rank[:3], [1, 0, 0])
    assert result.initial_diffuse_rank == period
    assert result.n_diffuse_observations == period
    assert result.diffuse_end_time == period - 1


def test_combined_difference_polynomial_builds_auditable_companion_state() -> None:
    transformed = StateSpaceModel(
        transition=np.array([[0.5, -0.2], [1.0, 0.0]]),
        design=np.array([[1.0, 0.0]]),
        selection=np.array([[1.0], [0.0]]),
        state_intercept=np.array([0.3, 0.0]),
        innovation_covariance=np.array([[0.4]]),
        ar_order=2,
        ma_order=0,
    )

    specification = build_exact_seasonal_integrated_state_space(
        transformed,
        ordinary_integration_order=1,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    model = specification.model

    np.testing.assert_allclose(
        specification.polynomial_coefficients,
        [1.0, -1.0, -1.0, 1.0],
    )
    np.testing.assert_allclose(
        model.transition,
        np.array(
            [
                [1.0, 1.0, -1.0, 0.5, -0.2],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, -0.2],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(model.state_intercept, [0.3, 0.0, 0.0, 0.3, 0.0])
    np.testing.assert_allclose(model.selection[:, 0], [1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(model.design, [[1.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        specification.initial_diffuse_covariance,
        np.diag([1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    assert specification.integration_degree == 3
    assert specification.n_diffuse_directions == 3
    assert model.ar_order == 5


def test_combined_state_recovers_the_complete_transformed_series() -> None:
    drift = -0.1
    variance = 0.16
    innovations = np.array([0.3, -0.2, 0.4, 0.1, -0.1, 0.2])
    history = [4.0, 4.7, 5.1]
    levels = list(history)
    for innovation in innovations:
        transformed = drift + innovation
        value = levels[-1] + levels[-2] - levels[-3] + transformed
        levels.append(value)
    observations = np.asarray(levels, dtype=float)[:, None]

    result = exact_seasonal_integrated_filter(
        observations,
        white_noise_model(drift=drift, variance=variance),
        ordinary_integration_order=1,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    transformed_observations = (
        observations[3:, 0]
        - observations[2:-1, 0]
        - observations[1:-2, 0]
        + observations[:-3, 0]
    )

    np.testing.assert_allclose(result.filtered_observations, observations)
    np.testing.assert_allclose(
        result.innovations[3:, 0],
        transformed_observations - drift,
    )
    np.testing.assert_allclose(result.finite_innovation_variance[3:, 0], variance)
    assert result.initial_diffuse_rank == 3
    assert result.n_diffuse_observations == 3
    assert result.final_diffuse_rank == 0


def test_zero_seasonal_order_reuses_ordinary_exact_diffuse_construction() -> None:
    transformed = StateSpaceModel(
        transition=np.array([[0.4]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.1]),
        innovation_covariance=np.array([[0.3]]),
        ar_order=1,
        ma_order=0,
    )
    data = np.array([[0.2], [0.4], [0.1], [0.5], [0.8]])
    ordinary = build_exact_integrated_state_space(transformed, 2)
    seasonal = build_exact_seasonal_integrated_state_space(
        transformed,
        ordinary_integration_order=2,
        seasonal_integration_order=0,
        seasonal_period=12,
    )

    assert isinstance(seasonal, ExactSeasonalIntegratedStateSpace)
    np.testing.assert_allclose(seasonal.model.transition, ordinary.model.transition)
    np.testing.assert_allclose(seasonal.model.design, ordinary.model.design)
    np.testing.assert_allclose(seasonal.model.selection, ordinary.model.selection)
    np.testing.assert_allclose(
        seasonal.model.state_intercept,
        ordinary.model.state_intercept,
    )
    np.testing.assert_allclose(seasonal.initial_state, ordinary.initial_state)
    np.testing.assert_allclose(
        seasonal.initial_covariance,
        ordinary.initial_covariance,
    )
    np.testing.assert_allclose(
        seasonal.initial_diffuse_covariance,
        ordinary.initial_diffuse_covariance,
    )
    assert seasonal.filter(data).log_likelihood == pytest.approx(
        ordinary.filter(data).log_likelihood
    )


def test_missing_observations_delay_seasonal_diffuse_completion() -> None:
    levels = np.array([[np.nan], [3.0], [3.4], [3.7], [4.1]])
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_model(),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    result = specification.filter(levels)

    np.testing.assert_array_equal(result.filtered_diffuse_rank, [2, 1, 0, 0, 0])
    assert result.diffuse_end_time == 2
    assert result.n_observations == 4
    assert result.n_diffuse_observations == 2


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
        build_exact_seasonal_integrated_state_space(
            nonstationary,
            ordinary_integration_order=0,
            seasonal_integration_order=1,
            seasonal_period=4,
        )


def test_seasonal_integrated_validation_errors() -> None:
    transformed = white_noise_model()
    with pytest.raises(TypeError, match="StateSpaceModel"):
        build_exact_seasonal_integrated_state_space(
            object(), 0, 1, 4  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="ordinary_integration_order"):
        build_exact_seasonal_integrated_state_space(transformed, -1, 1, 4)
    with pytest.raises(ValueError, match="seasonal_integration_order"):
        build_exact_seasonal_integrated_state_space(transformed, 0, -1, 4)
    with pytest.raises(ValueError, match="seasonal_period"):
        build_exact_seasonal_integrated_state_space(transformed, 0, 1, 0)
