from __future__ import annotations

import numpy as np
import pytest

from pystarmax.exact_diffuse import (
    ExactDiffuseFilterResult,
    exact_diffuse_filter,
    exact_diffuse_loglikelihood,
)
from pystarmax.state_space import StateSpaceModel, kalman_filter


def local_level_model(variance: float = 0.25) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[variance]]),
        ar_order=1,
        ma_order=0,
    )


def local_linear_trend_model(variance: float = 0.1) -> StateSpaceModel:
    return StateSpaceModel(
        transition=np.array([[1.0, 1.0], [0.0, 1.0]]),
        design=np.array([[1.0, 0.0]]),
        selection=np.array([[1.0], [0.0]]),
        state_intercept=np.zeros(2),
        innovation_covariance=np.array([[variance]]),
        ar_order=2,
        ma_order=0,
    )


def test_local_level_exact_diffuse_likelihood_is_increment_likelihood() -> None:
    variance = 0.25
    data = np.array([[2.0], [2.5], [1.5]])
    result = exact_diffuse_filter(data, local_level_model(variance))

    expected = -0.5 * np.log(2.0 * np.pi)
    for increment in (0.5, -1.0):
        expected -= 0.5 * (
            np.log(2.0 * np.pi)
            + np.log(variance)
            + increment * increment / variance
        )

    assert result.log_likelihood == pytest.approx(expected)
    assert exact_diffuse_loglikelihood(data, local_level_model(variance)) == pytest.approx(
        expected
    )
    np.testing.assert_allclose(result.filtered_state[:, 0], data[:, 0])
    np.testing.assert_allclose(result.filtered_covariance[:, 0, 0], 0.0, atol=1e-14)
    np.testing.assert_array_equal(result.filtered_diffuse_rank, [0, 0, 0])
    assert result.initial_diffuse_rank == 1
    assert result.n_diffuse_observations == 1
    assert result.diffuse_end_time == 0
    assert result.final_diffuse_rank == 0
    assert result.diffuse_update_mask[0, 0]
    assert not result.diffuse_update_mask[1, 0]
    assert result.diffuse_innovation_variance[0, 0] == pytest.approx(1.0)
    assert result.finite_innovation_variance[0, 0] == pytest.approx(variance)


def test_exact_diffuse_matches_large_variance_limit_after_adjustment() -> None:
    model = local_linear_trend_model(variance=0.15)
    data = np.array([[1.0], [1.4], [2.1], [2.5], [3.2]])
    finite = np.array([[0.3, 0.05], [0.05, 0.2]])
    diffuse = np.eye(2)
    exact = exact_diffuse_filter(
        data,
        model,
        initial_covariance=finite,
        initial_diffuse_covariance=diffuse,
    )

    scale = 1e9
    approximate = kalman_filter(
        data,
        model,
        initialization="known",
        initial_state=np.zeros(2),
        initial_covariance=finite + scale * diffuse,
    )

    assert exact.n_diffuse_observations == 2
    np.testing.assert_array_equal(exact.filtered_diffuse_rank[:3], [1, 0, 0])
    np.testing.assert_allclose(
        exact.filtered_state[1:],
        approximate.filtered_state[1:],
        rtol=5e-6,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        exact.filtered_covariance[1:],
        approximate.filtered_covariance[1:],
        rtol=2e-5,
        atol=2e-5,
    )
    adjusted = approximate.log_likelihood + 0.5 * 2 * np.log(scale)
    assert adjusted == pytest.approx(exact.log_likelihood, rel=2e-6, abs=2e-6)


def test_missing_observation_delays_diffuse_rank_reduction() -> None:
    data = np.array([[np.nan], [3.0], [4.0]])
    result = exact_diffuse_filter(data, local_level_model())

    np.testing.assert_array_equal(result.predicted_diffuse_rank, [1, 1, 0])
    np.testing.assert_array_equal(result.filtered_diffuse_rank, [1, 0, 0])
    assert result.diffuse_end_time == 1
    assert result.n_observations == 2
    assert result.n_diffuse_observations == 1
    assert result.log_likelihood_contributions[0] == 0.0
    assert np.isnan(result.innovations[0, 0])


def test_partial_locations_are_updated_sequentially() -> None:
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
            [1.5, 2.0],
            [np.nan, 2.5],
        ]
    )

    result = exact_diffuse_filter(data, model)

    np.testing.assert_array_equal(result.filtered_diffuse_rank, [1, 0, 0])
    np.testing.assert_array_equal(
        result.diffuse_update_mask,
        [[True, False], [False, True], [False, False]],
    )
    assert result.n_diffuse_observations == 2
    assert result.n_observations == 4
    np.testing.assert_allclose(result.filtered_state[1], [1.5, 2.0])


def test_zero_variance_conflict_raises_but_matching_value_is_valid() -> None:
    deterministic = StateSpaceModel(
        transition=np.array([[0.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[0.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[0.0]]),
        ar_order=0,
        ma_order=0,
    )

    matching = exact_diffuse_filter(
        np.array([[0.0]]),
        deterministic,
        initial_diffuse_covariance=np.zeros((1, 1)),
    )
    assert matching.log_likelihood == 0.0
    assert matching.final_diffuse_rank == 0

    with pytest.raises(np.linalg.LinAlgError, match="deterministic"):
        exact_diffuse_filter(
            np.array([[1.0]]),
            deterministic,
            initial_diffuse_covariance=np.zeros((1, 1)),
        )


def test_custom_diffuse_covariance_and_result_immutability() -> None:
    model = local_linear_trend_model()
    diffuse = np.diag([1.0, 0.0])
    result = exact_diffuse_filter(
        np.array([[2.0], [2.2]]),
        model,
        initial_diffuse_covariance=diffuse,
    )

    assert isinstance(result, ExactDiffuseFilterResult)
    assert result.initial_diffuse_rank == 1
    assert result.n_diffuse_observations == 1
    assert not result.filtered_state.flags.writeable
    assert not result.filtered_diffuse_covariance.flags.writeable
    with pytest.raises(ValueError):
        result.filtered_state[0, 0] = 0.0


def test_validation_errors() -> None:
    model = local_level_model()
    with pytest.raises(TypeError, match="StateSpaceModel"):
        exact_diffuse_filter(np.ones((3, 1)), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two-dimensional"):
        exact_diffuse_filter(np.ones(3), model)
    with pytest.raises(ValueError, match="locations"):
        exact_diffuse_filter(np.ones((3, 2)), model)
    with pytest.raises(ValueError, match="infinite"):
        exact_diffuse_filter(np.array([[np.inf]]), model)
    with pytest.raises(ValueError, match="tolerance"):
        exact_diffuse_filter(np.ones((3, 1)), model, tolerance=0.0)
    with pytest.raises(ValueError, match="initial_state"):
        exact_diffuse_filter(np.ones((3, 1)), model, initial_state=np.zeros(2))
    with pytest.raises(ValueError, match="positive semidefinite"):
        exact_diffuse_filter(
            np.ones((3, 1)),
            model,
            initial_diffuse_covariance=np.array([[-1.0]]),
        )
