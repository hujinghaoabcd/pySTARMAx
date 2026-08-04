from __future__ import annotations

import numpy as np
import pytest

from pystarmax import ExactDiffuseKalmanSTARIMA, SpatialWeights
from pystarmax.exact_diffuse import exact_diffuse_filter
from pystarmax.exact_diffuse_simulation_smoothing import (
    ExactDiffuseSimulationSmootherResult,
    exact_diffuse_simulation_smoother,
)
from pystarmax.state_space import StateSpaceModel


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


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def test_random_walk_bridge_draws_closed_form_middle_state() -> None:
    variance = 0.5
    data = np.array([[1.0], [np.nan], [3.0]])
    result = exact_diffuse_simulation_smoother(
        exact_diffuse_filter(data, local_level_model(variance)),
        n_simulations=12000,
        random_state=2026,
    )

    middle = result.state_paths[:, 1, 0]
    np.testing.assert_allclose(middle.mean(), 2.0, atol=0.018)
    np.testing.assert_allclose(middle.var(), variance / 2.0, rtol=0.04)
    np.testing.assert_allclose(result.state_paths[:, 0, 0], 1.0, atol=2e-9)
    np.testing.assert_allclose(result.state_paths[:, 2, 0], 3.0, atol=2e-9)
    np.testing.assert_allclose(
        result.posterior_state_mean,
        result.smoother_result.smoothed_state,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.posterior_state_covariance,
        result.smoother_result.smoothed_covariance,
        atol=1e-9,
    )
    assert result.maximum_constraint_residual < 1e-8
    assert result.maximum_mean_discrepancy < 1e-9
    assert result.maximum_covariance_discrepancy < 1e-9


def test_fully_observed_random_walk_paths_are_deterministic() -> None:
    data = np.array([[1.0], [2.0], [4.0], [3.5]])
    result = exact_diffuse_simulation_smoother(
        exact_diffuse_filter(data, local_level_model()),
        n_simulations=20,
        random_state=7,
    )

    expected = np.broadcast_to(data, result.state_paths.shape)
    np.testing.assert_allclose(result.state_paths, expected, atol=2e-9)
    np.testing.assert_allclose(result.observation_paths, expected, atol=2e-9)
    np.testing.assert_allclose(result.posterior_state_covariance, 0.0, atol=1e-10)
    assert result.posterior_source_dimension == 0


def test_stationary_incomplete_case_matches_information_smoother_marginals() -> None:
    rng = np.random.default_rng(41)
    data = rng.normal(size=(14, 1))
    data[4:7, 0] = np.nan
    filtered = exact_diffuse_filter(
        data,
        stable_ar_model(),
        initial_state=np.array([0.1 / (1.0 - 0.45)]),
        initial_covariance=np.array([[0.3 / (1.0 - 0.45**2)]]),
        initial_diffuse_covariance=np.zeros((1, 1)),
    )
    result = exact_diffuse_simulation_smoother(
        filtered,
        n_simulations=10000,
        random_state=11,
    )

    np.testing.assert_allclose(
        result.posterior_state_mean,
        result.smoother_result.smoothed_state,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        result.posterior_state_covariance,
        result.smoother_result.smoothed_covariance,
        atol=2e-9,
    )
    sample_mean = result.state_paths.mean(axis=0)
    sample_variance = result.state_paths.var(axis=0)
    np.testing.assert_allclose(sample_mean[4:7], result.posterior_state_mean[4:7], atol=0.02)
    np.testing.assert_allclose(
        sample_variance[4:7, 0],
        result.posterior_state_covariance[4:7, 0, 0],
        rtol=0.06,
        atol=0.006,
    )
    observed = np.isfinite(data[:, 0])
    np.testing.assert_allclose(
        result.observation_paths[:, observed, 0],
        data[None, observed, 0],
        atol=2e-9,
    )
    assert result.diffuse_rank == 0


def test_partially_observed_locations_are_reproduced_exactly() -> None:
    model = StateSpaceModel(
        transition=np.eye(2, dtype=float),
        design=np.eye(2, dtype=float),
        selection=np.eye(2, dtype=float),
        state_intercept=np.zeros(2, dtype=float),
        innovation_covariance=np.diag([0.4, 0.7]),
        ar_order=1,
        ma_order=0,
    )
    data = np.array(
        [
            [1.0, np.nan],
            [np.nan, 2.0],
            [2.5, 3.0],
            [3.0, np.nan],
        ]
    )
    result = exact_diffuse_simulation_smoother(
        exact_diffuse_filter(data, model),
        n_simulations=200,
        random_state=19,
    )

    for time_index, location_index in np.argwhere(np.isfinite(data)):
        np.testing.assert_allclose(
            result.observation_paths[:, time_index, location_index],
            data[time_index, location_index],
            atol=3e-9,
        )
    assert np.all(np.isfinite(result.observation_paths))


def test_fitted_model_facade_reuses_training_and_handles_new_data() -> None:
    rng = np.random.default_rng(77)
    increments = 0.12 + rng.normal(scale=0.3, size=90)
    levels = np.cumsum(increments)[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    fitted = model.fit(levels, identity_weights())

    training = model.simulate_smoothing_paths(
        n_simulations=30,
        random_state=5,
    )
    assert isinstance(training, ExactDiffuseSimulationSmootherResult)
    assert training.filter_result is fitted.filter_result
    np.testing.assert_allclose(training.observation_paths, levels[None, :, :], atol=3e-8)

    new_levels = levels[-12:].copy()
    new_levels[5, 0] = np.nan
    first = model.simulate_smoothing_paths(
        new_levels,
        n_simulations=200,
        random_state=31,
    )
    second = model.simulate_smoothing_paths(
        new_levels,
        n_simulations=200,
        random_state=31,
    )
    assert first.filter_result is not fitted.filter_result
    np.testing.assert_array_equal(first.state_paths, second.state_paths)
    assert float(first.state_paths[:, 5, 0].var()) > 0.0


def test_unresolved_diffuse_direction_is_rejected() -> None:
    filtered = exact_diffuse_filter(
        np.full((4, 1), np.nan),
        local_level_model(),
    )
    assert filtered.final_diffuse_rank == 1
    with pytest.raises(RuntimeError, match="every diffuse direction"):
        exact_diffuse_simulation_smoother(filtered, n_simulations=10)


def test_result_arrays_are_immutable_and_arguments_are_validated() -> None:
    filtered = exact_diffuse_filter(
        np.array([[1.0], [np.nan], [2.0]]),
        local_level_model(),
    )
    result = exact_diffuse_simulation_smoother(
        filtered,
        n_simulations=10,
        random_state=3,
    )

    assert result.n_simulations == 10
    assert not result.state_paths.flags.writeable
    assert not result.observation_paths.flags.writeable
    assert not result.posterior_state_covariance.flags.writeable
    with pytest.raises(ValueError):
        result.state_paths[0, 0, 0] = 0.0
    with pytest.raises(TypeError, match="ExactDiffuseFilterResult"):
        exact_diffuse_simulation_smoother(  # type: ignore[arg-type]
            object(),
            n_simulations=10,
        )
    with pytest.raises(ValueError, match="at least two"):
        exact_diffuse_simulation_smoother(filtered, n_simulations=1)
    with pytest.raises(ValueError, match="rcond"):
        exact_diffuse_simulation_smoother(filtered, n_simulations=10, rcond=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        exact_diffuse_simulation_smoother(
            filtered,
            n_simulations=10,
            tolerance=0.0,
        )
