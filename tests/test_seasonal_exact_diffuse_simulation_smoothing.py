from __future__ import annotations

import numpy as np
import pytest

from pystarmax import SeasonalExactDiffuseKalmanSTARIMA, SpatialWeights
from pystarmax.exact_diffuse_simulation_smoothing import (
    exact_diffuse_simulation_smoother,
)
from pystarmax.exact_seasonal_integrated import (
    build_exact_seasonal_integrated_state_space,
)
from pystarmax.seasonal_exact_diffuse_simulation_smoothing import (
    SeasonalExactDiffuseSimulationSmootherResult,
    seasonal_exact_diffuse_simulation_smoother,
)
from pystarmax.state_space import StateSpaceModel


def white_noise_model(
    covariance: np.ndarray | float = 0.5,
) -> StateSpaceModel:
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.ndim == 0:
        covariance_array = np.array([[float(covariance_array)]])
    n_locations = covariance_array.shape[0]
    return StateSpaceModel(
        transition=np.zeros((n_locations, n_locations), dtype=float),
        design=np.eye(n_locations, dtype=float),
        selection=np.eye(n_locations, dtype=float),
        state_intercept=np.zeros(n_locations, dtype=float),
        innovation_covariance=covariance_array,
        ar_order=0,
        ma_order=0,
    )


def period_two_specification(
    variance: float = 0.5,
):
    return build_exact_seasonal_integrated_state_space(
        white_noise_model(variance),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def test_period_two_bridge_paths_match_closed_form_and_difference_identity() -> None:
    variance = 0.5
    data = np.array([[0.0], [10.0], [np.nan], [np.nan], [4.0], [14.0]])
    specification = period_two_specification(variance)
    result = seasonal_exact_diffuse_simulation_smoother(
        specification.filter(data),
        specification,
        n_simulations=12000,
        random_state=2026,
    )

    np.testing.assert_allclose(
        result.observation_paths[:, 0, 0],
        0.0,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        result.observation_paths[:, 1, 0],
        10.0,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        result.observation_paths[:, 4, 0],
        4.0,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        result.observation_paths[:, 5, 0],
        14.0,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        result.observation_paths[:, 2, 0].mean(),
        2.0,
        atol=0.018,
    )
    np.testing.assert_allclose(
        result.observation_paths[:, 3, 0].mean(),
        12.0,
        atol=0.018,
    )
    np.testing.assert_allclose(
        result.observation_paths[:, 2, 0].var(),
        variance / 2.0,
        rtol=0.04,
    )
    np.testing.assert_allclose(
        result.observation_paths[:, 3, 0].var(),
        variance / 2.0,
        rtol=0.04,
    )
    np.testing.assert_allclose(
        result.transformed_observation_paths[:, 2:, 0],
        result.observation_paths[:, 2:, 0]
        - result.observation_paths[:, :-2, 0],
        atol=2e-12,
    )
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


def test_transformed_outputs_are_exact_projections_of_complete_paths() -> None:
    data = np.array([[1.0], [8.0], [np.nan], [11.0], [4.0], [np.nan]])
    specification = period_two_specification(0.4)
    result = seasonal_exact_diffuse_simulation_smoother(
        specification.filter(data),
        specification,
        n_simulations=300,
        random_state=12,
    )
    offset = specification.integration_degree * specification.model.n_locations

    np.testing.assert_array_equal(
        result.transformed_state_paths,
        result.state_paths[..., offset:],
    )
    np.testing.assert_allclose(
        result.transformed_observation_paths,
        result.transformed_state_paths @ specification.transformed_model.design.T,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        result.posterior_transformed_state_mean,
        result.posterior_state_mean[:, offset:],
    )
    np.testing.assert_array_equal(
        result.posterior_transformed_state_covariance,
        result.posterior_state_covariance[:, offset:, offset:],
    )


def test_partially_observed_locations_are_conditioned_exactly() -> None:
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_model(np.diag([0.4, 0.7])),
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    data = np.array(
        [
            [1.0, np.nan],
            [np.nan, 10.0],
            [np.nan, 12.0],
            [3.0, np.nan],
            [5.0, 14.0],
            [7.0, 16.0],
        ]
    )
    result = seasonal_exact_diffuse_simulation_smoother(
        specification.filter(data),
        specification,
        n_simulations=200,
        random_state=19,
    )

    for time_index, location_index in np.argwhere(np.isfinite(data)):
        np.testing.assert_allclose(
            result.observation_paths[:, time_index, location_index],
            data[time_index, location_index],
            atol=4e-9,
        )
    assert np.all(np.isfinite(result.observation_paths))
    assert np.all(np.isfinite(result.transformed_observation_paths))


def test_zero_seasonal_order_reduces_exactly_to_ordinary_simulation_smoother() -> None:
    specification = build_exact_seasonal_integrated_state_space(
        white_noise_model(0.3),
        ordinary_integration_order=1,
        seasonal_integration_order=0,
        seasonal_period=4,
    )
    data = np.array([[0.0], [np.nan], [2.0], [3.0], [np.nan], [5.0]])
    filtered = specification.filter(data)
    ordinary = exact_diffuse_simulation_smoother(
        filtered,
        n_simulations=400,
        random_state=73,
    )
    seasonal = seasonal_exact_diffuse_simulation_smoother(
        filtered,
        specification,
        n_simulations=400,
        random_state=73,
    )

    np.testing.assert_array_equal(
        seasonal.simulation_result.state_paths,
        ordinary.state_paths,
    )
    np.testing.assert_array_equal(
        seasonal.simulation_result.observation_paths,
        ordinary.observation_paths,
    )
    np.testing.assert_array_equal(
        seasonal.posterior_state_mean,
        ordinary.posterior_state_mean,
    )
    np.testing.assert_array_equal(
        seasonal.posterior_state_covariance,
        ordinary.posterior_state_covariance,
    )
    assert seasonal.diffuse_rank == ordinary.diffuse_rank
    assert seasonal.conditioning_rank == ordinary.conditioning_rank
    assert seasonal.posterior_source_dimension == ordinary.posterior_source_dimension


def test_fitted_facade_reuses_training_filter_and_refilters_new_data() -> None:
    rng = np.random.default_rng(77)
    innovations = rng.normal(scale=0.3, size=100)
    levels = np.empty(100, dtype=float)
    levels[:2] = (1.0, 10.0)
    for time_index in range(2, levels.size):
        levels[time_index] = levels[time_index - 2] + innovations[time_index]
    data = levels[:, None]
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
        max_iter=300,
    )
    fitted = model.fit(data, identity_weights())

    training = model.simulate_smoothing_paths(
        n_simulations=20,
        random_state=5,
    )
    assert isinstance(training, SeasonalExactDiffuseSimulationSmootherResult)
    assert training.filter_result is fitted.filter_result
    np.testing.assert_allclose(
        training.observation_paths,
        np.broadcast_to(data, training.observation_paths.shape),
        atol=4e-8,
    )

    new_data = data[-14:].copy()
    new_data[4, 0] = np.nan
    new_data[9, 0] = np.nan
    first = model.simulate_smoothing_paths(
        new_data,
        n_simulations=200,
        random_state=31,
    )
    second = model.simulate_smoothing_paths(
        new_data,
        n_simulations=200,
        random_state=31,
    )
    assert first.filter_result is not fitted.filter_result
    np.testing.assert_array_equal(first.state_paths, second.state_paths)
    np.testing.assert_array_equal(
        first.transformed_state_paths,
        second.transformed_state_paths,
    )
    assert float(first.observation_paths[:, 4, 0].var()) > 0.0
    assert float(first.observation_paths[:, 9, 0].var()) > 0.0


def test_unresolved_diffuse_rank_and_mismatched_specification_are_rejected() -> None:
    specification = period_two_specification()
    unresolved = specification.filter(np.full((4, 1), np.nan))
    assert unresolved.final_diffuse_rank != 0
    with pytest.raises(RuntimeError, match="every diffuse direction"):
        seasonal_exact_diffuse_simulation_smoother(
            unresolved,
            specification,
            n_simulations=10,
        )

    data = np.array([[0.0], [10.0], [2.0], [12.0]])
    filtered = specification.filter(data)
    other = period_two_specification()
    with pytest.raises(ValueError, match="state space used"):
        seasonal_exact_diffuse_simulation_smoother(
            filtered,
            other,
            n_simulations=10,
        )


def test_result_arrays_are_immutable_and_arguments_are_validated() -> None:
    specification = period_two_specification()
    filtered = specification.filter(
        np.array([[0.0], [10.0], [np.nan], [12.0], [4.0], [14.0]])
    )
    result = seasonal_exact_diffuse_simulation_smoother(
        filtered,
        specification,
        n_simulations=10,
        random_state=3,
    )

    assert result.n_simulations == 10
    assert not result.state_paths.flags.writeable
    assert not result.observation_paths.flags.writeable
    assert not result.transformed_state_paths.flags.writeable
    assert not result.transformed_observation_paths.flags.writeable
    assert not result.posterior_transformed_state_covariance.flags.writeable
    with pytest.raises(ValueError):
        result.transformed_state_paths[0, 0, 0] = 0.0
    with pytest.raises(TypeError, match="ExactDiffuseFilterResult"):
        seasonal_exact_diffuse_simulation_smoother(  # type: ignore[arg-type]
            object(),
            specification,
            n_simulations=10,
        )
    with pytest.raises(TypeError, match="ExactSeasonalIntegratedStateSpace"):
        seasonal_exact_diffuse_simulation_smoother(  # type: ignore[arg-type]
            filtered,
            object(),
            n_simulations=10,
        )
    with pytest.raises(ValueError, match="at least two"):
        seasonal_exact_diffuse_simulation_smoother(
            filtered,
            specification,
            n_simulations=1,
        )
    with pytest.raises(ValueError, match="rcond"):
        seasonal_exact_diffuse_simulation_smoother(
            filtered,
            specification,
            n_simulations=10,
            rcond=0.0,
        )
    with pytest.raises(ValueError, match="tolerance"):
        seasonal_exact_diffuse_simulation_smoother(
            filtered,
            specification,
            n_simulations=10,
            tolerance=0.0,
        )
