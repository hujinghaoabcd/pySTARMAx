from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    STAR,
    SpatialWeights,
    build_starma_state_space,
    kalman_filter,
    kalman_loglikelihood,
    simulate_starma,
)


def identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_build_state_space_preserves_spatial_operator_order() -> None:
    identity = np.eye(2, dtype=float)
    asymmetric = np.array([[0.0, 1.0], [0.25, 0.0]], dtype=float)
    weights = SpatialWeights(
        matrices=(identity, asymmetric),
        names=("W0", "W1"),
    )
    ar_parameters = np.array([[0.4, 0.2], [0.1, -0.05]], dtype=float)
    ma_parameters = np.array([[0.3, 0.1]], dtype=float)
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=float)

    model = build_starma_state_space(
        ar_parameters,
        ma_parameters,
        weights,
        covariance,
        intercept=np.array([1.0, -1.0]),
    )

    expected_ar_1 = 0.4 * identity + 0.2 * asymmetric
    expected_ar_2 = 0.1 * identity - 0.05 * asymmetric
    expected_ma_1 = 0.3 * identity + 0.1 * asymmetric
    assert model.state_dim == 6
    np.testing.assert_allclose(model.transition[:2, :2], expected_ar_1)
    np.testing.assert_allclose(model.transition[:2, 2:4], expected_ar_2)
    np.testing.assert_allclose(model.transition[:2, 4:6], expected_ma_1)
    np.testing.assert_allclose(model.transition[2:4, :2], identity)
    np.testing.assert_allclose(model.selection[:2], identity)
    np.testing.assert_allclose(model.selection[4:6], identity)
    np.testing.assert_allclose(model.state_intercept[:2], [1.0, -1.0])
    assert not model.transition.flags.writeable
    assert not model.innovation_covariance.flags.writeable


def test_scalar_ar1_stationary_likelihood_matches_closed_form() -> None:
    phi = 0.5
    variance = 2.0
    data = np.array([[1.0], [0.2], [-0.1], [0.5]], dtype=float)
    model = build_starma_state_space(
        [[phi]],
        np.empty((0, 1)),
        identity_weights(1),
        [[variance]],
    )

    result = kalman_filter(data, model)
    expected = -0.5 * (
        np.log(2.0 * np.pi * variance / (1.0 - phi**2))
        + data[0, 0] ** 2 / (variance / (1.0 - phi**2))
    )
    for time_index in range(1, data.shape[0]):
        innovation = data[time_index, 0] - phi * data[time_index - 1, 0]
        expected += -0.5 * (np.log(2.0 * np.pi * variance) + innovation**2 / variance)

    assert result.log_likelihood == pytest.approx(expected)
    assert kalman_loglikelihood(data, model) == pytest.approx(expected)
    np.testing.assert_allclose(
        result.innovations[:, 0],
        [1.0, -0.3, -0.2, 0.55],
    )
    np.testing.assert_allclose(
        result.predicted_observations[:, 0],
        [0.0, 0.5, 0.1, -0.05],
    )


def test_partial_and_fully_missing_observations_are_filtered() -> None:
    weights = identity_weights(2)
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=float)
    model = build_starma_state_space(
        [[0.3]],
        np.empty((0, 1)),
        weights,
        covariance,
    )
    data = np.array(
        [
            [1.0, 2.0],
            [np.nan, 1.0],
            [0.5, np.nan],
            [np.nan, np.nan],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    result = kalman_filter(data, model)

    assert result.n_observations == 6
    assert np.isnan(result.innovations[1, 0])
    assert np.isnan(result.innovations[2, 1])
    assert np.isnan(result.innovation_covariance[1, 0, 0])
    assert result.log_likelihood_contributions[3] == 0.0
    np.testing.assert_allclose(
        result.predicted_state[3],
        result.filtered_state[3],
    )
    np.testing.assert_allclose(
        result.predicted_covariance[3],
        result.filtered_covariance[3],
    )
    assert np.all(np.isfinite(result.filtered_observations))
    assert not result.filtered_state.flags.writeable
    assert not result.observed_mask.flags.writeable


def test_nonstationary_model_requires_diffuse_or_known_initialization() -> None:
    model = build_starma_state_space(
        [[1.0]],
        np.empty((0, 1)),
        identity_weights(1),
        [[1.0]],
    )
    data = np.array([[0.0], [1.0], [0.5]], dtype=float)

    with pytest.raises(ValueError, match="spectral radius"):
        kalman_filter(data, model)

    result = kalman_filter(
        data,
        model,
        initialization="diffuse",
        diffuse_scale=1e4,
    )
    assert result.initialization == "diffuse"
    assert np.isfinite(result.log_likelihood)


def test_known_initialization_and_white_noise_state() -> None:
    model = build_starma_state_space(
        np.empty((0, 1)),
        np.empty((0, 1)),
        identity_weights(1),
        [[0.5]],
        intercept=2.0,
    )
    data = np.array([[2.5], [1.5]], dtype=float)

    result = kalman_filter(
        data,
        model,
        initialization="known",
        initial_state=np.zeros(1),
        initial_covariance=np.zeros((1, 1)),
    )

    np.testing.assert_allclose(result.predicted_observations[:, 0], [2.0, 2.0])
    np.testing.assert_allclose(result.innovations[:, 0], [0.5, -0.5])
    assert result.n_observations == 2


def test_fitted_star_exposes_state_space_missing_data_path() -> None:
    weights = identity_weights(1)
    series = simulate_starma(
        phi=np.array([[0.35]], dtype=float),
        theta=np.empty((0, 1), dtype=float),
        weights=weights,
        n_steps=160,
        burnin=50,
        innovation_covariance=0.5,
        random_state=42,
    )
    model = STAR(ar_order=1, include_intercept=False)
    model.fit(series, weights)

    state_space = model.to_state_space()
    complete = model.filter_state_space()
    missing = series.copy()
    missing[::11, 0] = np.nan
    incomplete = model.filter_state_space(missing)

    assert state_space.ar_order == 1
    assert state_space.ma_order == 0
    assert complete.n_observations == series.size
    assert incomplete.n_observations == series.size - missing[::11].size
    assert np.isfinite(complete.log_likelihood)
    assert np.isfinite(incomplete.log_likelihood)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.eye(3), "spatial weights contain"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    ],
)
def test_state_space_validation(covariance: np.ndarray, message: str) -> None:
    weights = identity_weights(2)
    with pytest.raises(ValueError, match=message):
        build_starma_state_space(
            [[0.2]],
            np.empty((0, 1)),
            weights,
            covariance,
        )
