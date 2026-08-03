from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    KalmanSTARMA,
    SpatialWeights,
    build_starma_state_space,
    kalman_filter,
    kalman_smoother,
)


def identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_scalar_ar1_missing_bridge_matches_closed_form() -> None:
    phi = 0.6
    variance = 1.5
    first = 1.2
    last = -0.4
    model = build_starma_state_space(
        [[phi]],
        np.empty((0, 1)),
        identity_weights(1),
        [[variance]],
    )
    filtered = kalman_filter(
        np.array([[first], [np.nan], [last]], dtype=float),
        model,
    )

    smoothed = kalman_smoother(filtered)

    expected_mean = phi * first + phi / (1.0 + phi**2) * (last - phi**2 * first)
    expected_variance = variance / (1.0 + phi**2)
    assert smoothed.smoothed_observations[1, 0] == pytest.approx(expected_mean)
    assert smoothed.smoothed_observation_covariance[1, 0, 0] == pytest.approx(
        expected_variance
    )
    np.testing.assert_allclose(
        smoothed.smoothed_state[-1],
        filtered.filtered_state[-1],
    )
    np.testing.assert_allclose(
        smoothed.smoothed_covariance[-1],
        filtered.filtered_covariance[-1],
    )


def test_contiguous_missing_block_matches_joint_gaussian_conditioning() -> None:
    phi = 0.65
    variance = 1.4
    n_time = 5
    observed_values = np.array([0.7, -0.2], dtype=float)
    data = np.array(
        [[observed_values[0]], [np.nan], [np.nan], [np.nan], [observed_values[1]]]
    )
    model = build_starma_state_space(
        [[phi]],
        np.empty((0, 1)),
        identity_weights(1),
        [[variance]],
    )

    smoothed = kalman_smoother(kalman_filter(data, model))

    stationary_variance = variance / (1.0 - phi**2)
    covariance = stationary_variance * np.fromfunction(
        lambda row, column: phi ** np.abs(row - column),
        (n_time, n_time),
        dtype=float,
    )
    missing = np.array([1, 2, 3])
    observed = np.array([0, 4])
    covariance_missing_observed = covariance[np.ix_(missing, observed)]
    covariance_observed = covariance[np.ix_(observed, observed)]
    expected_mean = covariance_missing_observed @ np.linalg.solve(
        covariance_observed,
        observed_values,
    )
    expected_covariance = covariance[np.ix_(missing, missing)] - (
        covariance_missing_observed
        @ np.linalg.solve(covariance_observed, covariance_missing_observed.T)
    )

    np.testing.assert_allclose(
        smoothed.smoothed_observations[missing, 0],
        expected_mean,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        smoothed.smoothed_observation_covariance[missing, 0, 0],
        np.diag(expected_covariance),
        rtol=1e-10,
        atol=1e-10,
    )
    assert smoothed.lag_one_covariance[1, 0, 0] == pytest.approx(
        expected_covariance[0, 1]
    )
    assert smoothed.lag_one_covariance[2, 0, 0] == pytest.approx(
        expected_covariance[1, 2]
    )


def test_complete_scalar_ar1_recovers_state_disturbances() -> None:
    phi = 0.4
    data = np.array([[1.0], [0.3], [-0.2], [0.8]], dtype=float)
    model = build_starma_state_space(
        [[phi]],
        np.empty((0, 1)),
        identity_weights(1),
        [[0.8]],
    )

    smoothed = kalman_smoother(kalman_filter(data, model))

    expected = data[1:, 0] - phi * data[:-1, 0]
    np.testing.assert_allclose(smoothed.state_disturbance_mean[:, 0], expected)
    np.testing.assert_allclose(
        smoothed.state_disturbance_covariance,
        np.zeros((data.shape[0] - 1, 1, 1)),
        atol=1e-12,
    )


def test_rank_deficient_prediction_uses_explicit_pseudoinverse() -> None:
    model = build_starma_state_space(
        np.empty((0, 1)),
        [[0.0]],
        identity_weights(1),
        [[1.0]],
    )
    filtered = kalman_filter(np.array([[1.0], [0.5], [-0.2]]), model)

    smoothed = kalman_smoother(filtered)

    np.testing.assert_array_equal(smoothed.prediction_rank, [1, 1])
    np.testing.assert_array_equal(smoothed.used_pseudoinverse, [True, True])
    assert np.all(np.isfinite(smoothed.smoothing_gain))


def test_smoothing_covariance_does_not_exceed_filtered_covariance() -> None:
    model = build_starma_state_space(
        [[0.75]],
        np.empty((0, 1)),
        identity_weights(1),
        [[0.6]],
    )
    data = np.array([[0.3], [np.nan], [np.nan], [0.5], [np.nan], [-0.1]])
    filtered = kalman_filter(data, model)

    smoothed = kalman_smoother(filtered)

    for time_index in range(data.shape[0]):
        difference = (
            filtered.filtered_covariance[time_index]
            - smoothed.smoothed_covariance[time_index]
        )
        assert float(np.min(np.linalg.eigvalsh(difference))) >= -1e-10


def test_fitted_kalman_model_exposes_smoothing_for_new_missing_data() -> None:
    rng = np.random.default_rng(2026)
    training = rng.normal(scale=0.8, size=(180, 1))
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(training, identity_weights(1))
    incomplete = training[:20].copy()
    incomplete[5:9, 0] = np.nan

    smoothed = model.smooth(incomplete)

    assert smoothed.n_time == incomplete.shape[0]
    assert np.all(np.isfinite(smoothed.smoothed_observations))
    assert smoothed.filter_result.n_observations == incomplete.size - 4


def test_single_time_smoothing_has_empty_transition_outputs() -> None:
    model = build_starma_state_space(
        np.empty((0, 1)),
        np.empty((0, 1)),
        identity_weights(1),
        [[1.0]],
    )

    smoothed = kalman_smoother(kalman_filter(np.array([[0.5]]), model))

    assert smoothed.smoothing_gain.shape == (0, 1, 1)
    assert smoothed.lag_one_covariance.shape == (0, 1, 1)
    assert smoothed.state_disturbance_mean.shape == (0, 1)
    assert smoothed.prediction_rank.shape == (0,)


def test_smoothing_result_is_immutable_and_validates_rcond() -> None:
    model = build_starma_state_space(
        [[0.3]],
        np.empty((0, 1)),
        identity_weights(1),
        [[1.0]],
    )
    filtered = kalman_filter(np.array([[0.0], [np.nan], [1.0]]), model)
    smoothed = kalman_smoother(filtered)

    assert not smoothed.smoothed_state.flags.writeable
    assert not smoothed.smoothed_covariance.flags.writeable
    assert not smoothed.prediction_rank.flags.writeable
    assert not smoothed.used_pseudoinverse.flags.writeable
    with pytest.raises(ValueError, match="rcond"):
        kalman_smoother(filtered, rcond=0.0)
    with pytest.raises(TypeError, match="KalmanFilterResult"):
        kalman_smoother(object())  # type: ignore[arg-type]
