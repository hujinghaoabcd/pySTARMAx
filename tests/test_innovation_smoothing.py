from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    KalmanFilterResult,
    KalmanSmootherResult,
    KalmanSTARMA,
    SpatialWeights,
    StateSpaceModel,
    build_starma_state_space,
    innovation_disturbance_smoother,
    kalman_filter,
    kalman_smoother,
)


def identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def manual_smoother(
    selection: np.ndarray,
    innovation_covariance: np.ndarray,
    state_disturbance_mean: np.ndarray,
    state_disturbance_covariance: np.ndarray,
) -> KalmanSmootherResult:
    state_dim, n_locations = selection.shape
    n_transitions = state_disturbance_mean.shape[0]
    n_time = n_transitions + 1
    model = StateSpaceModel(
        transition=np.zeros((state_dim, state_dim), dtype=float),
        design=np.zeros((n_locations, state_dim), dtype=float),
        selection=selection,
        state_intercept=np.zeros(state_dim, dtype=float),
        innovation_covariance=innovation_covariance,
        ar_order=0,
        ma_order=0,
    )
    state = np.zeros((n_time, state_dim), dtype=float)
    state_covariance = np.zeros((n_time, state_dim, state_dim), dtype=float)
    filter_result = KalmanFilterResult(
        model=model,
        predicted_state=state,
        filtered_state=state,
        predicted_covariance=state_covariance,
        filtered_covariance=state_covariance,
        innovations=np.full((n_time, n_locations), np.nan, dtype=float),
        innovation_covariance=np.full(
            (n_time, n_locations, n_locations),
            np.nan,
            dtype=float,
        ),
        observed_mask=np.zeros((n_time, n_locations), dtype=bool),
        log_likelihood_contributions=np.zeros(n_time, dtype=float),
        jitter=np.zeros(n_time, dtype=float),
        log_likelihood=0.0,
        n_observations=0,
        initialization="known",
    )
    return KalmanSmootherResult(
        filter_result=filter_result,
        smoothed_state=state,
        smoothed_covariance=state_covariance,
        smoothing_gain=np.zeros(
            (n_transitions, state_dim, state_dim),
            dtype=float,
        ),
        lag_one_covariance=np.zeros(
            (n_transitions, state_dim, state_dim),
            dtype=float,
        ),
        state_disturbance_mean=state_disturbance_mean,
        state_disturbance_covariance=state_disturbance_covariance,
        prediction_rank=np.full(n_transitions, state_dim, dtype=np.int_),
        used_pseudoinverse=np.zeros(n_transitions, dtype=bool),
    )


def test_noninjective_selection_retains_unresolved_innovation_variance() -> None:
    selection = np.array([[1.0, 0.0]])
    innovation_covariance = np.array([[4.0, 2.0], [2.0, 9.0]])
    state_mean = np.array([[3.0]])
    state_covariance = np.array([[[1.5]]])
    smoothed = manual_smoother(
        selection,
        innovation_covariance,
        state_mean,
        state_covariance,
    )

    result = innovation_disturbance_smoother(smoothed)

    np.testing.assert_allclose(result.conditioning_map, [[1.0], [0.5]])
    np.testing.assert_allclose(
        result.unresolved_covariance,
        [[0.0, 0.0], [0.0, 8.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.innovation_mean, [[3.0, 1.5]])
    np.testing.assert_allclose(
        result.innovation_covariance[0],
        [[1.5, 0.75], [0.75, 8.375]],
    )
    np.testing.assert_allclose(
        selection @ result.innovation_mean[0],
        state_mean[0],
    )
    np.testing.assert_allclose(
        selection @ result.innovation_covariance[0] @ selection.T,
        state_covariance[0],
    )
    assert result.unresolved_variance == pytest.approx(8.0)
    assert result.process_rank == 1
    assert not result.used_pseudoinverse


def test_duplicated_selection_recovers_unique_innovation_with_pseudoinverse() -> None:
    selection = np.array([[1.0], [1.0]])
    innovation_covariance = np.array([[4.0]])
    state_mean = np.array([[2.0, 2.0], [-1.0, -1.0]])
    state_covariance = np.array(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.2, 0.2], [0.2, 0.2]],
        ]
    )
    smoothed = manual_smoother(
        selection,
        innovation_covariance,
        state_mean,
        state_covariance,
    )

    result = innovation_disturbance_smoother(smoothed)

    np.testing.assert_allclose(result.conditioning_map, [[0.5, 0.5]])
    np.testing.assert_allclose(result.unresolved_covariance, [[0.0]], atol=1e-12)
    np.testing.assert_allclose(result.innovation_mean[:, 0], [2.0, -1.0])
    np.testing.assert_allclose(result.innovation_covariance[:, 0, 0], [0.5, 0.2])
    np.testing.assert_allclose(result.mean_support_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.covariance_support_residual, 0.0, atol=1e-12)
    assert result.process_rank == 1
    assert result.used_pseudoinverse


def test_support_residuals_report_inconsistent_state_moments() -> None:
    selection = np.array([[1.0], [1.0]])
    smoothed = manual_smoother(
        selection,
        np.array([[1.0]]),
        np.array([[2.0, 3.0]]),
        np.array([np.eye(2, dtype=float)]),
    )

    result = innovation_disturbance_smoother(smoothed)

    assert result.innovation_mean[0, 0] == pytest.approx(2.5)
    assert result.innovation_covariance[0, 0, 0] == pytest.approx(0.5)
    assert result.mean_support_residual[0] == pytest.approx(np.sqrt(0.5))
    assert result.covariance_support_residual[0] == pytest.approx(1.0)


def test_scalar_ar1_innovation_moments_equal_state_disturbance_moments() -> None:
    phi = 0.4
    data = np.array([[1.0], [0.3], [-0.2], [0.8]], dtype=float)
    model = build_starma_state_space(
        [[phi]],
        np.empty((0, 1)),
        identity_weights(1),
        [[0.8]],
    )
    state_smoothed = kalman_smoother(kalman_filter(data, model))

    innovations = innovation_disturbance_smoother(state_smoothed)

    np.testing.assert_allclose(
        innovations.innovation_mean[:, 0],
        data[1:, 0] - phi * data[:-1, 0],
    )
    np.testing.assert_allclose(
        innovations.innovation_mean,
        state_smoothed.state_disturbance_mean,
    )
    np.testing.assert_allclose(
        innovations.innovation_covariance,
        state_smoothed.state_disturbance_covariance,
        atol=1e-12,
    )


def test_ma_state_duplication_reconstructs_state_disturbance() -> None:
    model = build_starma_state_space(
        np.empty((0, 1)),
        [[0.3]],
        identity_weights(1),
        [[1.2]],
    )
    data = np.array([[0.2], [-0.1], [0.7], [0.4]], dtype=float)
    state_smoothed = kalman_smoother(kalman_filter(data, model))

    innovations = innovation_disturbance_smoother(state_smoothed)

    reconstructed_mean = innovations.innovation_mean @ model.selection.T
    np.testing.assert_allclose(
        reconstructed_mean,
        state_smoothed.state_disturbance_mean,
        atol=1e-10,
    )
    for time_index in range(innovations.n_transitions):
        reconstructed_covariance = (
            model.selection
            @ innovations.innovation_covariance[time_index]
            @ model.selection.T
        )
        np.testing.assert_allclose(
            reconstructed_covariance,
            state_smoothed.state_disturbance_covariance[time_index],
            atol=1e-10,
        )


def test_fitted_model_exposes_innovation_disturbance_smoothing() -> None:
    rng = np.random.default_rng(2026)
    training = rng.normal(scale=0.7, size=(160, 1))
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(training, identity_weights(1))
    incomplete = training[:24].copy()
    incomplete[7:11, 0] = np.nan

    result = model.smooth_innovation_disturbances(incomplete)

    assert result.n_transitions == incomplete.shape[0] - 1
    assert result.innovation_mean.shape == (incomplete.shape[0] - 1, 1)
    assert np.all(np.isfinite(result.innovation_covariance))


def test_single_time_and_validation_behavior() -> None:
    model = build_starma_state_space(
        np.empty((0, 1)),
        np.empty((0, 1)),
        identity_weights(1),
        [[1.0]],
    )
    smoothed = kalman_smoother(kalman_filter(np.array([[0.5]]), model))

    result = innovation_disturbance_smoother(smoothed)

    assert result.innovation_mean.shape == (0, 1)
    assert result.innovation_covariance.shape == (0, 1, 1)
    assert result.mean_support_residual.shape == (0,)
    assert not result.innovation_mean.flags.writeable
    assert not result.innovation_covariance.flags.writeable
    assert not result.conditioning_map.flags.writeable
    assert not result.unresolved_covariance.flags.writeable
    with pytest.raises(ValueError, match="rcond"):
        innovation_disturbance_smoother(smoothed, rcond=0.0)
    with pytest.raises(TypeError, match="KalmanSmootherResult"):
        innovation_disturbance_smoother(object())  # type: ignore[arg-type]
