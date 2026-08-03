from __future__ import annotations

import numpy as np
import pytest

import pystarmax.seasonal_likelihood_inference as seasonal_inference
from pystarmax import (
    FiniteDifferenceCurvature,
    KalmanSTARMA,
    SeasonalKalmanSTARIMA,
    SpatialWeights,
    infer_seasonal_kalman_starima,
)


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_zero_seasonal_orders_match_stationary_likelihood_inference() -> None:
    rng = np.random.default_rng(2026)
    data = rng.normal(loc=0.25, scale=0.8, size=(120, 1))
    weights = identity_weights()
    start_params = np.array([0.1])
    start_covariance = np.array([[0.7]])

    stationary = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    seasonal = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    stationary.fit(
        data,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )
    seasonal.fit(
        data,
        weights,
        start_params=start_params,
        start_covariance=start_covariance,
    )

    stationary_result = stationary.infer(relative_step=2e-4, absolute_step=1e-6)
    seasonal_result = seasonal.infer(relative_step=2e-4, absolute_step=1e-6)

    assert seasonal_result.parameter_names == stationary_result.parameter_names
    np.testing.assert_allclose(seasonal_result.estimates, stationary_result.estimates)
    np.testing.assert_allclose(
        seasonal_result.hessian,
        stationary_result.hessian,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        seasonal_result.covariance,
        stationary_result.covariance,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        seasonal_result.standard_errors,
        stationary_result.standard_errors,
        rtol=2e-5,
        atol=2e-5,
    )
    assert seasonal_result.objective_value == pytest.approx(
        stationary_result.objective_value
    )
    assert seasonal_result.n_dynamic_params == stationary_result.n_dynamic_params


def test_pure_seasonal_ar_inference_and_natural_covariance() -> None:
    rng = np.random.default_rng(17)
    period = 4
    coefficient = 0.35
    data = np.zeros(220, dtype=float)
    innovations = rng.normal(scale=0.55, size=data.size)
    for time_index in range(period, data.size):
        data[time_index] = (
            coefficient * data[time_index - period] + innovations[time_index]
        )
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=period,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=400,
    )
    model.fit(
        data[:, None],
        identity_weights(),
        start_params=np.array([0.2]),
        start_covariance=np.array([[0.4]]),
    )

    result = model.infer(relative_step=3e-4, absolute_step=1e-6)
    direct = infer_seasonal_kalman_starima(
        model,
        relative_step=3e-4,
        absolute_step=1e-6,
    )
    natural = result.innovation_covariance_inference()

    assert result.parameter_names == ("sar.t4.W0", "cov.log_sd")
    assert result.n_dynamic_params == 1
    assert result.rank == 2
    assert result.positive_definite
    assert not result.used_pseudoinverse
    assert np.all(result.standard_errors > 0.0)
    assert result.stability_boundary_distance > 0.0
    assert result.invertibility_boundary_distance > 0.0
    np.testing.assert_allclose(result.hessian, direct.hessian)
    np.testing.assert_allclose(result.covariance, direct.covariance)
    assert natural.element_names == ("variance",)
    assert natural.estimates[0] == pytest.approx(
        model.result_.innovation_covariance[0, 0]  # type: ignore[union-attr]
    )
    assert natural.standard_errors[0] > 0.0
    assert natural.dynamic_cross_covariance.shape == (1, 1)


def test_expanded_admissibility_penalty_is_not_likelihood_curvature() -> None:
    rng = np.random.default_rng(27)
    data = rng.normal(size=(100, 1))
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=300,
    )
    result = model.fit(
        data,
        identity_weights(),
        start_params=np.array([0.1]),
        start_covariance=np.array([[0.8]]),
    )
    invalid = result.raw_optimizer_params.copy()
    invalid[0] = 1.2

    objective = seasonal_inference._negative_log_likelihood(model, invalid)

    assert objective >= 1e12


def test_finite_difference_rejects_stencil_crossing_expanded_boundary() -> None:
    rng = np.random.default_rng(37)
    period = 4
    data = np.zeros(180, dtype=float)
    innovations = rng.normal(scale=0.35, size=data.size)
    for time_index in range(period, data.size):
        data[time_index] = 0.92 * data[time_index - period] + innovations[time_index]
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=1,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=period,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=400,
    )
    model.fit(
        data[:, None],
        identity_weights(),
        start_params=np.array([0.85]),
        start_covariance=np.array([[0.2]]),
    )

    with pytest.raises(ValueError, match="invalid parameter region"):
        model.infer(relative_step=0.2, absolute_step=0.2)


def test_singular_hessian_requires_explicit_pseudoinverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(47)
    model = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=300,
    )
    model.fit(rng.normal(size=(80, 1)), identity_weights())

    def fake_curvature(*args: object, **kwargs: object) -> FiniteDifferenceCurvature:
        del args, kwargs
        point = model.result_.raw_optimizer_params  # type: ignore[union-attr]
        return FiniteDifferenceCurvature(
            point=point,
            steps=np.full(point.size, 1e-4),
            function_value=10.0,
            gradient=np.zeros(point.size),
            hessian=np.diag([2.0, 0.0]),
            n_function_evaluations=9,
        )

    monkeypatch.setattr(seasonal_inference, "finite_difference_curvature", fake_curvature)

    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        model.infer()

    result = model.infer(allow_singular=True)

    assert result.used_pseudoinverse
    assert not result.positive_definite
    assert result.rank == 1
    assert np.all(np.isfinite(result.covariance))
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(np.isfinite(result.p_values))


def test_validation_and_not_fitted_errors() -> None:
    model = SeasonalKalmanSTARIMA(
        seasonal_integration_order=0,
        seasonal_period=4,
    )
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.infer()
    with pytest.raises(RuntimeError, match="fit must be called"):
        infer_seasonal_kalman_starima(model)

    rng = np.random.default_rng(57)
    fitted = SeasonalKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=4,
        include_intercept=True,
        max_iter=300,
    )
    fitted.fit(rng.normal(size=(80, 1)), identity_weights())
    with pytest.raises(ValueError, match="rcond"):
        fitted.infer(rcond=0.0)
    with pytest.raises(ValueError, match="relative_step"):
        fitted.infer(relative_step=0.0)
