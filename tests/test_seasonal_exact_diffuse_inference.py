from __future__ import annotations

import numpy as np
import pytest

import pystarmax.seasonal_exact_diffuse_inference as inference_module
from pystarmax import (
    ExactDiffuseKalmanSTARIMA,
    FiniteDifferenceCurvature,
    SeasonalExactDiffuseKalmanSTARIMA,
    SpatialWeights,
)
from pystarmax.likelihood_inference import LikelihoodInferenceResult
from pystarmax.seasonal_exact_diffuse_inference import (
    infer_seasonal_exact_diffuse_kalman_starima,
)


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def seasonal_random_walk(
    increments: np.ndarray,
    *,
    initial: tuple[float, float] = (0.0, 10.0),
) -> np.ndarray:
    values = np.empty(increments.size + 2, dtype=float)
    values[:2] = initial
    for index, increment in enumerate(increments, start=2):
        values[index] = values[index - 2] + increment
    return values[:, None]


def test_seasonal_random_walk_information_matches_closed_form() -> None:
    rng = np.random.default_rng(2901)
    increments = 0.3 + rng.normal(scale=0.65, size=80)
    levels = seasonal_random_walk(increments)
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
        tol=1e-11,
    )
    fitted = model.fit(
        levels,
        identity_weights(),
        start_params=np.array([np.mean(increments)]),
        start_covariance=float(np.var(increments)),
    )

    inference = model.likelihood_inference(
        relative_step=2e-4,
        absolute_step=1e-6,
    )

    n_transitions = increments.size
    variance = float(fitted.innovation_covariance[0, 0])
    expected_hessian = np.diag([n_transitions / variance, 2.0 * n_transitions])
    expected_covariance = np.diag(
        [variance / n_transitions, 1.0 / (2.0 * n_transitions)]
    )

    assert isinstance(inference, LikelihoodInferenceResult)
    assert inference.parameter_names == fitted.optimizer_parameter_names
    assert inference.n_dynamic_params == 1
    assert inference.n_function_evaluations == 9
    assert inference.positive_definite
    assert not inference.used_pseudoinverse
    assert inference.rank == 2
    assert inference.objective_value == pytest.approx(-fitted.log_likelihood, abs=1e-8)
    np.testing.assert_allclose(
        inference.hessian,
        expected_hessian,
        rtol=3e-3,
        atol=3e-3,
    )
    np.testing.assert_allclose(
        inference.covariance,
        expected_covariance,
        rtol=3e-3,
        atol=3e-5,
    )

    natural = inference.innovation_covariance_inference()
    expected_variance_se = variance * np.sqrt(2.0 / n_transitions)
    np.testing.assert_allclose(natural.estimates, [variance])
    np.testing.assert_allclose(
        natural.standard_errors,
        [expected_variance_se],
        rtol=3e-3,
        atol=3e-5,
    )
    assert natural.dynamic_parameter_names == fitted.parameter_names


def test_zero_seasonal_orders_reduce_to_ordinary_exact_inference() -> None:
    rng = np.random.default_rng(2902)
    increments = 0.15 + rng.normal(scale=0.5, size=64)
    levels = np.concatenate([[2.0], 2.0 + np.cumsum(increments)])[:, None]
    weights = identity_weights()
    start = np.array([np.mean(increments)])
    variance = float(np.var(increments))

    ordinary = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
        tol=1e-11,
    )
    seasonal = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
        tol=1e-11,
    )
    ordinary.fit(
        levels,
        weights,
        start_params=start,
        start_covariance=variance,
    )
    seasonal.fit(
        levels,
        weights,
        start_params=start,
        start_covariance=variance,
    )

    ordinary_inference = ordinary.likelihood_inference(relative_step=2e-4)
    seasonal_inference = seasonal.likelihood_inference(relative_step=2e-4)

    assert seasonal_inference.parameter_names == ordinary_inference.parameter_names
    np.testing.assert_allclose(
        seasonal_inference.estimates,
        ordinary_inference.estimates,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        seasonal_inference.hessian,
        ordinary_inference.hessian,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        seasonal_inference.covariance,
        ordinary_inference.covariance,
        rtol=2e-5,
        atol=2e-7,
    )


def test_missing_data_inference_is_finite_and_immutable() -> None:
    rng = np.random.default_rng(2903)
    increments = 0.1 + rng.normal(scale=0.45, size=76)
    levels = seasonal_random_walk(increments, initial=(1.0, -3.0))
    levels[16:20, 0] = np.nan
    levels[49, 0] = np.nan
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
    )
    model.fit(levels, identity_weights())

    inference = infer_seasonal_exact_diffuse_kalman_starima(
        model,
        allow_singular=True,
    )

    assert np.all(np.isfinite(inference.hessian))
    assert np.all(np.isfinite(inference.covariance))
    assert np.all(np.isfinite(inference.standard_errors))
    assert inference.stability_boundary_distance > 0.0
    assert inference.invertibility_boundary_distance > 0.0
    assert not inference.estimates.flags.writeable
    assert not inference.covariance.flags.writeable
    with pytest.raises(ValueError):
        inference.estimates[0] = 0.0


def test_full_covariance_delta_method_uses_factor_parameter_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2904)
    covariance = np.array([[0.5, 0.18], [0.18, 0.8]])
    increments = rng.multivariate_normal(np.zeros(2), covariance, size=54)
    levels = np.empty((56, 2), dtype=float)
    levels[:2] = np.array([[0.0, 1.0], [2.0, -1.0]])
    for index in range(2, levels.shape[0]):
        levels[index] = levels[index - 2] + increments[index - 2]
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="full",
        include_intercept=False,
        max_iter=500,
    )
    fitted = model.fit(
        levels,
        identity_weights(2),
        start_covariance=covariance,
    )
    point = fitted.raw_optimizer_params
    curvature = FiniteDifferenceCurvature(
        point=point,
        steps=np.full(point.size, 1e-4, dtype=float),
        function_value=-fitted.log_likelihood,
        gradient=np.zeros(point.size, dtype=float),
        hessian=2.0 * np.eye(point.size),
        n_function_evaluations=(
            1
            + 2 * point.size
            + 4 * (point.size * (point.size - 1) // 2)
        ),
    )
    monkeypatch.setattr(
        inference_module,
        "finite_difference_curvature",
        lambda *args, **kwargs: curvature,
    )

    inference = model.likelihood_inference()
    natural = inference.innovation_covariance_inference()

    assert fitted.params.size == 0
    assert inference.n_dynamic_params == 0
    assert point.size == 3
    assert natural.parameter_names == (
        "variance.location0",
        "covariance.location1.location0",
        "variance.location1",
    )
    np.testing.assert_allclose(natural.covariance_matrix, fitted.innovation_covariance)
    assert natural.dynamic_cross_covariance.shape == (0, 3)
    assert np.all(np.isfinite(natural.standard_errors))
    assert not natural.covariance.flags.writeable


def test_singular_hessian_requires_explicit_generalized_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2905)
    increments = rng.normal(scale=0.5, size=48)
    levels = seasonal_random_walk(increments)
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=True,
    )
    fitted = model.fit(levels, identity_weights())
    point = fitted.raw_optimizer_params
    curvature = FiniteDifferenceCurvature(
        point=point,
        steps=np.full(point.size, 1e-4, dtype=float),
        function_value=-fitted.log_likelihood,
        gradient=np.zeros(point.size, dtype=float),
        hessian=np.diag([1.0, 0.0]),
        n_function_evaluations=9,
    )
    monkeypatch.setattr(
        inference_module,
        "finite_difference_curvature",
        lambda *args, **kwargs: curvature,
    )

    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        model.likelihood_inference()

    inference = model.likelihood_inference(allow_singular=True)

    assert inference.used_pseudoinverse
    assert not inference.positive_definite
    assert inference.rank == 1
    assert np.all(np.isfinite(inference.standard_errors))
    assert np.all(np.isfinite(inference.z_values))
    assert np.all(np.isfinite(inference.p_values))
    assert np.all(np.isfinite(inference.correlation))


def test_inference_validates_fit_steps_and_rcond() -> None:
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
    )
    with pytest.raises(RuntimeError, match="fit must be called"):
        infer_seasonal_exact_diffuse_kalman_starima(model)

    increments = np.linspace(-0.2, 0.4, 30)
    model.fit(seasonal_random_walk(increments), identity_weights())
    with pytest.raises(ValueError, match="relative_step"):
        infer_seasonal_exact_diffuse_kalman_starima(model, relative_step=0.0)
    with pytest.raises(ValueError, match="absolute_step"):
        infer_seasonal_exact_diffuse_kalman_starima(model, absolute_step=np.inf)
    with pytest.raises(ValueError, match="rcond"):
        infer_seasonal_exact_diffuse_kalman_starima(model, rcond=0.0)
