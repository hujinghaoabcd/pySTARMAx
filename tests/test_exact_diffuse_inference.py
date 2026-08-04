from __future__ import annotations

import numpy as np
import pytest

import pystarmax.exact_diffuse_inference as inference_module
from pystarmax import (
    ExactDiffuseKalmanSTARIMA,
    FiniteDifferenceCurvature,
    SpatialWeights,
)
from pystarmax.exact_diffuse_inference import infer_exact_diffuse_kalman_starima
from pystarmax.likelihood_inference import LikelihoodInferenceResult


def identity_weights(n_locations: int = 1) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_random_walk_observed_information_matches_closed_form() -> None:
    rng = np.random.default_rng(2301)
    increments = 0.35 + rng.normal(scale=0.7, size=80)
    levels = np.concatenate([[0.0], np.cumsum(increments)])[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
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

    n_increments = increments.size
    variance = float(fitted.innovation_covariance[0, 0])
    expected_hessian = np.diag([n_increments / variance, 2.0 * n_increments])
    expected_covariance = np.diag([variance / n_increments, 1.0 / (2.0 * n_increments)])

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
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        inference.covariance,
        expected_covariance,
        rtol=2e-3,
        atol=2e-5,
    )

    natural = inference.innovation_covariance_inference()
    expected_variance_se = variance * np.sqrt(2.0 / n_increments)
    np.testing.assert_allclose(natural.estimates, [variance])
    np.testing.assert_allclose(
        natural.standard_errors,
        [expected_variance_se],
        rtol=2e-3,
        atol=2e-5,
    )
    assert natural.dynamic_cross_covariance.shape == (1, 1)


def test_missing_data_inference_is_finite_and_immutable() -> None:
    rng = np.random.default_rng(2302)
    increments = 0.1 + rng.normal(scale=0.45, size=70)
    levels = np.concatenate([[2.0], 2.0 + np.cumsum(increments)])[:, None]
    levels[15:18, 0] = np.nan
    levels[42, 0] = np.nan
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=400,
    )
    model.fit(levels, identity_weights())

    inference = infer_exact_diffuse_kalman_starima(model)

    assert np.all(np.isfinite(inference.hessian))
    assert np.all(np.isfinite(inference.covariance))
    assert np.all(np.isfinite(inference.standard_errors))
    assert inference.stability_boundary_distance > 0.0
    assert inference.invertibility_boundary_distance > 0.0
    assert not inference.estimates.flags.writeable
    assert not inference.covariance.flags.writeable
    with pytest.raises(ValueError):
        inference.estimates[0] = 0.0


def test_singular_hessian_requires_explicit_generalized_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2303)
    increments = rng.normal(scale=0.5, size=50)
    levels = np.concatenate([[0.0], np.cumsum(increments)])[:, None]
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
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
    model = ExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=1,
        ma_order=0,
    )
    with pytest.raises(RuntimeError, match="fit must be called"):
        infer_exact_diffuse_kalman_starima(model)

    levels = np.arange(20.0)[:, None]
    model.fit(levels, identity_weights())
    with pytest.raises(ValueError, match="relative_step"):
        infer_exact_diffuse_kalman_starima(model, relative_step=0.0)
    with pytest.raises(ValueError, match="absolute_step"):
        infer_exact_diffuse_kalman_starima(model, absolute_step=np.inf)
    with pytest.raises(ValueError, match="rcond"):
        infer_exact_diffuse_kalman_starima(model, rcond=0.0)
