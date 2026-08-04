from __future__ import annotations

import numpy as np

from pystarmax import (
    ExactDiffuseDisturbanceResult,
    ExactDiffuseKalmanSTARIMA,
    SpatialWeights,
    exact_diffuse_disturbance_smoother,
)


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def test_fitted_model_smooths_training_and_new_innovation_disturbances() -> None:
    rng = np.random.default_rng(221)
    increments = 0.15 + rng.normal(scale=0.3, size=100)
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

    training = model.smooth_innovation_disturbances()
    direct_training = exact_diffuse_disturbance_smoother(model.smooth())

    assert isinstance(training, ExactDiffuseDisturbanceResult)
    assert training.smoother_result.filter_result is fitted.filter_result
    np.testing.assert_allclose(training.innovation_mean, direct_training.innovation_mean)
    np.testing.assert_allclose(
        training.innovation_covariance,
        direct_training.innovation_covariance,
    )
    np.testing.assert_allclose(
        training.state_disturbance_mean,
        direct_training.state_disturbance_mean,
    )

    new_levels = levels[-18:].copy()
    new_levels[6, 0] = np.nan
    new_result = model.smooth_innovation_disturbances(new_levels)
    direct_new = exact_diffuse_disturbance_smoother(model.smooth(new_levels))

    assert new_result.smoother_result.filter_result is not fitted.filter_result
    np.testing.assert_allclose(new_result.innovation_mean, direct_new.innovation_mean)
    np.testing.assert_allclose(
        new_result.innovation_covariance,
        direct_new.innovation_covariance,
    )
    assert np.all(np.isfinite(new_result.innovation_mean))
