from __future__ import annotations

import numpy as np

from pystarmax import (
    ExactDiffuseKalmanSTARIMA,
    ExactDiffuseSmootherResult,
    SpatialWeights,
    exact_diffuse_smoother,
)


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def test_fitted_model_smooths_training_and_new_original_levels() -> None:
    rng = np.random.default_rng(211)
    increments = 0.12 + rng.normal(scale=0.35, size=120)
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

    training = model.smooth()

    assert isinstance(training, ExactDiffuseSmootherResult)
    assert training.filter_result is fitted.filter_result
    np.testing.assert_allclose(training.smoothed_observations, levels, atol=1e-10)

    new_levels = levels[-16:].copy()
    new_levels[5, 0] = np.nan
    new_result = model.smooth(new_levels)
    direct = exact_diffuse_smoother(model.filter(new_levels))

    assert new_result.filter_result is not fitted.filter_result
    np.testing.assert_allclose(new_result.smoothed_state, direct.smoothed_state)
    np.testing.assert_allclose(
        new_result.smoothed_covariance,
        direct.smoothed_covariance,
    )
    assert np.isfinite(new_result.smoothed_observations[5, 0])
