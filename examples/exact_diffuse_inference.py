"""Validate exact diffuse random-walk observed information."""

from __future__ import annotations

import numpy as np

from pystarmax import ExactDiffuseKalmanSTARIMA, SpatialWeights


rng = np.random.default_rng(2300)
increments = 0.25 + rng.normal(scale=0.6, size=80)
levels = np.concatenate([[0.0], np.cumsum(increments)])[:, None]
weights = SpatialWeights(
    matrices=(np.eye(1, dtype=float),),
    names=("W0",),
)

model = ExactDiffuseKalmanSTARIMA(
    ar_order=0,
    integration_order=1,
    ma_order=0,
    covariance_type="scalar",
    include_intercept=True,
    max_iter=500,
    tol=1e-11,
)
fit = model.fit(
    levels,
    weights,
    start_params=np.array([np.mean(increments)]),
    start_covariance=float(np.var(increments)),
)
inference = model.likelihood_inference(
    relative_step=2e-4,
    absolute_step=1e-6,
)
natural = inference.innovation_covariance_inference()

n_increments = increments.size
variance = float(fit.innovation_covariance[0, 0])
expected_information = np.diag(
    [n_increments / variance, 2.0 * n_increments]
)
expected_variance_se = variance * np.sqrt(2.0 / n_increments)

print(inference.summary())
print("Observed-information Hessian:")
print(inference.hessian)
print("Natural innovation variance and standard error:")
print(natural.estimates, natural.standard_errors)

np.testing.assert_allclose(
    inference.hessian,
    expected_information,
    rtol=2e-3,
    atol=2e-3,
)
np.testing.assert_allclose(
    natural.standard_errors,
    [expected_variance_se],
    rtol=2e-3,
    atol=2e-5,
)
