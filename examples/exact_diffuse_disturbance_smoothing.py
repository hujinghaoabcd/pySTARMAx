"""Smooth exact diffuse process innovations across a missing level."""

from __future__ import annotations

import numpy as np

from pystarmax import (
    StateSpaceModel,
    exact_diffuse_disturbance_smoother,
    exact_diffuse_filter,
    exact_diffuse_smoother,
)


state_space = StateSpaceModel(
    transition=np.array([[1.0]]),
    design=np.array([[1.0]]),
    selection=np.array([[1.0]]),
    state_intercept=np.array([0.0]),
    innovation_covariance=np.array([[0.5]]),
    ar_order=1,
    ma_order=0,
)

levels = np.array([[1.0], [np.nan], [3.0]])
filtered = exact_diffuse_filter(levels, state_space)
smoothed = exact_diffuse_smoother(filtered)
disturbances = exact_diffuse_disturbance_smoother(smoothed)

print("Smoothed levels:")
print(smoothed.smoothed_observations)
print("Posterior innovation means:")
print(disturbances.innovation_mean)
print("Posterior innovation variances:")
print(np.diagonal(disturbances.innovation_covariance, axis1=1, axis2=2))

# The information smoother gives marginal innovation moments directly; it does
# not require unavailable exact diffuse lag-one state autocovariance.
# The observed two-step change is split equally across the two increments.
np.testing.assert_allclose(disturbances.innovation_mean[:, 0], [1.0, 1.0])
np.testing.assert_allclose(
    disturbances.innovation_covariance[:, 0, 0],
    [0.25, 0.25],
)
