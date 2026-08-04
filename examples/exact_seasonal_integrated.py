"""Build and filter a seasonal exact diffuse original-level state space."""

from __future__ import annotations

import numpy as np

from pystarmax import (
    StateSpaceModel,
    build_exact_seasonal_integrated_state_space,
)

transformed_model = StateSpaceModel(
    transition=np.array([[0.35]]),
    design=np.array([[1.0]]),
    selection=np.array([[1.0]]),
    state_intercept=np.array([0.1]),
    innovation_covariance=np.array([[0.2]]),
    ar_order=1,
    ma_order=0,
)

# A short scalar series with one ordinary and one period-four seasonal unit root.
data = np.array(
    [
        [10.0],
        [10.4],
        [10.7],
        [11.1],
        [11.6],
        [12.2],
        [12.5],
        [13.0],
        [13.7],
        [14.1],
    ]
)

specification = build_exact_seasonal_integrated_state_space(
    transformed_model,
    ordinary_integration_order=1,
    seasonal_integration_order=1,
    seasonal_period=4,
)
result = specification.filter(data)

print("combined polynomial:", specification.polynomial_coefficients)
print("integration degree:", specification.integration_degree)
print("initial diffuse rank:", result.initial_diffuse_rank)
print("diffuse end time:", result.diffuse_end_time)
print("log likelihood:", result.log_likelihood)
