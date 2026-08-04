"""Filter an integrated level process with exact diffuse initialization."""

from __future__ import annotations

import numpy as np

from pystarmax import (
    StateSpaceModel,
    build_exact_integrated_state_space,
    exact_diffuse_filter,
)


def main() -> None:
    rng = np.random.default_rng(2026)
    drift = 0.15
    variance = 0.25
    increments = drift + rng.normal(scale=np.sqrt(variance), size=120)
    levels = np.cumsum(increments)[:, None]

    transformed_model = StateSpaceModel(
        transition=np.array([[0.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([drift]),
        innovation_covariance=np.array([[variance]]),
        ar_order=0,
        ma_order=0,
    )
    specification = build_exact_integrated_state_space(
        transformed_model,
        integration_order=1,
    )
    result = specification.filter(levels)

    print("Exact diffuse log likelihood:", result.log_likelihood)
    print("Initial diffuse rank:", result.initial_diffuse_rank)
    print("Diffuse observations:", result.n_diffuse_observations)
    print("Diffuse end time:", result.diffuse_end_time)
    print("Final diffuse rank:", result.final_diffuse_rank)
    print("Filtered levels:")
    print(result.filtered_observations[-5:])

    generic = exact_diffuse_filter(
        levels,
        specification.model,
        initial_state=specification.initial_state,
        initial_covariance=specification.initial_covariance,
        initial_diffuse_covariance=specification.initial_diffuse_covariance,
    )
    np.testing.assert_allclose(generic.filtered_state, result.filtered_state)


if __name__ == "__main__":
    main()
