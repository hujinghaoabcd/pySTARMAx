"""Exact adjacent-time covariance for a diffuse local-level model."""

from __future__ import annotations

import numpy as np

from pystarmax import (
    StateSpaceModel,
    exact_diffuse_filter,
    exact_diffuse_lag_one_covariance,
    exact_diffuse_simulation_smoother,
)


def main() -> None:
    innovation_variance = 0.4
    model = StateSpaceModel(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.0]),
        innovation_covariance=np.array([[innovation_variance]]),
        ar_order=1,
        ma_order=0,
    )
    observations = np.array([[np.nan], [np.nan], [2.0]])
    filtered = exact_diffuse_filter(observations, model)
    moments = exact_diffuse_lag_one_covariance(filtered)

    print("Smoothed state means")
    print(moments.smoother_result.smoothed_state[:, 0])
    print("Smoothed state variances")
    print(moments.smoother_result.smoothed_covariance[:, 0, 0])
    print("Adjacent state covariances")
    print(moments.lag_one_covariance[:, 0, 0])
    print("Reconstructed state-disturbance variances")
    print(moments.state_disturbance_covariance[:, 0, 0])

    paths = exact_diffuse_simulation_smoother(
        filtered,
        n_simulations=20000,
        random_state=42,
    )
    empirical = []
    for time_index in range(moments.n_transitions):
        left = paths.state_paths[:, time_index, 0]
        right = paths.state_paths[:, time_index + 1, 0]
        empirical.append(
            np.mean(
                (left - left.mean()) * (right - right.mean()),
            )
        )

    print("Conditional-path empirical adjacent covariances")
    print(np.asarray(empirical))


if __name__ == "__main__":
    main()
