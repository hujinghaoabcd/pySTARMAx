"""Conditional simulation smoothing for seasonal exact-diffuse STARIMA."""

from __future__ import annotations

import numpy as np

from pystarmax import SeasonalExactDiffuseKalmanSTARIMA, SpatialWeights


def simulate_period_four_random_walk(
    *,
    n_rows: int,
    scale: float,
    random_state: int,
) -> np.ndarray:
    """Simulate y_t = y_(t-4) + eta_t."""
    generator = np.random.default_rng(random_state)
    values = np.zeros(n_rows, dtype=float)
    values[:4] = np.array([-0.5, 0.0, 0.4, 0.8])
    innovations = generator.normal(scale=scale, size=n_rows - 4)
    for index in range(4, n_rows):
        values[index] = values[index - 4] + innovations[index - 4]
    return values[:, None]


def main() -> None:
    data = simulate_period_four_random_walk(
        n_rows=100,
        scale=0.30,
        random_state=7,
    )
    data[42, 0] = np.nan
    data[63, 0] = np.nan
    weights = SpatialWeights(
        matrices=(np.eye(1),),
        names=("W0",),
    )

    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=4,
        covariance_type="scalar",
        include_intercept=False,
    )
    model.fit(
        data,
        weights,
        start_covariance=np.array([[0.1]]),
    )

    paths = model.simulate_smoothing_paths(
        n_simulations=2000,
        random_state=42,
    )

    print("Conditional means at missing observations")
    print(paths.observation_paths[:, [42, 63], 0].mean(axis=0))
    print("Conditional standard deviations at missing observations")
    print(paths.observation_paths[:, [42, 63], 0].std(axis=0))
    print("Maximum observed-cell residual")
    print(paths.maximum_constraint_residual)
    print("Maximum marginal mean discrepancy")
    print(paths.maximum_mean_discrepancy)

    seasonal_difference = (
        paths.observation_paths[:, 4:, 0]
        - paths.observation_paths[:, :-4, 0]
    )
    np.testing.assert_allclose(
        paths.transformed_observation_paths[:, 4:, 0],
        seasonal_difference,
        atol=1e-10,
    )


if __name__ == "__main__":
    main()
