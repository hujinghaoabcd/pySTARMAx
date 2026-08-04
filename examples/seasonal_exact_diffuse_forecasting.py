"""Seasonal exact-diffuse forecast paths and intervals."""

from __future__ import annotations

import numpy as np

from pystarmax import SeasonalExactDiffuseKalmanSTARIMA, SpatialWeights


def simulate_period_four_random_walk(
    *,
    n_rows: int,
    drift: float,
    scale: float,
    random_state: int,
) -> np.ndarray:
    """Simulate y_t = y_(t-4) + drift + eta_t."""
    generator = np.random.default_rng(random_state)
    values = np.zeros(n_rows, dtype=float)
    values[:4] = np.array([-0.5, 0.0, 0.4, 0.8])
    innovations = generator.normal(scale=scale, size=n_rows - 4)
    for index in range(4, n_rows):
        values[index] = values[index - 4] + drift + innovations[index - 4]
    return values[:, None]


def main() -> None:
    data = simulate_period_four_random_walk(
        n_rows=120,
        drift=0.15,
        scale=0.30,
        random_state=7,
    )
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
        include_intercept=True,
    )
    model.fit(
        data,
        weights,
        start_params=np.array([0.1]),
        start_covariance=np.array([[0.1]]),
    )

    original = model.predict_interval(
        steps=8,
        level=0.90,
        n_simulations=5000,
        random_state=42,
    )
    transformed = model.predict_differenced_interval(
        steps=8,
        level=0.90,
        n_simulations=5000,
        random_state=42,
    )

    print("Original-level posterior mean")
    print(original.mean[:, 0])
    print("Original-level 90% lower bound")
    print(original.lower[:, 0])
    print("Original-level 90% upper bound")
    print(original.upper[:, 0])
    print("Combined-difference posterior mean")
    print(transformed.mean[:, 0])


if __name__ == "__main__":
    main()
