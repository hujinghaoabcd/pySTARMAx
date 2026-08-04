"""Observed-information inference for a seasonal exact-diffuse model."""

from __future__ import annotations

import numpy as np

from pystarmax import SeasonalExactDiffuseKalmanSTARIMA, SpatialWeights


def seasonal_random_walk(
    increments: np.ndarray,
    *,
    initial: tuple[float, float] = (0.0, 4.0),
) -> np.ndarray:
    levels = np.empty(increments.size + 2, dtype=float)
    levels[:2] = initial
    for index, increment in enumerate(increments, start=2):
        levels[index] = levels[index - 2] + increment
    return levels[:, None]


def main() -> None:
    rng = np.random.default_rng(29)
    increments = 0.25 + rng.normal(scale=0.6, size=72)
    observations = seasonal_random_walk(increments)
    observations[24:27, 0] = np.nan

    weights = SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )
    model = SeasonalExactDiffuseKalmanSTARIMA(
        ar_order=0,
        integration_order=0,
        ma_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_ma_order=0,
        seasonal_period=2,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
        tol=1e-10,
    )
    fitted = model.fit(observations, weights)
    inference = model.likelihood_inference(
        relative_step=2e-4,
        absolute_step=1e-6,
    )
    natural_covariance = inference.innovation_covariance_inference()

    print(fitted.summary())
    print()
    print(inference.summary())
    print()
    print(natural_covariance.summary())
    print()
    print("Hessian rank:", inference.rank)
    print("Hessian condition number:", inference.condition_number)
    print("Natural covariance standard errors:", natural_covariance.standard_errors)


if __name__ == "__main__":
    main()
