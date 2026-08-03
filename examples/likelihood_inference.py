"""Estimate Kalman STARMA coefficients and observed-information uncertainty."""

from __future__ import annotations

import numpy as np

from pystarmax import KalmanSTARMA, SpatialWeights, simulate_starma

weights = SpatialWeights(
    matrices=(np.eye(2, dtype=float),),
    names=("W0",),
)

series = simulate_starma(
    phi=np.array([[0.45]], dtype=float),
    theta=np.empty((0, 1), dtype=float),
    weights=weights,
    n_steps=500,
    innovation_covariance=np.array(
        [[0.8, 0.2], [0.2, 1.1]],
        dtype=float,
    ),
    random_state=42,
)
series[40:450:17, 0] = np.nan
series[160, :] = np.nan

model = KalmanSTARMA(
    ar_order=1,
    ma_order=0,
    covariance_type="full",
    include_intercept=False,
    max_iter=600,
)
fit = model.fit(series, weights)
inference = model.infer(
    relative_step=2e-4,
    absolute_step=1e-6,
)

print(fit.summary())
print("\nObserved-information inference")
print(inference.summary())
print("\n95% dynamic coefficient intervals")
print(inference.confidence_intervals(level=0.95))
