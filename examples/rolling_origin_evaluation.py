"""Evaluate STARMA prediction intervals over rolling forecast origins."""

import numpy as np

from pystarmax import STARMA, lattice_weights, rolling_origin_evaluate, simulate_starma

weights = lattice_weights(1, 3)
series = simulate_starma(
    phi=np.array([[0.40, 0.10]]),
    theta=np.empty((0, 2)),
    weights=weights,
    n_steps=180,
    burnin=100,
    random_state=42,
)

evaluation = rolling_origin_evaluate(
    lambda: STARMA(ar_order=1),
    series,
    weights,
    initial_window=120,
    horizon=3,
    step=6,
    interval_method="conditional",
    level=0.90,
    interval_kwargs={"n_simulations": 500},
    random_state=7,
)

print(evaluation.metrics())
for horizon, metrics in enumerate(evaluation.metrics_by_horizon(), start=1):
    print(f"h={horizon}: {metrics}")
