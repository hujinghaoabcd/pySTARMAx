# Kalman maximum-likelihood estimation

Version 0.0.9 adds a dedicated `KalmanSTARMA` estimator. It maximizes the
Gaussian likelihood evaluated by the state-space and Kalman filtering core added
in version 0.0.8. The established conditional `STARMA.fit()` estimator remains
available and unchanged.

## Basic use

```python
from pystarmax import KalmanSTARMA

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=True,
)
result = model.fit(data, weights)

print(result.summary())
print(model.predict(steps=6))
```

The input convention remains `(time, location)`. Missing cells may be represented
by `NaN`; they are omitted from the corresponding Kalman measurement update.
They are not interpolated or replaced inside the likelihood.

## Dynamic parameters

For every positive temporal AR or MA order, the estimator includes all matrices
in the supplied ordered `SpatialWeights` collection. The model is

\[
z_t = c + \sum_i \sum_k \phi_{ik} W_k z_{t-i}
      + \varepsilon_t
      + \sum_j \sum_k \theta_{jk} W_k \varepsilon_{t-j}.
\]

The spatial matrices retain their supplied orientation. Symmetry and
commutativity are not assumed.

## Innovation covariance

Three covariance models are available:

- `scalar`: one variance shared by all locations;
- `diagonal`: one variance per location, with zero contemporaneous covariance;
- `full`: a positive-definite covariance represented by a lower-triangular
  Cholesky factor.

The Cholesky diagonal is exponentiated, so every optimized full covariance is
positive definite by construction. Off-diagonal Cholesky entries remain
unconstrained. Information criteria count all dynamic and covariance parameters.

## Starting values

Automatic starts use the existing conditional estimator when dynamic terms are
present. Missing cells are filled only for this starting-value calculation,
using each location's observed mean. The actual objective always receives the
original incomplete observation matrix.

Users may supply dynamic starting values and an initial covariance:

```python
result = model.fit(
    data,
    weights,
    start_params=dynamic_start,
    start_covariance=covariance_start,
)
```

`start_params` contains the optional common intercept followed by AR and MA
coefficients in temporal-major, spatial-minor order.

## Stationarity handling

With the default stationary initialization, candidates whose transition spectral
radius reaches `1 - stability_margin` receive a large feasibility penalty. The
final candidate is checked again before a result is returned.

This is an explicit feasibility strategy, not a smooth stationarity
reparameterization. Setting `enforce_stationarity=False` is mainly useful with
approximate diffuse initialization; stationary initialization itself still
requires a stable transition matrix.

## Result object

`KalmanSTARMAResult` records:

- dynamic coefficients and names;
- raw optimizer parameters and names;
- AR and MA coefficient matrices;
- the fitted innovation covariance;
- log likelihood, AIC, and BIC;
- observed-cell count and estimated-parameter count;
- optimizer convergence, iteration count, function evaluations, and message;
- transition spectral radius;
- the final `KalmanFilterResult`.

All numerical arrays are immutable. Convenience DataFrames expose the dynamic
coefficient vector and innovation covariance.

## Filtering and forecasting

```python
training_filter = model.filter()
new_filter = model.filter(new_incomplete_data)
forecast = model.predict(steps=12)
```

`predict()` propagates the final filtered state with future innovations set to
zero. It returns recursive conditional means on the stationary observation
scale.

## Current statistical scope

Included in 0.0.9:

- Gaussian Kalman maximum-likelihood estimation;
- complete and partially missing stationary observations;
- scalar, diagonal, and full innovation covariance;
- positive-definite full covariance parameterization;
- explicit optimizer diagnostics;
- stationarity feasibility checks;
- native filtering and conditional-mean prediction.

Not yet included:

- likelihood-Hessian standard errors;
- profile or likelihood-ratio intervals;
- a smooth stationarity parameterization;
- MA invertibility constraints;
- exact diffuse likelihood;
- state or disturbance smoothing;
- integrated and multiplicative seasonal maximum-likelihood wrappers;
- sparse state matrices and large-network optimization.
