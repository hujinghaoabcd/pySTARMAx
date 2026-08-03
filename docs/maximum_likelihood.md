# Kalman maximum-likelihood estimation

`KalmanSTARMA` is a dedicated stationary Gaussian maximum-likelihood estimator.
It maximizes the likelihood evaluated by the package's state-space and Kalman
filtering core. The established conditional `STARMA.fit()` estimator remains
available and separate.

Version 0.0.11 adds explicit autoregressive stationarity and moving-average
invertibility diagnostics and applies both criteria consistently to automatic
starting values, likelihood candidates, final fitted results, and
likelihood-Hessian inference.

## Basic use

```python
from pystarmax import KalmanSTARMA

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=True,
    enforce_stationarity=True,
    stability_margin=1e-6,
    enforce_invertibility=True,
    invertibility_margin=1e-6,
)
result = model.fit(data, weights)

print(result.summary())
print(model.admissibility().summary())
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

After conditional initialization, the AR and MA coefficient matrices are
checked independently. When the corresponding constraint is enabled, each block
is iteratively scaled until it lies safely inside its requested spectral-radius
region. This affects only automatic starts.

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
coefficients in temporal-major, spatial-minor order. User-supplied dynamic starts
are checked for shape and finiteness but are not silently projected or rescaled.

## Autoregressive stationarity

For lag matrices

\[
A_i = \sum_k \phi_{ik}W_k,
\]

the AR companion matrix has top block row `[A_1, ..., A_p]`. With
`enforce_stationarity=True`, candidates at or beyond

```text
spectral_radius >= 1 - stability_margin
```

receive a large feasibility penalty. The final candidate is checked again before
a result is returned.

Stationary Kalman initialization itself requires a stable transition matrix.
`enforce_stationarity=False` is therefore mainly meaningful with approximate
diffuse initialization or for explicit diagnostic experiments.

## Moving-average invertibility

pySTARMAx uses the positive MA sign

\[
\varepsilon_t + \sum_j B_j\varepsilon_{t-j}.
\]

The innovation inverse recursion consequently has companion top block row
`[-B_1, ..., -B_q]`. With `enforce_invertibility=True`, candidates at or beyond

```text
inverse_ma_spectral_radius >= 1 - invertibility_margin
```

receive the same feasibility treatment and the final candidate is hard-checked.

Set `enforce_invertibility=False` only when a non-invertible solution is
intentionally required for research or diagnosis. The fitted result still
computes and reports the inverse-MA radius, boundary distance, and invertibility
decision even when enforcement is disabled.

See [Stationarity and invertibility](admissibility.md) for the exact companion
matrices, scalar reductions, public diagnostics, and limitations.

## Feasibility strategy

The current optimizer uses L-BFGS-B and a large continuous penalty outside either
enabled spectral region. Squared AR and MA boundary excesses are accumulated
before the penalty is returned.

This is an explicit feasibility strategy, not a smooth bijective
stationarity/invertibility reparameterization. The objective is smooth inside the
admissible region but the spectral radius itself may be non-differentiable where
dominant eigenvalues exchange. Near-boundary optimization and curvature should
therefore be interpreted cautiously.

## Result object

`KalmanSTARMAResult` records:

- dynamic coefficients and names;
- raw optimizer parameters and names;
- AR and MA coefficient matrices;
- the fitted innovation covariance;
- log likelihood, AIC, and BIC;
- observed-cell count and estimated-parameter count;
- optimizer convergence, iteration count, function evaluations, and message;
- AR companion spectral radius and configured limit;
- inverse-MA companion spectral radius and configured limit;
- whether stationarity and invertibility were enforced;
- signed AR and MA boundary distances;
- stationary, invertible, and jointly admissible decisions;
- the final `KalmanFilterResult`.

All numerical arrays are immutable. Convenience DataFrames expose the dynamic
coefficient vector and innovation covariance. Full companion matrices and
complex eigenvalues are available through `model.admissibility()`.

## Filtering and forecasting

```python
training_filter = model.filter()
new_filter = model.filter(new_incomplete_data)
forecast = model.predict(steps=12)
```

`predict()` propagates the final filtered state with future innovations set to
zero. It returns recursive conditional means on the stationary observation
scale.

## Likelihood inference

```python
inference = model.infer(relative_step=1e-4)
print(inference.coefficient_table)
print(inference.stability_boundary_distance)
print(inference.invertibility_boundary_distance)
```

Finite-difference stencils use the same enabled AR and MA feasibility rules as
fitting. A perturbed point entering either penalty region is rejected instead of
differentiating the artificial penalty surface. See
[Likelihood inference](likelihood_inference.md).

## Current statistical scope

Included through 0.0.11:

- Gaussian Kalman maximum-likelihood estimation;
- complete and partially missing stationary observations;
- scalar, diagonal, and full innovation covariance;
- positive-definite full covariance parameterization;
- explicit optimizer diagnostics;
- AR stationarity and MA invertibility companion diagnostics;
- default dual feasibility enforcement with independent margins;
- fitted boundary distances and full companion eigensystems;
- observed-information Hessian inference;
- native filtering and conditional-mean prediction.

Not yet included:

- a smooth stationarity/invertibility parameterization;
- projection of arbitrary starts to the nearest admissible parameter vector;
- profile or likelihood-ratio intervals;
- covariance-element delta-method transforms;
- exact diffuse likelihood;
- state or disturbance smoothing;
- integrated and multiplicative seasonal maximum-likelihood wrappers;
- sparse state matrices and large-network optimization.
