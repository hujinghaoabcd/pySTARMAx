# Exact diffuse STARIMA maximum likelihood

Version 0.0.20 adds an optimizer-facing ordinary integrated estimator:

```python
from pystarmax import ExactDiffuseKalmanSTARIMA

model = ExactDiffuseKalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    covariance_type="full",
)
result = model.fit(level_observations, weights)
```

This estimator maximizes an exact diffuse Gaussian likelihood on the original
level observations. It is deliberately separate from `KalmanSTARIMA`, which
continues to maximize a conditional likelihood after ordinary differencing.

## Model

Let

\[
x_t=\Delta^d y_t
\]

follow the stationary transformed STARMA equation

\[
x_t
=c+\sum_{i=1}^{p}A_i x_{t-i}
+\eta_t+\sum_{j=1}^{q}M_j\eta_{t-j},
\qquad
\eta_t\sim\mathcal N(0,Q).
\]

The transformed dynamics are represented by

\[
\beta_t=c_\beta+T_\beta\beta_{t-1}+R_\beta\eta_t,
\qquad
x_t=Z_\beta\beta_t.
\]

For integration order `d`, pySTARMAx augments the state with

\[
\alpha_t=
\left[
 y_t^\top,
 (\Delta y_t)^\top,
 \ldots,
 (\Delta^{d-1}y_t)^\top,
 \beta_t^\top
\right]^\top.
\]

The integration directions receive exact diffuse covariance. The stationary
transformed state receives its finite stationary mean and covariance.

## Difference from conditional `KalmanSTARIMA`

`KalmanSTARIMA` first constructs \(\Delta^d y_t\), discards the first `d`
transformation rows from the likelihood, and conditions on the removed history:

\[
L_c
=
L\left(
\Delta^d y_{d+1:T}
\mid y_{1:d}
\right).
\]

`ExactDiffuseKalmanSTARIMA` instead evaluates the original levels through the
augmented state:

\[
L_D
=
L_D\left(y_{1:T}\right),
\]

where the unresolved level and lower-difference initial states are represented
by \(P_\infty\). The two likelihood values, AIC values, and BIC values are
therefore not interchangeable.

## Optimization coordinates

The optimizer coordinates are the same factor coordinates used by stationary
`KalmanSTARMA`:

1. optional scalar intercept;
2. `p * K` AR coefficients;
3. `q * K` positive-sign MA coefficients;
4. scalar, diagonal, or full-Cholesky covariance coordinates.

Here `K` is the number of supplied spatial-weight matrices.

The integration order adds latent states but no free coefficient. Therefore the
number of optimized parameters is

\[
\mathbf 1_c+K(p+q)+k_Q,
\]

where \(k_Q\) is the covariance-codec dimension.

## Candidate reconstruction

At every L-BFGS-B candidate, the estimator:

1. decodes the intercept, AR factors, MA factors, and innovation covariance;
2. constructs the stationary transformed STARMA state space;
3. checks the transformed AR spectral radius;
4. checks the positive-sign inverse-MA spectral radius;
5. builds the exact integrated original-level state space;
6. initializes the transformed subsystem from its stationary distribution;
7. filters the original observations with exact diffuse initialization;
8. returns the negative exact diffuse log likelihood.

This reconstruction prevents the optimizer from evaluating a conditional
recursion while reporting an exact diffuse objective.

## Starts

Initial dynamic parameters and covariance use the existing stationary
`KalmanSTARMA` start machinery on the differenced observations:

- a conditional STARMA start is attempted for dynamic terms;
- a regularized residual covariance initializes `Q`;
- AR starts are shrunk toward a stationary region when enforcement is enabled;
- MA starts are shrunk toward an invertible region when enforcement is enabled.

These starts do not determine the final likelihood convention. Every optimizer
candidate, including the start, is evaluated on the original-level exact
diffuse state space.

User-supplied starts are available:

```python
result = model.fit(
    level_observations,
    weights,
    start_params=start_params,
    start_covariance=start_covariance,
)
```

## Admissibility

The integration unit roots are intentional and are not subjected to the
stationarity test. Stationarity and invertibility apply only to the transformed
STARMA subsystem:

\[
\rho(C_{AR})<1-m_{AR},
\]

\[
\rho(C_{MA}^{-1})<1-m_{MA}.
\]

Candidates outside enabled limits receive an explicit feasibility penalty.
Final candidates are checked again and rejected if they violate an enforced
limit.

```python
diagnostics = model.admissibility()
print(diagnostics.summary())
```

## Result object

`ExactDiffuseKalmanSTARIMAResult` is immutable and contains:

- dynamic and raw optimizer parameters and names;
- intercept, AR factors, MA factors, and natural innovation covariance;
- covariance type and model order;
- exact diffuse log likelihood, AIC, and BIC;
- finite observation and diffuse-observation counts;
- optimizer convergence, iterations, evaluations, method, and message;
- AR and inverse-MA spectral radii and limits;
- enforcement flags;
- the complete `ExactDiffuseFilterResult`;
- the stationary transformed `StateSpaceModel`;
- the `ExactIntegratedStateSpace` specification.

```python
print(result.summary())
print(result.coefficients)
print(result.covariance)
print(result.filter_result.filtered_diffuse_rank)
```

AIC and BIC use the original-level exact diffuse likelihood:

\[
AIC=-2\ell_D+2k,
\]

\[
BIC=-2\ell_D+k\log n_{obs},
\]

where `n_obs` counts finite observed cells on the original scale.

## Filtering fitted parameters

```python
training = model.filter()
new_result = model.filter(new_level_observations)
```

Both calls use the fitted integrated state space and the same exact diffuse
initial decomposition. Filtering a new segment therefore starts a new diffuse
initialization; it does not continue the terminal posterior from the training
sample.

## State-space access

```python
level_state_space = model.to_state_space()
transformed_state_space = model.to_transformed_state_space()
```

The first contains original levels and lower ordinary differences. The second
contains only the stationary STARMA state for \(\Delta^d y_t\).

## Point forecasts

Original-level recursive means use the final filtered augmented state:

```python
level_forecast = model.predict(steps=12)
```

Highest-difference means use the terminal transformed-state block:

```python
transformed_forecast = model.predict_differenced(steps=12)
```

These are fixed-parameter point forecasts. Forecast interval simulation from
the exact diffuse estimator is not yet exposed.

## Fixed-interval state smoothing

Version 0.0.21 adds:

```python
smoothed = model.smooth()
new_smoothed = model.smooth(new_level_observations)
```

The exact diffuse smoother propagates ordinary and diffuse backward
information quantities separately and returns marginal smoothed state and
observation moments. Smoothing new data starts a new diffuse initialization;
it does not continue the training posterior. See
[Exact diffuse smoothing](exact_diffuse_smoothing.md).

## Primitive innovation and state-disturbance smoothing

Version 0.0.22 adds:

```python
disturbances = model.smooth_innovation_disturbances()
new_disturbances = model.smooth_innovation_disturbances(new_level_observations)
```

Posterior means and marginal covariance use the exact backward information
equations `Q R.T r_t` and `Q - Q R.T N_t R Q`. The route preserves
selection-nullspace uncertainty and does not require unavailable diffuse
lag-one state autocovariance. See
[Exact diffuse disturbance smoothing](exact_diffuse_disturbance_smoothing.md).

## Observed-information and natural covariance inference

Version 0.0.23 adds:

```python
inference = model.likelihood_inference(
    relative_step=1e-4,
    absolute_step=1e-6,
    rcond=1e-10,
)
natural = inference.innovation_covariance_inference()
```

Every finite-difference candidate rebuilds the transformed STARMA state,
integrated original-level state, and exact diffuse filter. The Hessian is
therefore curvature of the original-level exact diffuse objective rather than
the conditional differenced likelihood. A non-positive-definite or
rank-deficient Hessian raises by default; `allow_singular=True` explicitly
enables a labelled positive-eigenspace diagnostic generalized inverse.
Scalar, diagonal, and full-Cholesky innovation covariance coordinates use the
existing analytic natural-covariance Jacobian. See
[Exact diffuse likelihood inference](exact_diffuse_inference.md).

## Missing observations

`NaN` cells are allowed.

- fully missing rows perform prediction only;
- partially observed rows are updated location by location;
- missing initial levels delay diffuse-rank reduction;
- finite original-level cells determine `n_observations`;
- missing levels are not imputed before likelihood evaluation.

The differenced observations used only to generate starts may contain additional
`NaN` values because missingness propagates through the finite-difference
stencil. The objective itself uses the original observations.

## Closed-form reference cases

### Random walk with drift

For

\[
y_t=y_{t-1}+\mu+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,\sigma^2),
\]

the first observed level resolves the one-dimensional diffuse state. The
remaining exact likelihood is the Gaussian likelihood of increments. The MLEs
are

\[
\hat\mu
=\frac{1}{T-1}\sum_{t=2}^{T}\Delta y_t,
\]

\[
\hat\sigma^2
=\frac{1}{T-1}
\sum_{t=2}^{T}
(\Delta y_t-\hat\mu)^2.
\]

### Second-order integrated white noise

For

\[
\Delta^2 y_t=\mu+\eta_t,
\]

two observed levels resolve level and slope diffuse directions. The remaining
likelihood is the Gaussian likelihood of second differences, yielding the
corresponding sample mean and maximum-likelihood residual variance.

### Zero integration order

When `d=0`, no diffuse direction is added. The estimator uses the transformed
state's stationary mean and covariance and is numerically equivalent to
stationary `KalmanSTARMA` under identical data, starts, and optimizer settings.

## Validation references

The test suite checks:

- random-walk drift and variance against closed-form MLEs;
- second-order integrated drift and variance against second differences;
- `d=0` optimizer coordinates and likelihood against stationary
  `KalmanSTARMA`;
- missing initial levels and delayed diffuse completion;
- fitted filter and state-space identity contracts;
- original and transformed recursive forecasts;
- admissibility diagnostics;
- result-array immutability;
- constructor, fitted-state, data-length, and start validation;
- the complete inherited package test suite.

## Current limitations

- exact diffuse observed-information and natural innovation covariance
  inference are implemented, but analytic derivatives, robust covariance, and
  parameter uncertainty propagated into forecasts or smoothers are not;
- exact diffuse marginal state and primitive innovation/state-disturbance
  smoothing are available, but lag-one state autocovariance and cross-time
  disturbance covariance are not;
- seasonal diffuse state augmentation and smoothing are not implemented;
- forecast intervals are not yet exposed from this estimator;
- the current observation equation has no separate measurement-noise covariance;
- exact filtering processes locations sequentially;
- finite-difference rank decisions use `diffuse_tolerance`;
- dense state and covariance matrices can be expensive;
- optimizer starts use differenced data and may be weak for highly incomplete
  series;
- exogenous regressors and intervention variables are unsupported.
