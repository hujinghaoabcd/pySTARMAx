# Seasonal STARIMA

pySTARMAx provides two multiplicative seasonal routes with the shared order
convention

\[
(p,d,q)\times(P,D,Q)_s.
\]

1. `SeasonalSTARIMA` uses nonlinear conditional least squares and supports
   conditional and bootstrap forecast intervals.
2. `SeasonalKalmanSTARIMA` uses a Gaussian Kalman likelihood for the combined
   ordinary-seasonally transformed process, supports missing observations,
   filtering, RTS smoothing, original innovation smoothing, expanded
   admissibility checks, and full scalar/diagonal/Cholesky innovation
   covariance.

Both transform the original process through

\[
w_t=(1-B)^d(1-B^s)^D z_t.
\]

`predict_differenced()` remains on this transformed scale and `predict()`
restores the original scale from ordinary terminal anchors and seasonal cycle
histories.

## Multiplicative matrix-polynomial convention

Let

\[
A_i=\sum_{\ell}\phi_{i\ell}W_\ell,
\qquad
S_r=\sum_{\ell}\Phi_{r\ell}W_\ell,
\]

and define ordinary and seasonal moving-average factors `M_j` and `N_u`
analogously. pySTARMAx uses

\[
\left(I-\sum_r S_rB^{rs}\right)
\left(I-\sum_i A_iB^i\right)w_t
=
\left(I+\sum_u N_uB^{us}\right)
\left(I+\sum_j M_jB^j\right)\varepsilon_t.
\]

The seasonal factor is on the left. Expanding the autoregressive side gives
ordinary terms `+A_i`, seasonal terms `+S_r`, and cross terms

\[
-S_rA_iB^{rs+i}.
\]

The positive-sign moving-average product gives cross terms

\[
+N_uM_jB^{us+j}.
\]

Matrix order is preserved. With non-commuting spatial operators, `S_r @ A_i`
is not interchangeable with `A_i @ S_r`.

Cross-lag matrices are applied directly. They are not projected back onto the
supplied `SpatialWeights` basis because that basis need not be closed under
matrix multiplication.

## Factor parameters and parameter counting

A freely expanded recursion could estimate each cross-lag matrix independently,
but that would no longer be a multiplicative seasonal model. Both seasonal
estimators optimize only the factor coefficients:

- `ar.t1.W0`, `ar.t1.W1`, ... for ordinary AR factors;
- `sar.t24.W0`, ... for seasonal AR factors;
- `ma.t1.W0`, ... for ordinary MA factors;
- `sma.t24.W0`, ... for seasonal MA factors.

For `K` spatial weights, the dynamic factor count is

\[
\mathbf 1_c+K(p+P+q+Q).
\]

Induced cross-lag matrices are deterministic functions of these factors and are
not counted as independent parameters in AIC or BIC.

## Conditional least-squares route

```python
from pystarmax import SeasonalSTARIMA

conditional = SeasonalSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=1,
    seasonal_period=24,
    include_intercept=False,
)
conditional_result = conditional.fit(observations, weights)

print(conditional_result.summary())
print(conditional.predict_differenced(steps=24))
print(conditional.predict(steps=24))
print(conditional.fitted_original())
print(conditional.predict_interval(steps=24, random_state=42))
```

This route estimates factor parameters by nonlinear conditional least squares
with zero pre-sample innovations. It also supports residual and parametric
bootstrap intervals through the package bootstrap API.

## Gaussian Kalman route

```python
from pystarmax import SeasonalKalmanSTARIMA

kalman = SeasonalKalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=1,
    seasonal_period=24,
    covariance_type="full",
    include_intercept=False,
    enforce_stationarity=True,
    enforce_invertibility=True,
)
kalman_result = kalman.fit(incomplete_observations, weights)

print(kalman_result.summary())
print(kalman.admissibility().summary())
print(kalman.predict_differenced(steps=24))
print(kalman.predict(steps=24))
print(kalman.smooth().smoothed_observations)
```

`SeasonalKalmanSTARIMA` constructs arbitrary-lag companion matrices from the
complete multiplicative expansion. Gaps between lags are represented by zero
matrix blocks. Stationarity and invertibility are checked on the complete
expanded AR recursion and the inverse of the complete positive-sign MA
recursion.

The likelihood is conditional on the history removed by ordinary and seasonal
differencing:

\[
\ell_c(\vartheta;z_{1:T})
=
\ell\left(
\vartheta;(1-B)^d(1-B^s)^D z_{d+Ds+1:T}
\mid\mathcal H_{d+Ds}
\right).
\]

It is not an exact diffuse integrated seasonal level-state likelihood.

See [Multiplicative seasonal Kalman STARIMA](seasonal_maximum_likelihood.md) for
state-space construction, covariance options, constraints, missing-data rules,
and result fields.

## Combined differencing state

`seasonal_difference()` stores the final complete seasonal cycle at each lower
seasonal-difference level. For one seasonal difference,

\[
z_t=\Delta_s z_t+z_{t-s}.
\]

Forecast inversion uses a rolling queue containing the final `s` values. Higher
seasonal orders apply the same recursion from the highest seasonal-difference
level downward.

`combined_difference()` applies ordinary differencing first and seasonal
differencing second. `CombinedDifferencingState.inverse_forecast()` reverses
those operations in the opposite order: seasonal histories are restored first,
then ordinary terminal levels.

The Kalman estimator follows the same convention. Original-scale forecasting is
refused when any required ordinary anchor or seasonal history contains a missing
value, while transformed-scale forecasting remains available.

## Missing observations

Neither route imputes level observations before differencing. In the Kalman
route, `NaN` values propagate through the complete ordinary-seasonal stencil and
are then handled by the transformed measurement update:

- finite locations participate in the update;
- missing locations are omitted;
- a fully missing transformed row performs prediction only;
- only finite transformed cells contribute to the likelihood.

The Kalman result reports original and transformed missing-cell counts.

## Auditing the expansion

`expand_multiplicative_operators()` exposes the actual factor-induced lag
matrices:

```python
from pystarmax import expand_multiplicative_operators

terms = expand_multiplicative_operators(
    ordinary_phi,
    seasonal_phi,
    weights,
    seasonal_period=24,
    kind="ar",
)
for term in terms:
    print(term.lag, term.label, term.matrix)
```

The Kalman result additionally stores aggregated `ar_lags`, `ar_matrices`,
`ma_lags`, and `ma_matrices` after terms at equal temporal lags are combined.

## Simulation

`simulate_seasonal_starma()` generates the stationary transformed process.
`simulate_seasonal_starima()` then applies ordinary and seasonal integration,
using zero initial histories by default or a supplied
`CombinedDifferencingState`.

## Current limits

Conditional route:

- nonlinear estimation uses zero pre-sample innovations;
- likelihood, AIC, and BIC use the conditional scalar-innovation approximation;
- local Jacobian standard errors do not include all nonlinear uncertainty;
- intervals condition on fitted factors unless bootstrap refitting is used.

Kalman route:

- likelihood is conditional on the differencing history, not exact diffuse;
- observed-information Hessian and natural covariance inference are not yet
  exposed for seasonal factor parameters;
- original-scale forecast intervals are unavailable;
- original-scale filtered and smoothed level-state distributions are not
  returned;
- dense companion states can grow rapidly with seasonal period and order.
