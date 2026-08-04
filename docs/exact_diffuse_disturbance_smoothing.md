# Exact diffuse disturbance smoothing

Version 0.0.22 adds posterior moments for primitive process innovations and
their state-equation images under exact diffuse initialization.

This stage follows the exact diffuse fixed-interval state smoother introduced in
0.0.21. It does **not** approximate the diffuse covariance with a large finite
number, and it does **not** require lag-one smoothed state covariance.

## State equation and indexing

Let

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_{t+1},
\qquad
\eta_{t+1}\sim\mathcal N(0,Q).
\]

The disturbance result uses transition index `t` for the innovation entering
state \(\alpha_{t+1}\). A series with \(n\) time points therefore returns
\(n-1\) disturbance rows.

The primitive innovation \(\eta_t\) lives in the covariance space of \(Q\).
The state-equation disturbance is

\[
w_t=R\eta_t.
\]

The state dimension can exceed the primitive innovation dimension because the
state may contain integration, autoregressive, or moving-average blocks.

## Backward information formula

The exact diffuse state smoother retains the ordinary backward information
vector \(r_t\) and matrix \(N_t\), in addition to the diffuse terms needed for
state moments. Conditional on all observations,

\[
\operatorname E(\eta_{t+1}\mid Y)
=QR^\top r_t,
\]

and

\[
\operatorname{Var}(\eta_{t+1}\mid Y)
=Q-QR^\top N_tRQ.
\]

The corresponding state-equation disturbance moments are

\[
\operatorname E(w_{t+1}\mid Y)
=R\operatorname E(\eta_{t+1}\mid Y),
\]

\[
\operatorname{Var}(w_{t+1}\mid Y)
=R\operatorname{Var}(\eta_{t+1}\mid Y)R^\top.
\]

These equations use the exact backward information quantities directly. No
smoothed lag-one state autocovariance is reconstructed.

## Why lag-one covariance is not required

A tempting alternative is to calculate

\[
w_{t+1}=\alpha_{t+1}-c-T\alpha_t
\]

from marginal smoothed states. Its covariance requires
\(\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y)\). During the exact diffuse
phase, that autocovariance recursion requires an additional nontrivial
higher-order \(L_2\) term.

The univariate diffuse smoother in Statsmodels also leaves this autocovariance
path unimplemented and documents the same limitation. Version 0.0.22 therefore
uses the disturbance-smoother information equations above rather than applying
an ordinary RTS lag-one formula outside its valid regime.

## Low-level API

```python
from pystarmax import (
    exact_diffuse_disturbance_smoother,
    exact_diffuse_filter,
    exact_diffuse_smoother,
)

filtered = exact_diffuse_filter(observations, state_space)
smoothed = exact_diffuse_smoother(filtered)
disturbances = exact_diffuse_disturbance_smoother(smoothed)

print(disturbances.innovation_mean)
print(disturbances.innovation_covariance)
print(disturbances.state_disturbance_mean)
print(disturbances.state_disturbance_covariance)
```

The fitted estimator facade is:

```python
disturbances = fitted_model.smooth_innovation_disturbances()
new_disturbances = fitted_model.smooth_innovation_disturbances(new_levels)
```

Calling the method with new data starts a new exact diffuse filtering and
smoothing problem under the fitted parameters. It does not continue the
terminal training posterior.

## Immutable result

`ExactDiffuseDisturbanceResult` contains:

- `smoother_result`, the exact diffuse state smoother used as input;
- `innovation_mean`, shape `(n_time - 1, innovation_dim)`;
- `innovation_covariance`, shape
  `(n_time - 1, innovation_dim, innovation_dim)`;
- `state_disturbance_mean`, shape `(n_time - 1, state_dim)`;
- `state_disturbance_covariance`, shape
  `(n_time - 1, state_dim, state_dim)`;
- maximum innovation- and state-covariance stabilization corrections;
- the numerical tolerance used by the covariance guard.

All public arrays are copied, contiguous, finite, and read-only.

## Selection null space

If a primitive innovation direction is not loaded into the state by \(R\), the
observations cannot identify it through the state equation. The formula

\[
Q-QR^\top N_tRQ
\]

retains that unresolved prior uncertainty automatically. The implementation
does not invert \(R\), drop its null space, or force a minimum-norm innovation.

This distinction matters when primitive innovations are more detailed than the
state disturbance they generate.

## Missing observations

Missing cells are handled upstream by exact diffuse filtering and smoothing.
The disturbance posterior therefore inherits these contracts:

- a missing scalar cell performs no measurement update;
- a fully missing row still receives information from later observations;
- leading missing levels can leave early process innovations unresolved;
- a missing bridge can split an observed total change across multiple
  innovations with nonzero posterior covariance;
- no missing observation is imputed before the state-space recursion.

## Covariance policy

For every posterior covariance:

- the matrix is symmetrized;
- floating-point-scale negative eigenvalues are clipped to zero;
- the largest correction is reported;
- materially indefinite covariance raises `LinAlgError`;
- no arbitrary diffuse scale or diagonal jitter is added.

## Validation references

Tests cover:

1. fully observed random-walk innovations;
2. a missing random-walk bridge with closed-form conditional increment moments;
3. leading missing diffuse levels with unresolved prior innovation variance;
4. equality with the ordinary stationary RTS innovation smoother when
   \(P_\infty=0\);
5. a valid rank-deficient selection matrix that preserves primitive innovation
   null-space variance;
6. single-observation empty transition arrays;
7. positive-semidefinite covariance and immutable arrays;
8. validation errors and tolerance handling;
9. fitted-model smoothing for training and newly initialized data;
10. the complete inherited package suite.

Core CI #472 reported 193 passing tests, 87.19% total branch coverage, and
88.1% coverage for `exact_diffuse_disturbance_smoothing.py`.

## Scope and limitations

Available in 0.0.22:

- primitive process innovation posterior mean and marginal covariance;
- state-equation disturbance posterior mean and marginal covariance;
- exact diffuse, missing-data-aware fitted-model facade;
- selection-nullspace uncertainty preservation;
- immutable numerical diagnostics.

Not yet available:

- exact diffuse lag-one state autocovariance;
- cross-time disturbance covariance;
- exact diffuse simulation smoothing;
- measurement-disturbance smoothing with a separate observation-noise matrix;
- seasonal diffuse disturbance smoothing;
- parameter-uncertainty propagation;
- exact diffuse observed-information inference.

## References

- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press.
- Koopman, S. J. (1997). Exact initial Kalman filtering and smoothing for
  nonstationary time series models. *Journal of the American Statistical
  Association*, 92(440), 1630–1638.
- Statsmodels univariate exact diffuse smoothing source, used as an independent
  implementation-boundary cross-check for the unavailable diffuse state
  autocovariance recursion.
