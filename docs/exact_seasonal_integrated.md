# Seasonal exact diffuse integration foundation

Version 0.0.26 adds the fixed-parameter original-level state-space foundation
for ordinary and seasonal integration. It is the first stage toward a complete
seasonal exact diffuse maximum-likelihood model.

## Scale contract

Let the stationary transformed observation be

\[
x_t = \Delta^d\Delta_s^D y_t
    = (1-B)^d(1-B^s)^D y_t,
\]

with

\[
x_t = Z\beta_t,
\qquad
\beta_t = c + T\beta_{t-1} + R\eta_t,
\qquad
\eta_t\sim N(0,Q).
\]

The conditional seasonal Kalman API estimates the transformed sample after
removing differencing history. This module instead constructs an
**original-level exact diffuse likelihood**. The two likelihood conventions are
not interchangeable.

## Combined differencing polynomial

The package first computes

\[
\delta(B)=(1-B)^d(1-B^s)^D
         =1+\delta_1B+\cdots+\delta_mB^m,
\qquad
m=d+Ds.
\]

For positive seasonal order, the augmented state is

\[
\alpha_t=
\begin{bmatrix}
 y_t & y_{t-1} & \cdots & y_{t-m+1} & \beta_t
\end{bmatrix}^{\mathsf T}.
\]

The first location block obeys

\[
y_t=-\sum_{j=1}^{m}\delta_j y_{t-j}+Z\beta_t.
\]

The remaining level blocks shift the lag history forward, while the transformed
state follows its stationary transition.

This companion representation keeps every coefficient of the combined
ordinary-seasonal differencing polynomial visible. It avoids conditioning away
the initial seasonal cycles.

## Exact diffuse initialization

The original-level lag block has dimension

\[
m n=(d+Ds)n,
\]

where \(n\) is the number of locations. These coordinates receive the diffuse
covariance component

\[
P_{\infty,1}=
\begin{bmatrix}
 I_{mn} & 0\\
 0 & 0
\end{bmatrix}.
\]

The transformed state must be stationary. Its finite initial mean and covariance
are

\[
a_{\beta,1}=(I-T)^{-1}c,
\qquad
P_{\beta,1}=TP_{\beta,1}T^{\mathsf T}+RQR^{\mathsf T}.
\]

A nonstationary transformed subsystem is rejected instead of being hidden inside
another diffuse approximation.

## Ordinary-equivalence contract

When \(D=0\), the implementation delegates to the existing ordinary
`build_exact_integrated_state_space()` constructor. This preserves the existing
difference-level state coordinates and gives exact numerical equality for
filters and likelihoods already supported before version 0.0.26.

## API

```python
from pystarmax import build_exact_seasonal_integrated_state_space

specification = build_exact_seasonal_integrated_state_space(
    transformed_model,
    ordinary_integration_order=1,
    seasonal_integration_order=1,
    seasonal_period=12,
)

result = specification.filter(original_level_data)
print(result.log_likelihood)
print(result.initial_diffuse_rank)
```

Functional convenience routes are also available:

```python
from pystarmax import (
    exact_seasonal_integrated_filter,
    exact_seasonal_integrated_loglikelihood,
)
```

All stored polynomial, state, finite-covariance, and diffuse-covariance arrays
are immutable.

## Missing observations

The exact diffuse filter processes observed location-time cells sequentially.
A missing cell does not identify a diffuse direction. Fully missing rows perform
prediction only. Consequently, seasonal diffuse completion can occur later than
\(d+Ds-1\) when early observations are missing.

## Verification

The foundation is tested against:

- a closed-form scalar seasonal random walk;
- the explicit `(1-B)(1-B^2)` companion matrix;
- direct reconstruction of the complete transformed series;
- exact equality with ordinary integration when `D=0`;
- delayed diffuse-rank resolution under missing data;
- nonstationary transformed-state rejection;
- immutable public numerical arrays;
- the complete inherited cross-platform test suite.

## Deliberate boundary

Version 0.0.26 is a state-space and fixed-parameter likelihood foundation. It
does not yet provide:

- optimizer-facing seasonal exact diffuse MLE;
- seasonal exact diffuse smoothing or disturbance smoothing facades;
- seasonal exact diffuse observed-information inference;
- seasonal original-level forecast intervals;
- parameter-uncertainty propagation;
- sparse or parallel seasonal diffuse execution.

Those operations should be built on this common original-level state contract
rather than introducing separate incompatible augmentations.
