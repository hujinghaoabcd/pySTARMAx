# Fixed-interval state smoothing

Version 0.0.13 adds a Rauch--Tung--Striebel (RTS) fixed-interval smoother for the
linear Gaussian state-space representation used by `KalmanSTARMA`.

Filtering conditions on observations available up to time \(t\). Smoothing uses
the complete observation interval \(1{:}T\), so future observations can refine
latent states and missing observation blocks at earlier times.

## State convention

The state equation is

\[
\alpha_t = d + T\alpha_{t-1} + w_t,
\qquad
w_t \sim \mathcal N(0,Q_\alpha),
\]

and the observation equation is

\[
y_t = Z\alpha_t.
\]

For STARMA, \(Q_\alpha=R\Sigma R^\top\), where \(R\) is the state selection
matrix and \(\Sigma\) is the location-level innovation covariance.

The smoother consumes a complete `KalmanFilterResult`; it does not rerun or
modify parameter estimation.

## RTS recursion

Let \(a_{t|t}\) and \(P_{t|t}\) be the filtered state mean and covariance, and
let \(a_{t+1|t}\) and \(P_{t+1|t}\) be the next predicted mean and covariance.
The smoothing gain is

\[
J_t = P_{t|t}T^\top P_{t+1|t}^{+},
\]

where \((\cdot)^+\) is an ordinary inverse when full rank and a documented
positive-eigenspace pseudoinverse when the prediction covariance is rank
deficient.

The backward recursion is

\[
a_{t|T}
= a_{t|t} + J_t(a_{t+1|T}-a_{t+1|t}),
\]

\[
P_{t|T}
= P_{t|t} + J_t(P_{t+1|T}-P_{t+1|t})J_t^\top.
\]

The final smoothed state equals the final filtered state. The implementation
symmetrizes covariance matrices and clips only numerically tiny negative
eigenvalues. A materially indefinite result raises rather than being silently
repaired.

## Lag-one covariance

The result stores

\[
C_{t,t+1|T}
= \operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid y_{1:T})
= J_tP_{t+1|T}.
\]

This cross covariance is required for state-equation disturbance moments and for
later EM or diagnostic extensions.

## State-equation disturbance moments

For the transition from time \(t\) to \(t+1\), define

\[
w_{t+1}=\alpha_{t+1}-d-T\alpha_t.
\]

Its smoothed conditional mean is

\[
E(w_{t+1}\mid y_{1:T})
= a_{t+1|T}-d-Ta_{t|T}.
\]

Its conditional covariance is

\[
\begin{aligned}
\operatorname{Var}(w_{t+1}\mid y_{1:T})
={}&P_{t+1|T}+TP_{t|T}T^\top\\
&-C_{t,t+1|T}^\top T^\top
-TC_{t,t+1|T}.
\end{aligned}
\]

These arrays are named `state_disturbance_mean` and
`state_disturbance_covariance`.

!!! warning
    A state-equation disturbance is not automatically identical to the original
    location-level innovation. The state disturbance is `R @ eta_t`. Version
    0.0.14 adds a separate covariance-weighted conditional-Gaussian map to the
    original innovation. It does not apply a naive inverse of `R`.

See [Original innovation smoothing](innovation_smoothing.md) for that map,
selection-matrix null-space uncertainty, and support diagnostics.

## Usage with a fitted model

```python
import numpy as np
from pystarmax import KalmanSTARMA

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
)
model.fit(training_data, weights)

incomplete = training_data.copy()
incomplete[20:24, 1] = np.nan
incomplete[50, :] = np.nan

smoothed = model.smooth(incomplete)

print(smoothed.smoothed_observations)
print(smoothed.smoothed_observation_covariance)
print(smoothed.state_disturbance_mean)
print(smoothed.prediction_rank)
print(smoothed.used_pseudoinverse)
```

`KalmanSTARMA.smooth()` first calls the fitted model's existing `filter()` route.
Calling it without data smooths the training sample. Supplying a new matrix uses
the fitted parameters and supports the same partial-location and fully missing
rows as filtering.

Original innovations can then be obtained through the integrated route:

```python
innovation_result = model.smooth_innovation_disturbances(incomplete)
print(innovation_result.innovation_mean)
print(innovation_result.unresolved_covariance)
```

## Low-level usage

```python
from pystarmax import kalman_filter, kalman_smoother

filtered = kalman_filter(observations, state_space)
smoothed = kalman_smoother(filtered, rcond=1e-10)
```

The `rcond` threshold determines the retained positive eigenspace of a singular
predicted covariance. Each transition reports its numerical `prediction_rank`
and whether a pseudoinverse was used.

## Result fields

`KalmanSmootherResult` stores immutable read-only arrays:

- `smoothed_state` and `smoothed_covariance`;
- `smoothing_gain`;
- `lag_one_covariance`;
- `state_disturbance_mean` and `state_disturbance_covariance`;
- `prediction_rank` and `used_pseudoinverse`;
- the original `filter_result` and state-space `model`.

Convenience properties provide `smoothed_observations` and
`smoothed_observation_covariance`.

## Validation references

The implementation is checked against more than qualitative smoothness:

- a scalar AR(1) single-gap Gaussian bridge with closed-form conditional mean and
  variance;
- a contiguous missing block compared with direct joint-Gaussian conditioning;
- exact state-disturbance recovery for fully observed scalar AR(1) data;
- final-state equality, covariance reduction, rank-deficient prediction, fitted
  model integration, immutability, and one-time-point edge cases.

## Current scope

Included in 0.0.13:

- fixed-parameter RTS state smoothing;
- partial and fully missing observations through the forward filter;
- lag-one state covariance;
- state-equation disturbance moments;
- explicit rank-deficiency diagnostics.

Added separately in 0.0.14:

- original location-level innovation posterior means and marginal covariances;
- covariance-weighted conditioning through `Q R.T (R Q R.T)+`;
- retained unresolved innovation covariance for non-injective selection maps;
- process-rank, pseudoinverse, and support-residual diagnostics.

Not yet included:

- cross-time original innovation covariance;
- the disturbance entering the first stored state from initialization;
- exact diffuse smoothing;
- simulation smoothing;
- parameter-uncertainty propagation into smoothed states or innovations;
- integrated or multiplicative seasonal MLE wrappers;
- sparse state matrices for large spatial systems.
