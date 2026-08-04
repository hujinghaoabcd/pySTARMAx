# Seasonal exact-diffuse smoothing

Version 0.0.28 extends the fitted multiplicative seasonal exact-diffuse model
with fixed-interval state smoothing and primitive innovation/state-disturbance
smoothing on the original observation scale.

## Public API

```python
from pystarmax import SeasonalExactDiffuseKalmanSTARIMA

model = SeasonalExactDiffuseKalmanSTARIMA(
    ar_order=p,
    integration_order=d,
    ma_order=q,
    seasonal_ar_order=P,
    seasonal_integration_order=D,
    seasonal_ma_order=Q,
    seasonal_period=s,
    covariance_type="full",
)
result = model.fit(data, weights)

smoothed = model.smooth()
disturbances = model.smooth_innovation_disturbances()
```

The state smoother returns `ExactDiffuseSmootherResult`. The disturbance route
returns `ExactDiffuseDisturbanceResult`. Both are immutable and retain the exact
filter object used to produce the posterior.

Passing new data starts a fresh exact-diffuse initialization under the fitted
parameters:

```python
new_smoothed = model.smooth(new_data)
new_disturbances = model.smooth_innovation_disturbances(new_data)
```

This is not a continuation from the terminal training posterior.

## Why no separate seasonal recursion is needed

The exact-diffuse backward information recursion is defined for a general
linear Gaussian state-space model. It depends on the filtered finite and diffuse
covariance sequences, the transition and design matrices, and the scalar
sequential-update records retained by `ExactDiffuseFilterResult`.

The seasonal state introduced in 0.0.26 is already a valid generic state-space
model. It stores the original-level lag companion required by

\[
(1-B)^d(1-B^s)^D y_t
\]

and the stationary transformed STARMA state. Consequently, the same exact
backward recursion applies without replacing seasonal diffuse directions by a
large finite covariance and without deriving an incompatible second state
representation.

## State posterior

For state \(\alpha_t\), predicted finite covariance \(P_t\), predicted diffuse
covariance \(P_{\infty,t}\), and backward information quantities
\(r_t,r_{\infty,t},N_t,N_{1,t},N_{2,t}\), the smoother reconstructs the posterior
mean and covariance using the exact diffuse decomposition. The returned object
includes:

- `smoothed_state`;
- `smoothed_covariance`;
- `smoothed_observations`;
- `smoothed_observation_covariance`;
- finite and diffuse backward information arrays;
- reconstruction and symmetry diagnostics;
- pseudoinverse-use indicators.

Observed cells remain equal to their measurements in deterministic-observation
models. Missing cells are inferred from all available past and future
observations. The input array is never imputed before filtering.

## Primitive innovation posterior

The augmented seasonal state uses a full selection matrix \(R\) that maps the
primitive innovation \(\eta_t\) into both the original-level integration block
and the transformed dynamic state:

\[
\alpha_{t+1}=T\alpha_t+c+R\eta_t,
\qquad
\eta_t\sim\mathcal N(0,Q).
\]

The disturbance smoother computes

\[
E(\eta_t\mid y)=Q R^\top r_t,
\]

\[
\operatorname{Var}(\eta_t\mid y)
=Q-Q R^\top N_t RQ,
\]

and then maps those moments to the state-equation disturbance
\(w_t=R\eta_t\). This preserves uncertainty in primitive innovation directions
that cannot be identified from the state disturbance. It is not a pseudoinverse
recovery from an already-smoothed state path.

## Closed-form seasonal bridge reference

For a one-location seasonal random walk with period two,

\[
y_t=y_{t-2}+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,q),
\]

the even and odd subsequences are independent random walks. With observations

```text
[0, 10, missing, missing, 4, 14]
```

the exact posterior means are

```text
[0, 10, 2, 12, 4, 14]
```

and the two missing-level posterior variances are \(q/2\). The associated
identified bridge innovations have mean two and variance \(q/2\). The first
transition innovation remains unresolved by the diffuse initial seasonal levels
and retains variance \(q\).

This reference tests the seasonal decomposition, diffuse initialization,
backward state recursion, primitive innovation timing, and full selection-matrix
mapping simultaneously.

## Missing observations

Partially observed rows use only finite locations in the measurement update.
Fully missing rows perform prediction only. Smoothing uses future observations
to recover posterior moments but never changes the original observation mask.

When `data` is supplied to a fitted facade, every location must still satisfy
the estimator/filter validation contract. The fresh sample receives its own
initial diffuse rank determined by \(d\), \(D\), the seasonal period, and the
number of locations.

## Ordinary reduction

For `seasonal_integration_order=0`, `seasonal_ar_order=0`, and
`seasonal_ma_order=0`, the seasonal facade reduces to the ordinary exact-diffuse
model. Version 0.0.28 verifies equality of:

- smoothed state means;
- smoothed state covariances;
- primitive innovation means;
- primitive innovation covariances.

## Deliberate boundary

Version 0.0.28 does not expose lag-one state covariance for a nontrivial diffuse
phase. That requires a separately derived and independently validated diffuse
`L2` recursion. It also does not add seasonal likelihood inference, forecast
intervals, conditional simulation smoothing, parameter-aware paths, or sparse
execution.
