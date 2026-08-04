# Exact diffuse forecast intervals

Version 0.0.24 adds fixed-parameter Gaussian forecast intervals to
`ExactDiffuseKalmanSTARIMA`.

The implementation forecasts directly from the original-level augmented state
used by the exact diffuse likelihood. It does not difference the fitted sample,
construct a separate conditional likelihood, or approximate unresolved diffuse
uncertainty with a large finite variance.

## Public API

Original-level intervals:

```python
interval = fitted_model.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

Highest ordinary-difference intervals:

```python
differenced = fitted_model.predict_differenced_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

Low-level original-level entry points:

```python
from pystarmax import (
    exact_diffuse_forecast_interval,
    simulate_exact_diffuse_forecast_paths,
)

paths = simulate_exact_diffuse_forecast_paths(
    fitted_model.filter(),
    steps=12,
    n_simulations=5000,
    random_state=2026,
)

interval = exact_diffuse_forecast_interval(
    fitted_model.filter(),
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

The returned immutable `ForecastInterval` contains `mean`, `lower`, `upper`,
`level`, `method`, `n_simulations`, and `refit_parameters`.

## Exact diffuse terminal state

The exact diffuse filter represents the state covariance as

\[
P_t(\kappa)=P_{\ast,t}+\kappa P_{\infty,t},
\qquad \kappa\rightarrow\infty.
\]

During filtering, observed cells remove diffuse directions. At the final time,
forecast simulation is valid only when

\[
\operatorname{rank}(P_{\infty,T\mid T})=0.
\]

Then the terminal state has the proper Gaussian posterior

\[
\alpha_T\mid Y_{1:T}
\sim
\mathcal N(a_{T\mid T},P_{\ast,T\mid T}).
\]

When the final diffuse rank is positive, the terminal distribution remains
improper in at least one state direction. Version 0.0.24 raises a `RuntimeError`
instead of:

- replacing diffuse variance with an arbitrary large number;
- dropping the unresolved direction;
- using only the finite covariance component;
- returning deceptively narrow intervals.

This refusal is an inferential requirement, not merely a numerical guard.

## Forecast path recursion

For the fitted original-level state-space system

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_{t+1},
\qquad
\eta_{t+1}\sim\mathcal N(0,Q),
\]

\[
y_{t+1}=Z\alpha_{t+1},
\]

each Monte Carlo path starts with

\[
\alpha_T^{(b)}
=
a_{T\mid T}+L_T z_0^{(b)},
\qquad
z_0^{(b)}\sim\mathcal N(0,I),
\]

where

\[
L_TL_T^\top=P_{\ast,T\mid T}.
\]

Future states follow

\[
\eta_{T+h}^{(b)}\sim\mathcal N(0,Q),
\]

\[
\alpha_{T+h}^{(b)}
=
c+T\alpha_{T+h-1}^{(b)}+R\eta_{T+h}^{(b)},
\]

and original-level observations are

\[
y_{T+h}^{(b)}=Z\alpha_{T+h}^{(b)}.
\]

The interval therefore contains both:

1. finite uncertainty in the final filtered state; and
2. future process-innovation uncertainty.

It conditions on fitted parameters.

## Why no inverse differencing is required

Conditional `KalmanSTARIMA` estimates a stationary model for transformed data
and reconstructs original levels after forecasting. Its original-scale forecast
intervals must inverse-difference every simulated transformed path.

`ExactDiffuseKalmanSTARIMA` instead filters an augmented original-level state.
For ordinary integration order `d`, that state contains

\[
[y_t,\Delta y_t,\ldots,\Delta^{d-1}y_t,\beta_t],
\]

where `beta_t` is the stationary transformed STARMA state. The state transition
already performs the integration recursion. Consequently, projecting simulated
augmented states through the model design matrix directly returns original
levels.

This route preserves:

- level and lower-difference posterior uncertainty;
- cross-covariance between integrated and transformed state blocks;
- future accumulation through the augmented transition matrix;
- the exact original-level state convention used by fitting and filtering.

## Highest ordinary-difference intervals

The augmented exact state also contains the stationary transformed STARMA
substate. Let

\[
J_d=[0\;\cdots\;0\;Z_\beta]
\]

select that substate and apply its transformed observation design. Then

\[
\Delta^d y_{T+h}^{(b)}=J_d\alpha_{T+h}^{(b)}.
\]

`predict_differenced_interval()` projects the same simulated augmented state
paths through `J_d`. It does not run a separate conditional filter and therefore
retains covariance between the transformed substate and the integrated blocks
while drawing the terminal state.

The interval mean is the existing deterministic
`predict_differenced()` recursion.

## Point forecast convention

The reported `ForecastInterval.mean` is always the deterministic recursive
forecast from the final filtered mean:

\[
\bar\alpha_{T+h}=c+T\bar\alpha_{T+h-1},
\qquad
\bar y_{T+h}=Z\bar\alpha_{T+h}.
\]

Monte Carlo sample means are not substituted. This keeps point forecasts
independent of `random_state` and simulation count.

## Positive-semidefinite terminal covariance

The finite terminal covariance can be singular because some state coordinates
are deterministically related. pySTARMAx symmetrizes the covariance and uses an
eigendecomposition:

\[
P_{\ast,T\mid T}=V\Lambda V^\top,
\]

\[
L_T=V\operatorname{diag}(\sqrt{\max(\lambda_i,0)}).
\]

Floating-point-scale negative eigenvalues are clipped to zero. Materially
negative eigenvalues raise. No diagonal jitter is added.

## Missing observations

Partial-location and fully missing rows are handled by the exact diffuse filter.
A fully missing row performs prediction only and does not reduce diffuse rank.

Forecast intervals remain available after trailing missing observations when
later filtering has already resolved every diffuse direction. The terminal
finite posterior then reflects prediction through those missing rows.

Intervals are unavailable when the sample never identifies all initial diffuse
directions. This can occur with:

- very short series;
- only missing observations in an integrated location;
- rank-deficient observation designs;
- state directions never connected to an observed location.

The caller should add informative observations, revise the model, or report
that a proper terminal forecast distribution is unavailable.

## Reproducibility and simulation size

`random_state` accepts an integer, a NumPy `Generator`, or `None`. Reusing the
same integer seed with identical fitted state and arguments returns identical
paths and bounds.

The default uses 2,000 paths. Tail quantiles have Monte Carlo error, so research
reporting will often benefit from at least 5,000 paths and a documented seed.

## Relationship to the exact diffuse likelihood

Forecast paths use fitted parameters from the original-level exact diffuse
likelihood

\[
L_D(Y_{1:T}).
\]

They do not propagate the observed-information covariance added in version
0.0.23. Thus the forecast law is

\[
p(Y_{T+1:T+H}\mid Y_{1:T},\widehat\theta),
\]

not the parameter-integrated distribution

\[
\int p(Y_{T+1:T+H}\mid Y_{1:T},\theta)
\,p(\theta\mid Y_{1:T})\,d\theta.
\]

Parameter-aware paths, parametric bootstrap refitting, or asymptotic parameter
draws are future extensions.

## Validation references

The test suite verifies:

- an analytic scalar state process with forecast means `1.5` and `2.0` and
  variances `5.0` and `6.0`;
- exact equality with ordinary stationary Kalman paths when the diffuse rank is
  zero and the two filters have identical terminal moments;
- random-walk forecast variance increasing linearly with horizon;
- original-level interval means matching `predict()`;
- highest-difference interval means matching `predict_differenced()`;
- deterministic reproducibility under a fixed seed;
- widening original-level random-walk intervals;
- second-order integrated projection;
- rejection of unresolved terminal diffuse rank;
- validation of steps, level, simulation count, and result type;
- immutable public interval arrays.

## Deliberate limitations

Version 0.0.24 does not provide:

- fitted-parameter uncertainty in forecast paths;
- exact diffuse seasonal augmentation;
- simulation smoothing conditional on the complete observed sample;
- forecast cross-time covariance as a public analytic tensor;
- measurement-noise draws from a separate observation disturbance;
- continuation filtering from an externally supplied terminal exact diffuse
  posterior;
- sparse state transitions.
