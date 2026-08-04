# Exact diffuse Kalman filtering

Version 0.0.19 introduces a separate exact diffuse filtering kernel and an
ordinary integrated level-state constructor. It does **not** change the existing
`initialization="diffuse"` option, which remains an approximate large-variance
initialization for the ordinary Kalman filter.

## Why a separate exact method is needed

A nonstationary state component does not have a finite stationary initial
variance. Replacing that uncertainty with an arbitrary large number can be
useful numerically, but the resulting likelihood depends on the chosen scale
unless an asymptotic adjustment is made.

Exact diffuse initialization represents the initial covariance as

\[
P_1(\kappa)=P_{\ast,1}+\kappa P_{\infty,1},
\qquad \kappa\rightarrow\infty.
\]

Here:

- \(P_{\ast,1}\) is the finite covariance component;
- \(P_{\infty,1}\) identifies diffuse directions;
- the rank of \(P_{\infty,1}\) is the number of unresolved diffuse state
  directions.

The filter propagates both covariance components until the diffuse rank reaches
zero, after which ordinary Gaussian Kalman updates continue.

## Public low-level API

```python
from pystarmax import exact_diffuse_filter

result = exact_diffuse_filter(
    observations,
    state_space,
    initial_state=initial_state,
    initial_covariance=P_star,
    initial_diffuse_covariance=P_inf,
)
```

The scalar likelihood helper is:

```python
log_likelihood = exact_diffuse_loglikelihood(
    observations,
    state_space,
    initial_state=initial_state,
    initial_covariance=P_star,
    initial_diffuse_covariance=P_inf,
)
```

If no covariance components are supplied, the finite component defaults to zero
and the diffuse component defaults to the identity over the complete state.
This default is appropriate only when every state direction is intentionally
unknown with diffuse variance. Integrated builders below construct a more
specific decomposition automatically.

## Sequential observation updates

Observed locations are processed one at a time. For a scalar observation row
\(z\), innovation \(v\), finite covariance \(P_\ast\), and diffuse covariance
\(P_\infty\), define

\[
F_\infty=zP_\infty z^\top,
\qquad
F_\ast=zP_\ast z^\top,
\]

\[
M_\infty=P_\infty z^\top,
\qquad
M_\ast=P_\ast z^\top.
\]

When \(F_\infty>0\), the exact diffuse gains are

\[
K_0=\frac{M_\infty}{F_\infty},
\]

\[
K_1=\frac{M_\ast}{F_\infty}
-K_0\frac{F_\ast}{F_\infty}.
\]

The state and covariance components update as

\[
a^+=a+K_0v,
\]

\[
P_\ast^+
=P_\ast-M_\ast K_0^\top-M_\infty K_1^\top,
\]

\[
P_\infty^+
=P_\infty-M_\infty K_0^\top.
\]

The corresponding diffuse likelihood contribution is

\[
\ell_t^{(D)}
=-\frac12\log(2\pi F_\infty).
\]

The innovation does not contribute a quadratic term while that observation is
used to resolve an unknown diffuse direction.

When \(F_\infty\) is numerically zero and \(F_\ast>0\), the ordinary scalar
Gaussian update is used:

\[
K=\frac{M_\ast}{F_\ast},
\qquad
a^+=a+Kv,
\]

\[
P_\ast^+=P_\ast-M_\ast K^\top,
\]

\[
\ell_t^{(G)}
=-\frac12\left[
\log(2\pi)+\log F_\ast+\frac{v^2}{F_\ast}
\right].
\]

## Missing observations

`NaN` represents a missing cell.

- a missing location performs no scalar measurement update;
- a fully missing row performs only the state prediction;
- diffuse rank is not artificially reduced by missing observations;
- finite observed locations in a partially missing row are processed
  sequentially;
- the likelihood contribution of a fully missing row is zero.

Sequential updates make rank-deficient diffuse covariance and partial-location
observations explicit without constructing a singular multivariate innovation
inverse.

## Deterministic measurements

If both \(F_\infty\) and \(F_\ast\) are numerically zero, the measurement is
deterministic under the current state.

- a matching observation contributes zero and leaves the state unchanged;
- a materially conflicting observation raises `LinAlgError`.

The filter never hides a deterministic contradiction by injecting arbitrary
jitter.

## Result object

`ExactDiffuseFilterResult` stores immutable arrays for:

- predicted and filtered states;
- predicted and filtered finite covariance \(P_\ast\);
- predicted and filtered diffuse covariance \(P_\infty\);
- scalar innovations;
- finite and diffuse scalar innovation variances;
- observed and diffuse-update masks;
- per-time likelihood contributions;
- predicted and filtered diffuse-rank paths.

It also reports:

- total exact diffuse log likelihood;
- finite observation count;
- diffuse observation count;
- initial and final diffuse rank;
- `diffuse_end_time`, the first time index whose filtered diffuse rank is zero;
- predicted and filtered values on the observation scale.

## Ordinary integrated level-state construction

Given a stationary transformed model

\[
\beta_t=c+T\beta_{t-1}+R\eta_t,
\qquad
x_t=Z\beta_t,
\]

and

\[
x_t=\Delta^d y_t,
\]

`build_exact_integrated_state_space()` constructs the augmented state

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

Every integration block obeys

\[
\Delta^r y_t
=
\sum_{k=r}^{d-1}\Delta^k y_{t-1}+x_t,
\qquad r=0,\ldots,d-1.
\]

Because

\[
x_t=Zc+ZT\beta_{t-1}+ZR\eta_t,
\]

the ordered transition, intercept, and innovation-selection effects are inserted
directly into every integration block.

The initial decomposition is:

- identity diffuse covariance over the \(dN\) integration directions;
- zero finite covariance on those integration directions;
- the stationary mean and covariance of the transformed state \(\beta_t\);
- zero initial finite cross covariance between integration and transformed
  blocks.

The transformed transition must be stationary. A nonstationary transformed
model is rejected because its nonstationarity must be represented explicitly in
the integration chain rather than hidden inside the finite component.

## Integrated API

```python
specification = build_exact_integrated_state_space(
    transformed_state_space,
    integration_order=1,
)

result = specification.filter(level_observations)
print(result.log_likelihood)
print(result.filtered_diffuse_rank)
```

Convenience functions are:

```python
result = exact_integrated_filter(
    level_observations,
    transformed_state_space,
    integration_order=1,
)
```

```python
log_likelihood = exact_integrated_loglikelihood(
    level_observations,
    transformed_state_space,
    integration_order=1,
)
```

For `integration_order=0`, the specification uses the stationary mean and
covariance of the transformed state, sets \(P_\infty=0\), and is numerically
equivalent to ordinary stationary initialization in the scalar observation
case.

## Random-walk interpretation

For a scalar random walk with drift,

\[
y_t=y_{t-1}+\mu+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,\sigma^2),
\]

the first observed level resolves the single diffuse level direction and
contributes only the diffuse normalization. Remaining observations contribute
the ordinary Gaussian likelihood of increments

\[
\Delta y_t-\mu.
\]

For second-order integration, the first two observations resolve level and
slope directions. Remaining observations contribute the Gaussian likelihood of
second differences.

## Exact versus approximate diffuse initialization

Existing code such as

```python
kalman_filter(data, model, initialization="diffuse", diffuse_scale=1e6)
```

uses a finite covariance \(10^6I\). That route remains useful for a broad
stationary-state API and preserves backward compatibility, but it is not renamed
or reinterpreted as exact.

The exact route:

- stores \(P_\ast\) and \(P_\infty\) separately;
- has no arbitrary diffuse scale;
- exposes the diffuse rank and end time;
- uses the exact diffuse likelihood terms;
- supports integrated level-state construction.

Version 0.0.20 adds `ExactDiffuseKalmanSTARIMA`, which rebuilds the
transformed STARMA state, integrated level state, and exact diffuse likelihood
at every optimizer candidate. Version 0.0.21 adds exact diffuse fixed-interval
marginal state smoothing through the fitted estimator and low-level
`exact_diffuse_smoother()`. Version 0.0.22 adds primitive innovation and
state-equation disturbance marginal smoothing through
`exact_diffuse_disturbance_smoother()` and the fitted estimator facade.
Version 0.0.23 adds observed-information and natural innovation covariance
inference by rebuilding the complete original-level exact diffuse objective at
every curvature candidate. See
[Exact diffuse STARIMA maximum likelihood](exact_diffuse_mle.md),
[Exact diffuse likelihood inference](exact_diffuse_inference.md),
[Exact diffuse smoothing](exact_diffuse_smoothing.md), and
[Exact diffuse disturbance smoothing](exact_diffuse_disturbance_smoothing.md).
The conditional `KalmanSTARIMA` estimator remains unchanged.

## Validation references

Tests cover:

- a local-level model whose exact likelihood equals a diffuse first-level term
  plus the Gaussian increment likelihood;
- a local-linear-trend model compared with the adjusted large-variance limit;
- missing observations that delay diffuse-rank reduction;
- partially observed multi-location rows with sequential rank reduction;
- deterministic zero-variance agreement and contradiction;
- custom rank-deficient diffuse covariance;
- immutable output arrays;
- first- and second-order integrated random-walk likelihoods;
- exact integrated transition, intercept, selection, and design matrices;
- `d=0` equivalence with stationary initialization;
- rejection of a nonstationary transformed finite-state model.

## Scope and limitations

- exact diffuse filtering, fixed-parameter likelihoods, and optimizer-facing
  ordinary STARIMA MLE are available;
- exact diffuse observed-information and natural innovation covariance
  inference are implemented, but analytic derivatives, robust covariance, and
  parameter-uncertainty propagation are not;
- exact diffuse marginal state and primitive innovation/state-disturbance
  smoothing are implemented, but lag-one state autocovariance, cross-time
  disturbance covariance, and simulation smoothing are not;
- seasonal diffuse state augmentation and smoothing are not implemented;
- the current observation equation has no separate measurement-noise matrix;
- observations are processed sequentially, so location order is part of the
  floating-point evaluation path even though equivalent Gaussian models should
  agree up to numerical error;
- rank decisions use a configurable numerical tolerance;
- dense covariance matrices remain unsuitable for very large state spaces.

## References

- Koopman, S. J. (1997). Exact initial Kalman filtering and smoothing for
  nonstationary time series models. *Journal of the American Statistical
  Association*, 92(440), 1630–1638.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press.
- Statsmodels univariate exact diffuse filtering implementation, used as an
  independent open-source formula cross-check.
