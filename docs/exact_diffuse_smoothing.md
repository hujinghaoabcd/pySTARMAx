# Exact diffuse fixed-interval smoothing

Version 0.0.21 adds fixed-interval state smoothing for models filtered with
exact diffuse initialization.

```python
from pystarmax import ExactDiffuseKalmanSTARIMA

model = ExactDiffuseKalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=0,
)
model.fit(level_observations, weights)
smoothed = model.smooth()
```

The low-level route accepts an `ExactDiffuseFilterResult`:

```python
from pystarmax import exact_diffuse_smoother

smoothed = exact_diffuse_smoother(exact_filter_result)
```

This is a separate smoother. It does not approximate diffuse covariance with a
large finite number and then call the ordinary RTS routine.

## Why ordinary RTS is insufficient during the diffuse phase

The exact diffuse filter represents predicted covariance as

\[
P_t(\kappa)=P_{\ast,t}+\kappa P_{\infty,t},
\qquad \kappa\rightarrow\infty.
\]

During the diffuse phase, neither the full covariance nor the ordinary RTS gain
has a finite direct limit. Exact smoothing instead propagates ordinary and
diffuse information quantities backward through every sequential scalar
measurement update.

The implementation retains five backward quantities:

- \(r_t\), the ordinary scaled smoothed-state estimator;
- \(r_{\infty,t}\), its diffuse counterpart;
- \(N_t\), ordinary scaled estimator covariance;
- \(N_{1,t}\), the ordinary/diffuse cross term;
- \(N_{2,t}\), the second diffuse covariance term.

These quantities are exposed as immutable diagnostic arrays.

## Forward scalar records

The exact filter processes observed locations sequentially. The smoother
reconstructs the same scalar covariance path from the retained predicted
finite/diffuse covariance and recorded scalar innovation variances.

For design row \(z\), define

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

When \(F_\infty>0\),

\[
K_0=M_\infty/F_\infty,
\]

\[
K_1=M_\ast/F_\infty
-K_0F_\ast/F_\infty,
\]

\[
L_0=I-K_0z,
\qquad
L_1=-K_1z.
\]

When diffuse variance is zero and finite variance is positive,

\[
K=M_\ast/F_\ast,
\qquad
L=I-Kz.
\]

`maximum_filter_reconstruction_error` reports the largest absolute difference
between the reconstructed scalar-update covariance and the finite/diffuse
filtered covariance retained by the forward filter. This guards against a
smoother using a different sequential convention from the filter.

## Diffuse backward recursion

For a diffuse scalar measurement with innovation \(v\), define

\[
f_1=F_\infty^{-1},
\qquad
f_2=-F_\ast F_\infty^{-2}.
\]

Let superscript `+` denote the information quantities after the current scalar
measurement in the backward pass. Then

\[
r_\infty
=zvf_1+L_0^\top r_\infty^+ + L_1^\top r^+,
\]

\[
r=L_0^\top r^+,
\]

\[
N_2
=zz^\top f_2
+L_0^\top N_2^+L_0
+L_0^\top N_1^+L_1
+L_1^\top N_1^{+\top}L_0
+L_1^\top N^+L_1,
\]

\[
N_1
=zz^\top f_1
+L_0^\top N_1^+L_0
+L_1^\top N^+L_0,
\]

\[
N=L_0^\top N^+L_0.
\]

For an ordinary finite-variance scalar measurement,

\[
r=zv/F_\ast+L^\top r^+,
\]

\[
N=zz^\top/F_\ast+L^\top N^+L,
\]

while the diffuse cross information is propagated through the same ordinary
measurement transformation.

After all scalar observations at time \(t\) are processed, the transition
propagation to the previous time is

\[
r_{t-1}=T^\top r_t,
\qquad
r_{\infty,t-1}=T^\top r_{\infty,t},
\]

\[
N_{t-1}=T^\top N_tT,
\]

\[
N_{1,t-1}=T^\top N_{1,t}T,
\qquad
N_{2,t-1}=T^\top N_{2,t}T.
\]

## Smoothed state mean

Using the predicted state \(a_t\), finite covariance \(P_{\ast,t}\), and
diffuse covariance \(P_{\infty,t}\), the fixed-interval mean is

\[
\hat\alpha_t
=a_t+P_{\ast,t}r_t+P_{\infty,t}r_{\infty,t}.
\]

Both ordinary and diffuse information therefore contribute during the diffuse
phase. After \(P_{\infty,t}=0\), the expression reduces to the ordinary
information smoother.

## Smoothed state covariance

The finite posterior covariance is

\[
V_t
=P_{\ast,t}
-P_{\ast,t}N_tP_{\ast,t}
-P_{\infty,t}N_{1,t}P_{\ast,t}
-P_{\ast,t}N_{1,t}^\top P_{\infty,t}
-P_{\infty,t}N_{2,t}P_{\infty,t}.
\]

All four ordinary/diffuse correction terms are required. Omitting either cross
term generally produces incorrect uncertainty before diffuse completion.

The result is symmetrized. Negative eigenvalues within a floating-point
tolerance are clipped to zero and the largest correction is reported as
`maximum_covariance_correction`. A materially indefinite result raises instead
of being silently repaired.

## Observation-scale moments

For observation design matrix \(Z\),

\[
\hat y_t=Z\hat\alpha_t,
\]

\[
\operatorname{Var}(y_t\mid Y_{1:T})
=ZV_tZ^\top.
\]

The result exposes both `smoothed_observations` and
`smoothed_observation_covariance`.

In the current STARMA state-space convention there is no separate measurement
noise. Consequently, a fully observed deterministic measurement may have zero
posterior observation variance.

## Missing observations

- a missing scalar cell creates no forward or backward measurement update;
- a fully missing row still receives information from later observations
  through transition propagation;
- leading missing levels remain diffuse during forward filtering but can receive
  finite smoothed posterior moments from later observed levels;
- partially observed rows preserve the same location order used by the exact
  filter;
- missing values are never imputed before filtering or smoothing.

## Random-walk bridge example

For

\[
y_t=y_{t-1}+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,q),
\]

with observed endpoints \(y_1=a\), \(y_3=b\), and missing \(y_2\),

\[
E(y_2\mid y_1=a,y_3=b)=\frac{a+b}{2},
\]

\[
\operatorname{Var}(y_2\mid y_1,y_3)=q/2.
\]

The first observed level resolves the diffuse level direction. The smoother
returns endpoint variance zero and the Gaussian bridge variance at the missing
middle point.

For two leading missing values followed by one observed level, the smoothed
means equal the observed level and the backward variances are proportional to
the number of unresolved random-walk transitions.

## Stationary zero-diffuse limit

When the initial diffuse covariance is zero, all diffuse information quantities
remain zero. The exact diffuse smoother is then numerically equivalent to the
ordinary stationary RTS smoother for state means and marginal state
covariances.

This equivalence is independently tested on incomplete stationary AR data.

## Large-variance reference

For a local linear trend with finite covariance \(P_\ast\) and diffuse
covariance \(P_\infty\), an ordinary Kalman smoother initialized with

\[
P_\ast+cP_\infty
\]

approaches the exact diffuse smoother as \(c\) grows. Tests compare exact
smoothed state means and marginal covariances against a numerically stable large
finite scale.

This reference is used only for validation. The production exact smoother does
not choose or expose a diffuse scale.

## Public result

`ExactDiffuseSmootherResult` contains immutable arrays for:

- `smoothed_state`;
- `smoothed_covariance`;
- `smoothed_observations`;
- `smoothed_observation_covariance`;
- `scaled_smoothed_estimator` for \(r\);
- `scaled_smoothed_diffuse_estimator` for \(r_\infty\);
- `scaled_smoothed_estimator_covariance` for \(N\);
- `scaled_smoothed_diffuse1_estimator_covariance` for \(N_1\);
- `scaled_smoothed_diffuse2_estimator_covariance` for \(N_2\).

It also reports:

- the source exact diffuse filter result;
- diffuse completion time and final diffuse rank;
- maximum covariance stabilization correction;
- maximum forward covariance reconstruction error;
- numerical tolerance.

## Fitted estimator API

```python
training = fitted_exact_model.smooth()
```

returns smoothing for the retained training filter result.

```python
new_result = fitted_exact_model.smooth(new_level_observations)
```

starts a new exact diffuse filtering and smoothing problem under the fitted
parameters. It does not continue from the terminal training posterior.

## Why lag-one covariance is not returned

Ordinary RTS smoothing can obtain lag-one state covariance from the retained
smoothing gains. During the exact diffuse phase, the corresponding autocovariance
recursion requires an additional higher-order transition term beyond the
`L0`/`L1` information retained by the current filter and smoother.

Version 0.0.21 therefore does **not** fabricate lag-one exact diffuse
state covariance. Version 0.0.22 adds primitive innovation and state-equation
disturbance **marginal** posterior moments directly from the backward
information quantities:

\[
E(\eta_{t+1}\mid Y)=QR^	op r_t,
\qquad
\operatorname{Var}(\eta_{t+1}\mid Y)=Q-QR^	op N_tRQ.
\]

This does not require lag-one state autocovariance. See
[Exact diffuse disturbance smoothing](exact_diffuse_disturbance_smoothing.md).
Lag-one state covariance, cross-time disturbance covariance, and simulation
smoothing still require a dedicated extension with the complete diffuse
autocovariance recursion.

## Parameter uncertainty

All smoothing moments condition on fitted or supplied model parameters. The
optimizer covariance is not propagated through state means or covariance.

## Validation references

Tests cover:

- a closed-form random-walk missing bridge;
- leading missing random-walk levels;
- stationary zero-diffuse equivalence with ordinary RTS smoothing;
- local-linear-trend agreement with a large-variance limit;
- partially observed multi-location rows;
- exact reproduction of observed cells under the no-measurement-noise model;
- positive-semidefinite posterior covariance;
- immutable result arrays;
- forward covariance reconstruction diagnostics;
- top-level `ExactDiffuseKalmanSTARIMA.smooth()` for training and new data;
- the complete inherited package suite.

## Current limitations

- no lag-one exact diffuse state covariance;
- primitive innovation and state-disturbance marginal moments are available,
  but cross-time disturbance covariance is not;
- no exact diffuse simulation smoother;
- no seasonal diffuse smoothing;
- no separate measurement-noise covariance;
- no parameter-uncertainty propagation;
- sequential location order can affect floating-point rounding;
- dense matrices remain expensive for large spatial systems.

## References

- Koopman, S. J. (1997). Exact initial Kalman filtering and smoothing for
  nonstationary time series models. *Journal of the American Statistical
  Association*, 92(440), 1630–1638.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods* (2nd ed.). Oxford University Press.
- Statsmodels exact diffuse univariate smoothing implementation, used as an
  independent formula cross-check.
