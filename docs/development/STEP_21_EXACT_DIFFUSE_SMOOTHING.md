# Step 21 handoff: exact diffuse fixed-interval smoothing

## Repository position

- development version: `0.0.21`;
- branch: `agent/exact-diffuse-smoothing`;
- pull request: PR #21, `Add exact diffuse fixed-interval smoothing`;
- base: version 0.0.20 on `main` at merge commit
  `36edc8b37f56e456aef1daa2df7aa496a07f13fb`;
- authoritative implementation/documentation validation: GitHub Actions CI
  #463, run ID `30887094839`;
- validation result: 184 tests passed and 87.16% total branch coverage;
- `src/pystarmax/exact_diffuse_smoothing.py` coverage: 91.6%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and Ubuntu/Windows/
  macOS Python 3.11–3.14 all passed;
- a validation-record-only merge-gate CI is required before merge.

## Delivered API

Low-level smoothing:

```python
result = exact_diffuse_smoother(exact_filter_result)
```

Fitted estimator facade:

```python
result = fitted_exact_model.smooth()
new_result = fitted_exact_model.smooth(new_level_observations)
```

Public immutable result:

- `ExactDiffuseSmootherResult`.

## Backward information quantities

The exact diffuse smoother retains:

- ordinary scaled estimator `r`;
- diffuse scaled estimator `r_inf`;
- ordinary estimator covariance `N`;
- ordinary/diffuse cross covariance `N1`;
- second diffuse covariance `N2`.

These are returned as immutable arrays for auditability and numerical
verification.

## Scalar diffuse recursion

For one observed scalar design row `z`,

\[
F_\infty=zP_\infty z^\top,
\qquad
F_\ast=zP_\ast z^\top,
\]

\[
K_0=P_\infty z^\top/F_\infty,
\]

\[
K_1=P_\ast z^\top/F_\infty-K_0F_\ast/F_\infty,
\]

\[
L_0=I-K_0z,
\qquad L_1=-K_1z.
\]

With `f1 = 1/F_inf` and `f2 = -F_*/F_inf**2`, the backward diffuse update is

\[
r_\infty=zvf_1+L_0^\top r_\infty^+ + L_1^\top r^+,
\]

\[
r=L_0^\top r^+,
\]

\[
N_2=zz^\top f_2+L_0^\top N_2^+L_0
+L_0^\top N_1^+L_1
+L_1^\top N_1^{+\top}L_0
+L_1^\top N^+L_1,
\]

\[
N_1=zz^\top f_1+L_0^\top N_1^+L_0+L_1^\top N^+L_0,
\]

\[
N=L_0^\top N^+L_0.
\]

Ordinary finite-variance scalar observations use the ordinary information
smoother and propagate the remaining diffuse cross information through the same
measurement transformation.

## State moments

The smoothed state mean is

\[
\hat\alpha_t
=a_t+P_{\ast,t}r_t+P_{\infty,t}r_{\infty,t}.
\]

The finite posterior covariance is

\[
V_t=P_{\ast,t}
-P_{\ast,t}N_tP_{\ast,t}
-P_{\infty,t}N_{1,t}P_{\ast,t}
-P_{\ast,t}N_{1,t}^\top P_{\infty,t}
-P_{\infty,t}N_{2,t}P_{\infty,t}.
\]

The result also returns observation-scale means and covariance through the
state-space design matrix.

## Forward reconstruction guard

Because filtering and smoothing process locations sequentially, the smoother
reconstructs every forward scalar finite/diffuse covariance update. It compares
the reconstructed final covariance at each time against the retained filter
output and reports `maximum_filter_reconstruction_error`.

This prevents accidental use of a different scalar update order or a different
finite/diffuse covariance rule in the backward pass.

## Covariance stabilization

- posterior covariance is symmetrized;
- floating-point-scale negative eigenvalues are clipped to zero;
- the largest correction is reported;
- materially indefinite covariance raises;
- no arbitrary large diffuse scale enters the production smoother.

## Missing data

- missing scalar cells create no measurement update;
- fully missing rows still receive future information through transition
  propagation;
- leading missing levels can receive finite posterior state moments from later
  observations;
- partially observed rows preserve the exact filter's location order;
- no missing value is imputed before filtering or smoothing.

## Fitted-model contract

`ExactDiffuseKalmanSTARIMA.smooth()` returns smoothing for the retained training
filter. Calling `smooth(new_data)` starts a new exact diffuse filtering and
smoothing problem under fitted parameters. It does not continue the terminal
training posterior.

## Validation references

Tests include:

1. closed-form random-walk bridge mean and variance;
2. leading missing random-walk levels;
3. `P_inf = 0` equality with ordinary stationary RTS smoothing;
4. local-linear-trend agreement with a stable large-variance limit;
5. partially observed multi-location rows;
6. observed-cell reconstruction under the no-measurement-noise convention;
7. positive-semidefinite posterior covariance and immutable arrays;
8. forward finite/diffuse covariance reconstruction diagnostics;
9. top-level fitted estimator smoothing for training and new data;
10. the complete inherited package test suite.

Authoritative CI #463 reported:

- 184 tests passed in 71.97 seconds in the coverage job;
- total branch coverage: 87.16%;
- exact diffuse smoothing module coverage: 91.6%;
- exact diffuse MLE module coverage: 84.3%;
- exact diffuse filter coverage: 87.8%;
- exact integrated constructor coverage: 86.9%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and diagnostic fixture
  regeneration passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

The validation-record-only head changes only this handoff and
`PROJECT_STATUS.md`. It receives one final merge-gate CI before PR #21 is
marked ready and merged.

## Deliberate omissions

Version 0.0.21 does not return lag-one exact diffuse state covariance. During the
diffuse phase the autocovariance recursion requires an additional higher-order
transition term beyond the retained `L0` and `L1` information.

Consequently this stage also does not claim:

- exact diffuse state-disturbance covariance;
- original innovation disturbance smoothing;
- exact diffuse simulation smoothing;
- seasonal diffuse smoothing.

These must be implemented through a dedicated complete diffuse
autocovariance/disturbance recursion.

## Files introduced or changed

Core:

- `src/pystarmax/exact_diffuse_smoothing.py`;
- `src/pystarmax/exact_diffuse_model.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_exact_diffuse_smoothing.py`;
- `tests/test_exact_diffuse_smoothing_model.py`.

Documentation and example:

- `docs/exact_diffuse_smoothing.md`;
- `examples/exact_diffuse_smoothing.py`;
- README, documentation home, navigation, exact diffuse guides, roadmap,
  project status, and this Step 21 handoff.

## Current limitations

- no exact diffuse lag-one state covariance;
- no exact diffuse state or innovation disturbance smoothing;
- no exact diffuse simulation smoother;
- no seasonal diffuse smoothing;
- no separate measurement-noise covariance;
- no parameter-uncertainty propagation;
- sequential location order can affect floating-point rounding;
- state and covariance matrices remain dense.

## Next recommended stage

The next statistically complete extension should implement the additional
diffuse autocovariance transition term required for lag-one state covariance and
state-disturbance smoothing. A separate inference track can add
observed-information curvature for the exact diffuse MLE objective.
