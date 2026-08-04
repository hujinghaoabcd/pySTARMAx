# Step 19 handoff: exact diffuse filtering and integrated level states

## Repository position

- development version: `0.0.19`;
- branch: `agent/exact-diffuse-kalman`;
- pull request: PR #19, `Add exact diffuse Kalman filtering`;
- base: version 0.0.18 on `main` at merge commit
  `407ced8fec73fd63a4a5357da3b422f62b650e95`;
- authoritative implementation/documentation validation: GitHub Actions CI
  #422, run ID `30880321271`;
- validation result: 169 tests passed and 87.12% total branch coverage;
- `src/pystarmax/exact_diffuse.py` coverage: 87.8%;
- `src/pystarmax/exact_integrated.py` coverage: 86.9%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and Ubuntu/Windows/
  macOS Python 3.11–3.14 all passed;
- a validation-record-only merge-gate CI is required before merge.

## Delivered API

Generic exact diffuse filtering:

```python
result = exact_diffuse_filter(
    observations,
    state_space,
    initial_state=initial_state,
    initial_covariance=P_star,
    initial_diffuse_covariance=P_inf,
)
```

Scalar likelihood helper:

```python
log_likelihood = exact_diffuse_loglikelihood(...)
```

Ordinary integrated construction:

```python
specification = build_exact_integrated_state_space(
    transformed_state_space,
    integration_order=d,
)
result = specification.filter(level_observations)
```

Convenience functions:

- `exact_integrated_filter()`;
- `exact_integrated_loglikelihood()`.

Public immutable types:

- `ExactDiffuseFilterResult`;
- `ExactIntegratedStateSpace`.

## Separation from approximate diffuse initialization

The pre-existing call

```python
kalman_filter(data, model, initialization="diffuse", diffuse_scale=1e6)
```

continues to mean a finite large-variance approximation. Version 0.0.19 does not
rename or silently alter that behavior.

The exact route stores

\[
P_1(\kappa)=P_{\ast,1}+\kappa P_{\infty,1},
\qquad \kappa\rightarrow\infty,
\]

as two separate covariance components and applies exact diffuse likelihood
updates until the rank of `P_inf` reaches zero.

## Sequential exact diffuse recursion

For one observed scalar row `z`, define

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

When `F_inf` is positive,

\[
K_0=M_\infty/F_\infty,
\]

\[
K_1=M_\ast/F_\infty-K_0F_\ast/F_\infty.
\]

The state uses `K0`, while the finite covariance uses both `K0` and `K1`:

\[
a^+=a+K_0v,
\]

\[
P_\ast^+=P_\ast-M_\ast K_0^\top-M_\infty K_1^\top,
\]

\[
P_\infty^+=P_\infty-M_\infty K_0^\top.
\]

The exact diffuse likelihood term is

\[
-\tfrac12\log(2\pi F_\infty).
\]

There is no innovation-squared term for an observation used to identify a
diffuse direction. Once `F_inf` is zero, ordinary scalar Gaussian updates are
used.

## Missing and deterministic observations

- `NaN` cells are skipped;
- fully missing rows perform state prediction only;
- missing observations do not reduce diffuse rank;
- partially observed locations are processed sequentially;
- deterministic zero-variance agreement contributes zero;
- deterministic contradiction raises `LinAlgError` rather than injecting
  arbitrary jitter.

## Result contract

`ExactDiffuseFilterResult` stores:

- predicted and filtered states;
- predicted and filtered finite covariance;
- predicted and filtered diffuse covariance;
- scalar innovations;
- finite and diffuse innovation variances;
- observed and diffuse-update masks;
- per-time likelihood contributions;
- predicted and filtered diffuse-rank paths;
- total likelihood and observation counts;
- initial/final diffuse rank and `diffuse_end_time`;
- observation-scale predicted and filtered values.

All public numerical arrays are immutable.

## Ordinary integrated augmentation

For a stationary transformed state

\[
\beta_t=c+T\beta_{t-1}+R\eta_t,
\qquad x_t=Z\beta_t,
\]

and `x_t = Delta^d y_t`, the augmented state is

\[
[y_t,\Delta y_t,\ldots,\Delta^{d-1}y_t,\beta_t].
\]

Each integration block follows

\[
\Delta^r y_t
=\sum_{k=r}^{d-1}\Delta^k y_{t-1}+x_t.
\]

Every block therefore receives the ordered transformed effects `Z @ T`,
`Z @ c`, and `Z @ R`.

Initialization is:

- identity diffuse covariance over the `d * N` integration directions;
- stationary finite mean/covariance for the transformed state;
- zero finite integration covariance and initial cross covariance.

The transformed transition must be stationary. Nonstationarity must be placed in
the explicit integration chain rather than hidden inside the finite component.

## Validation references

Tests include:

1. local-level exact likelihood equal to a diffuse first-level term plus the
   Gaussian increment likelihood;
2. local-linear-trend comparison with the adjusted large-variance limit;
3. delayed diffuse-rank reduction under missing observations;
4. sequential partial-location rank reduction;
5. deterministic zero-variance agreement and contradiction;
6. custom rank-deficient diffuse covariance and immutable outputs;
7. first-order integrated likelihood equal to the drift-adjusted increment
   likelihood;
8. second-order integrated likelihood equal to the second-difference likelihood;
9. exact augmented transition/intercept/selection/design orientation;
10. `d=0` equality with stationary initialization;
11. rejection of a nonstationary transformed finite state;
12. the complete inherited package test suite.

Authoritative CI #422 reported:

- 169 tests passed in 61.57 seconds in the coverage job;
- total branch coverage: 87.12%;
- exact diffuse module coverage: 87.8%;
- exact integrated module coverage: 86.9%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and diagnostic fixture
  regeneration passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

The validation-record-only head changes only this handoff and
`PROJECT_STATUS.md`. It receives one final merge-gate CI before PR #19 is
marked ready and merged.

## Files introduced or changed

Core:

- `src/pystarmax/exact_diffuse.py`;
- `src/pystarmax/exact_integrated.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_exact_diffuse.py`;
- `tests/test_exact_integrated.py`.

Documentation and example:

- `docs/exact_diffuse.md`;
- `examples/exact_diffuse.py`;
- README, documentation home, navigation, roadmap, project status, and this
  Step 19 handoff.

## Current limitations

- no optimizer-facing exact diffuse `KalmanSTARIMA.fit()` yet;
- no exact diffuse smoother;
- no seasonal integration-state augmentation;
- no separate observation-noise covariance;
- sequential location order can change floating-point rounding;
- rank decisions depend on a numerical tolerance;
- dense covariance matrices remain unsuitable for very large state spaces.

## Next recommended stage

The next stage should add an optimizer-facing exact diffuse ordinary STARIMA
estimator that rebuilds the integrated level-state model at every parameter
candidate and preserves the distinction from conditional transformed
likelihood. Exact diffuse smoothing should remain a separate subsequent stage.
