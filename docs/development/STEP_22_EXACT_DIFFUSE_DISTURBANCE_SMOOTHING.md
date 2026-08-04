# Step 22 handoff: exact diffuse disturbance smoothing

## Repository position

- development version: `0.0.22`;
- branch: `agent/exact-diffuse-disturbance-smoothing`;
- pull request: PR #22, `Add exact diffuse disturbance smoothing`;
- base: version 0.0.21 on `main` at merge commit
  `b6e9b167a012191590aae2abef0218f25482d00b`;
- authoritative implementation/documentation validation: GitHub Actions CI
  #483, run ID `30892932606`;
- validation result: 193 tests passed and 87.19% total branch coverage;
- `src/pystarmax/exact_diffuse_disturbance_smoothing.py` coverage: 88.1%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and Ubuntu/Windows/
  macOS Python 3.11–3.14 all passed;
- the merge-gate candidate is frozen except for this validation-record note;
- a validation-record-only merge-gate CI is required before merge.

## Delivered API

Low-level disturbance smoothing:

```python
result = exact_diffuse_disturbance_smoother(exact_smoother_result)
```

Fitted estimator facade:

```python
result = fitted_exact_model.smooth_innovation_disturbances()
new_result = fitted_exact_model.smooth_innovation_disturbances(new_levels)
```

Public immutable result:

- `ExactDiffuseDisturbanceResult`.

## Model equation and index convention

For

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_{t+1},
\qquad
\eta_{t+1}\sim\mathcal N(0,Q),
\]

transition row `t` stores the posterior moments of the primitive innovation
entering state \(\alpha_{t+1}\). The state-equation disturbance is

\[
w_{t+1}=R\eta_{t+1}.
\]

A series with `n_time` observations returns `n_time - 1` disturbance rows.

## Exact information-smoother formulas

The exact diffuse state smoother retains ordinary backward information
quantities \(r_t\) and \(N_t\). The primitive innovation moments are

\[
E(\eta_{t+1}\mid Y)=QR^\top r_t,
\]

\[
\operatorname{Var}(\eta_{t+1}\mid Y)
=Q-QR^\top N_tRQ.
\]

State-equation disturbance moments follow by multiplication with \(R\):

\[
E(w_{t+1}\mid Y)=R E(\eta_{t+1}\mid Y),
\]

\[
\operatorname{Var}(w_{t+1}\mid Y)
=R\operatorname{Var}(\eta_{t+1}\mid Y)R^\top.
\]

The implementation uses `scaled_smoothed_estimator[1:]` and
`scaled_smoothed_estimator_covariance[1:]`, whose row `t + 1` corresponds to
information for the transition disturbance entering state `t + 1`.

## Why this does not require lag-one state covariance

Computing state disturbance covariance from

\[
\alpha_{t+1}-c-T\alpha_t
\]

would require exact diffuse lag-one state autocovariance. The diffuse
univariate autocovariance recursion requires a nontrivial higher-order `L2`
term beyond the retained `L0` and `L1` quantities. Statsmodels also leaves that
path unimplemented and documents this limitation.

Step 22 therefore uses the exact disturbance-smoother information equations and
does not apply an ordinary RTS lag-one formula during the diffuse phase.

## Selection-nullspace contract

The primitive innovation covariance is not recovered by inverting the selection
matrix. If a direction of \(\eta_t\) lies in the null space of \(R\), then

\[
QR^\top N_tRQ
\]

contains no correction in that direction, so its unresolved prior variance is
retained. This is required for non-injective process loading and avoids a
minimum-norm pseudo-innovation claim.

## Covariance policy

- innovation and state-disturbance covariance are symmetrized;
- only floating-point-scale negative eigenvalues are clipped;
- maximum innovation and state covariance corrections are returned;
- materially indefinite covariance raises `LinAlgError`;
- no arbitrary diffuse scale, inverse of `R`, or diagonal jitter is introduced.

## Missing-data behavior

- missing scalar cells perform no measurement update upstream;
- fully missing rows still receive future information;
- missing bridges produce conditional innovation means and nonzero marginal
  covariance;
- leading missing diffuse levels can leave early innovations at their prior
  mean and variance;
- new data passed to the fitted facade starts a new exact diffuse
  initialization rather than continuing the training terminal posterior.

## Validation references

Tests include:

1. exact fully observed random-walk increments;
2. a missing random-walk bridge with closed-form increment posterior moments;
3. leading missing levels with unresolved innovation variance;
4. zero-diffuse equivalence with ordinary stationary RTS innovation smoothing;
5. valid rank-deficient selection with retained primitive null-space variance;
6. one-observation empty transition arrays;
7. positive-semidefinite covariance and immutable arrays;
8. type and tolerance validation;
9. fitted-model smoothing for retained training data and newly initialized data;
10. the complete inherited package test suite.

Authoritative CI #483 reported:

- 193 tests passed in 74.40 seconds in the coverage job;
- total branch coverage: 87.19%;
- exact diffuse disturbance smoothing module coverage: 88.1%;
- exact diffuse state smoothing module coverage: 91.6%;
- exact diffuse MLE module coverage: 84.3%;
- exact diffuse filter coverage: 87.8%;
- exact integrated constructor coverage: 86.9%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and diagnostic fixture
  regeneration passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

The validation-record-only head changes only this handoff and
`PROJECT_STATUS.md`. It receives one final merge-gate CI before PR #22 is
marked ready and merged.

## Files introduced or changed

Core and metadata:

- `src/pystarmax/exact_diffuse_disturbance_smoothing.py`;
- `src/pystarmax/exact_diffuse_model.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_exact_diffuse_disturbance_smoothing.py`;
- `tests/test_exact_diffuse_disturbance_model.py`.

Documentation and example:

- `docs/exact_diffuse_disturbance_smoothing.md`;
- `examples/exact_diffuse_disturbance_smoothing.py`;
- README, documentation home, navigation, exact diffuse guides, roadmap,
  project status, and this Step 22 handoff.

## Deliberate omissions

Version 0.0.22 does not claim:

- exact diffuse lag-one state autocovariance;
- cross-time primitive innovation or state-disturbance covariance;
- exact diffuse simulation smoothing;
- measurement-disturbance smoothing with a separate observation-noise matrix;
- seasonal diffuse disturbance smoothing;
- parameter-uncertainty propagation through disturbance moments;
- exact diffuse observed-information inference.

## Next recommended stage

The most independent next extension is exact diffuse observed-information and
natural covariance inference, because it does not depend on the unresolved
`L2` autocovariance recursion. Simulation smoothing and seasonal diffuse state
augmentation remain separate later stages.
