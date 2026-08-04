# Step 20 handoff: exact diffuse ordinary STARIMA maximum likelihood

## Repository position

- development version: `0.0.20`;
- branch: `agent/exact-diffuse-mle`;
- pull request: PR #20, `Add exact diffuse STARIMA maximum likelihood`;
- base: version 0.0.19 on `main` at merge commit
  `405b9886f446330e5adf479e03ccb595b95740eb`;
- core validation: GitHub Actions CI #433, run ID `30883369794`;
- core result: 176 tests passed and 86.99% total branch coverage;
- `src/pystarmax/exact_diffuse_mle.py` coverage: 84.3%;
- quality, mypy, strict MkDocs, distributions, coverage, Ubuntu, and macOS were
  green when the core result was recorded;
- a complete final documentation-head CI is required before merge.

## Delivered API

```python
model = ExactDiffuseKalmanSTARIMA(
    ar_order=p,
    integration_order=d,
    ma_order=q,
    covariance_type="full",
)
result = model.fit(level_observations, weights)
```

Public immutable result:

- `ExactDiffuseKalmanSTARIMAResult`.

Fitted methods:

- `filter(data=None)`;
- `to_state_space()`;
- `to_transformed_state_space()`;
- `admissibility()`;
- `predict()`;
- `predict_differenced()`.

## Likelihood distinction

The existing `KalmanSTARIMA` continues to maximize

\[
L_c
=
L\left(
\Delta^d y_{d+1:T}
\mid y_{1:d}
\right),
\]

on differenced observations conditional on removed level history.

The new estimator maximizes

\[
L_D=L_D(y_{1:T})
\]

on original levels through an integrated state with exact diffuse covariance.
The estimators, likelihoods, AIC values, and BIC values remain separate named
contracts.

## Candidate reconstruction

For every raw optimizer candidate:

1. split optional intercept, transformed AR factors, transformed MA factors, and
   covariance coordinates;
2. decode scalar, diagonal, or full-Cholesky innovation covariance;
3. construct the stationary transformed STARMA state space;
4. compute transformed AR and positive-sign inverse-MA spectral radii;
5. reject enabled inadmissible candidates with explicit penalties;
6. augment original level and lower ordinary-difference states;
7. assign `P_inf` only to the `d * N` integration directions;
8. initialize the transformed subsystem from its stationary mean and
   covariance;
9. run exact diffuse filtering on original observations;
10. return the negative exact diffuse log likelihood.

The objective never evaluates a conditional differenced likelihood while
reporting an exact diffuse result.

## Starts and optimizer

- starts use the existing stationary `KalmanSTARMA` machinery on the differenced
  observations;
- user-supplied dynamic and covariance starts remain available;
- stationarity and invertibility start shrinkage are inherited;
- L-BFGS-B uses the existing covariance-coordinate bounds;
- every start and candidate is finally evaluated on the original-level exact
  diffuse state space;
- final candidates receive hard admissibility checks after optimization.

## Parameter counting

For `K` spatial weights,

\[
k=\mathbf 1_c+K(p+q)+k_Q.
\]

Integration order changes latent-state dimension but adds no optimized
coefficient. Information criteria are

\[
AIC=-2\ell_D+2k,
\]

\[
BIC=-2\ell_D+k\log n_{obs},
\]

where `n_obs` counts finite original-level cells.

## Result contract

The immutable result stores:

- dynamic and raw optimizer coordinates and names;
- intercept, transformed AR factors, transformed MA factors, and natural
  innovation covariance;
- covariance type and `(p,d,q)` order;
- exact diffuse likelihood, AIC, and BIC;
- finite and diffuse observation counts;
- optimizer convergence, iterations, evaluations, method, and message;
- transformed AR and inverse-MA radii, limits, and enforcement flags;
- complete exact diffuse filter result;
- stationary transformed state space;
- exact integrated state-space specification.

Convenience properties provide coefficient and covariance tables, stationarity,
invertibility, joint admissibility, and a summary explicitly labelled as an
original-level exact diffuse likelihood.

## Missing data

- original observations may contain `NaN` cells;
- fully missing rows perform prediction only;
- partially observed rows use sequential location updates;
- missing initial levels delay diffuse completion;
- missing values are not imputed;
- differenced observations are used only to create starts and may have a wider
  missing stencil than the exact objective data.

## Fitted filtering

`filter()` returns the retained training result. `filter(new_data)` starts a new
exact diffuse initialization under the fitted parameters and does not continue
the training terminal posterior.

`to_state_space()` returns the augmented original-level state model.
`to_transformed_state_space()` returns the stationary STARMA model for
`Delta^d y_t`.

## Forecasts

- `predict()` recursively propagates the final augmented filtered state and
  returns original-level means;
- `predict_differenced()` propagates only the transformed terminal-state block
  and returns highest-difference means;
- both condition on fitted parameters;
- forecast intervals are not included in this stage.

## Validation references

Tests include:

1. random-walk drift and innovation variance against closed-form increment MLEs;
2. second-order integrated drift and variance against closed-form second-
   difference MLEs;
3. `d=0` raw optimizer coordinates and likelihood against stationary
   `KalmanSTARMA` under identical starts;
4. missing initial levels and delayed diffuse completion;
5. retained training filter and new-data filtering;
6. original and transformed state-space identity contracts;
7. original-level and highest-difference forecast recursions;
8. admissibility diagnostics;
9. immutable result arrays and summary labels;
10. constructor, fitted-state, data-length, and start validation;
11. the complete inherited package test suite.

Core CI #433 reported:

- 176 tests passed in 42.17 seconds in the coverage job;
- total branch coverage: 86.99%;
- exact diffuse MLE module coverage: 84.3%;
- Black, isort, Ruff, mypy, strict MkDocs, distribution checks, and diagnostic
  fixture regeneration passed;
- final cross-platform completion is confirmed by the documentation-head CI.

## Files introduced or changed

Core:

- `src/pystarmax/exact_diffuse_mle.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_exact_diffuse_mle.py`.

Documentation and example:

- `docs/exact_diffuse_mle.md`;
- `examples/exact_diffuse_mle.py`;
- README, documentation home, navigation, exact diffuse guide, conditional
  integrated guide, roadmap, project status, and this Step 20 handoff.

## Current limitations

- no exact diffuse observed-information inference;
- no exact diffuse smoothing or disturbance smoothing;
- no ordinary-seasonal diffuse augmentation;
- no forecast intervals from this estimator;
- no separate observation-noise covariance;
- sequential location order can affect floating-point rounding;
- diffuse rank decisions use a numerical tolerance;
- dense state and covariance matrices can be expensive;
- starts can be weak for highly incomplete differenced data;
- exogenous regressors and interventions are unsupported.

## Next recommended stage

The next stage should add exact diffuse fixed-interval smoothing for the
augmented ordinary-integrated state, preserving finite and diffuse-phase
semantics. An alternative inference stage is observed-information curvature for
this exact diffuse objective.
