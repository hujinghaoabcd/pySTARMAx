# Step 23 handoff: exact diffuse likelihood inference

## Repository position

- development version: `0.0.23`;
- branch: `agent/exact-diffuse-inference`;
- pull request: PR #23, `Add exact diffuse likelihood inference`;
- base: version 0.0.22 on `main` at merge commit
  `6a9f4cf9d9e19edd54123b22a892f36b7daf7b66`;
- core validation: GitHub Actions CI #497, run ID `30895807519`;
- core result: 197 tests passed and 87.27% total branch coverage;
- `src/pystarmax/exact_diffuse_inference.py` coverage: 88.8%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and Ubuntu/Windows/
  macOS Python 3.11–3.14 passed;
- a complete final documentation-head CI is required before merge.

## Delivered API

Fitted estimator inference:

```python
inference = fitted_exact_model.likelihood_inference()
```

Low-level entry point:

```python
inference = infer_exact_diffuse_kalman_starima(fitted_exact_model)
```

Natural innovation covariance delta inference:

```python
natural = inference.innovation_covariance_inference()
```

The public result remains the immutable `LikelihoodInferenceResult` already
used by stationary and seasonal likelihood inference.

## Objective contract

For every finite-difference candidate, Step 23 rebuilds:

1. optimizer-coordinate intercept, AR, and positive-sign MA coefficients;
2. scalar, diagonal, or full-Cholesky innovation covariance;
3. transformed stationary STARMA state space;
4. ordinary integrated original-level state space;
5. exact finite/diffuse initialization;
6. the complete exact diffuse likelihood on the original level observations.

It does not reuse conditional differenced likelihood values and does not replace
diffuse variance with an arbitrary finite scale.

## Curvature definition

The observed-information estimate is

\[
\mathcal I(\widehat\vartheta)
=
\left.
\frac{\partial^2[-\ell_D(\vartheta)]}
{\partial\vartheta\partial\vartheta^\top}
\right|_{\widehat\vartheta}.
\]

Central finite differences use

\[
h_j=\max(h_{\mathrm{abs}},
          h_{\mathrm{rel}}\max(1,|\widehat\vartheta_j|)).
\]

The existing curvature engine records gradient, Hessian, steps, fitted
objective value, and function-evaluation count. It reduces symmetric steps when
an initial candidate is invalid.

## Admissibility boundary

Candidate reconstruction uses the fitted estimator's:

- AR stationarity limit;
- positive-sign inverse-MA invertibility limit;
- smooth large invalid-region penalty.

The result records fitted stability and invertibility boundary distances.

## Hessian covariance policy

By default, the Hessian must be full rank and positive definite. Otherwise the
method raises `LinAlgError` with rank and minimum-eigenvalue diagnostics.

`allow_singular=True` explicitly enables a positive-eigenspace generalized
inverse. Such output is labelled with:

- `used_pseudoinverse=True`;
- `positive_definite=False`;
- the retained numerical rank.

This path is diagnostic and must not be presented as ordinary identified Wald
inference.

## Natural covariance delta method

The optimizer covariance coordinates are transformed with the package analytic
Jacobian. For natural covariance map \(g\),

\[
\operatorname{Var}[g(\widehat\vartheta)]
\approx
J_g\operatorname{Var}(\widehat\vartheta)J_g^\top.
\]

Scalar, diagonal, and full-Cholesky covariance parameterizations are supported.
Dynamic/natural-covariance cross uncertainty is retained.

## Analytic random-walk reference

For \(n\) scalar increments in

\[
y_t=y_{t-1}+\mu+\eta_t,
\qquad \eta_t\sim\mathcal N(0,\sigma^2),
\]

and optimizer coordinates \((\mu,\log\sigma)\),

\[
\mathcal I=
\operatorname{diag}(n/\widehat\sigma^2,2n).
\]

The raw covariance is

\[
\operatorname{diag}(\widehat\sigma^2/n,1/(2n)),
\]

and the natural innovation variance delta standard error is

\[
\operatorname{SE}(\widehat\sigma^2)
=
\widehat\sigma^2\sqrt{2/n}.
\]

The test suite verifies all three expressions and confirms that the curvature
objective equals the retained fitted exact diffuse objective.

## Missing-data behavior

Each curvature candidate preserves `NaN` observations. Missing locations,
fully missing rows, and delayed diffuse-rank resolution therefore follow the
same exact diffuse filtering path used during estimation. No imputation is
introduced for inference.

## Validation references

Tests include:

1. scalar random-walk closed-form Hessian;
2. closed-form optimizer covariance;
3. natural variance delta standard error;
4. exact fitted-objective identity;
5. missing-data finite inference and immutable arrays;
6. strict singular-Hessian rejection;
7. explicit positive-eigenspace generalized inverse diagnostics;
8. fit, finite-difference-step, and `rcond` validation;
9. the complete inherited package test suite.

Core CI #497 reported:

- 197 tests passed in 59.18 seconds in the coverage job;
- total branch coverage: 87.27%;
- exact diffuse inference module coverage: 88.8%;
- exact diffuse disturbance smoothing coverage: 88.1%;
- exact diffuse state smoothing coverage: 91.6%;
- exact diffuse MLE coverage: 84.3%;
- exact diffuse filter coverage: 87.8%;
- exact integrated constructor coverage: 86.9%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and diagnostic fixture
  regeneration passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

## Files introduced or changed

Core and metadata:

- `src/pystarmax/exact_diffuse_inference.py`;
- `src/pystarmax/exact_diffuse_model.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_exact_diffuse_inference.py`.

Documentation and example:

- `docs/exact_diffuse_inference.md`;
- `examples/exact_diffuse_inference.py`;
- README, documentation home, navigation, exact diffuse guides, roadmap,
  project status, and this Step 23 handoff.

## Deliberate omissions

Version 0.0.23 does not provide:

- analytic score or Hessian recursions;
- robust or sandwich exact diffuse covariance;
- parameter uncertainty propagated into forecasts, state smoothing, or
  disturbance smoothing;
- seasonal exact diffuse inference;
- profile-likelihood or bootstrap parameter intervals;
- sparse or parameter-aware derivative paths.

## Next recommended stage

The next independent user-facing extension should add forecast intervals to
`ExactDiffuseKalmanSTARIMA`, reusing the terminal exact diffuse posterior once
the diffuse rank is resolved and explicitly refusing unsupported unresolved
terminal diffuse states. Exact diffuse simulation smoothing remains a separate
larger stage.
