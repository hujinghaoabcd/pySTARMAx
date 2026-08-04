# Step 25 handoff: exact diffuse simulation smoothing

## Repository position

- development version: `0.0.25`;
- branch: `agent/exact-diffuse-simulation-smoothing`;
- pull request: PR #25, `Add exact diffuse simulation smoothing`;
- base: version 0.0.24 on `main` at merge commit
  `0639f4bd932aa86020215b50b6ef9867bb5ad566`;
- README, roadmap, root project status, exact diffuse MLE cross-links,
  documentation home, and MkDocs navigation are synchronized through 0.0.25;
- the temporary formatting and status-patching workflows have self-deleted and
  are absent from the final PR diff;
- final validation: pending on the ordinary post-cleanup merge-gate CI head.

## Delivered API

```python
result = exact_diffuse_simulation_smoother(
    exact_filter_result,
    n_simulations=2000,
    random_state=2026,
)
```

```python
result = fitted_model.simulate_smoothing_paths(
    n_simulations=2000,
    random_state=2026,
)
```

The fitted facade accepts retained training data or a newly initialized data
segment.

## Mathematical construction

The first predicted state is decomposed into explicit flat diffuse coordinates
and finite standard-normal coordinates. Later process innovations add further
proper standard-normal coordinates. The observed sample becomes one linear
constraint system.

All identified diffuse coordinates are eliminated analytically. The remaining
proper Gaussian source vector is conditioned in the left null space of the
diffuse observation design. Complete state paths are then reconstructed from
conditional source draws.

The implementation never uses an arbitrary finite diffuse scale and does not
require the missing diffuse lag-one covariance recursion.

## Deterministic verification

The source-coordinate construction independently computes posterior marginal
state means and covariances. These are checked against the existing exact
diffuse information smoother before any result is returned.

The validation suite includes:

1. a closed-form random-walk bridge;
2. deterministic fully observed random-walk paths;
3. zero-diffuse stationary incomplete data;
4. partial-location observations;
5. fitted training and new-data routes;
6. fixed-seed reproducibility;
7. immutable result arrays;
8. unresolved diffuse-rank refusal;
9. argument validation;
10. the complete inherited package suite.

## First CI findings

Initial CI #521, run ID `30943099910`, found three test-contract issues and one
formatting issue:

- two expected arrays relied on implicit broadcasting inside
  `numpy.testing.assert_allclose`, which is not portable across the supported
  NumPy/platform matrix;
- a fully observed deterministic path asserted that the remaining proper source
  coordinate dimension must be zero, although a null source direction can be
  exactly cancelled by the analytically solved diffuse coordinate while the
  state posterior remains deterministic;
- Black required formatting of the new core and test files.

The tests were changed to broadcast expected arrays explicitly and to assert the
actual inferential contract: deterministic state paths and zero posterior state
covariance. The analytic mean/covariance comparisons had already passed at
floating-point precision, so these findings did not indicate a defect in the
simulation posterior.

## Deliberate boundaries

Step 25 does not claim:

- primitive innovation or state-disturbance draws;
- exact diffuse lag-one state autocovariance;
- cross-time disturbance covariance;
- seasonal exact diffuse state augmentation;
- parameter-aware simulation paths;
- sparse or parallel execution;
- separate observation-noise simulation.

## Remaining roadmap after Step 25

The major remaining workstreams are:

1. seasonal ordinary-seasonal exact diffuse state augmentation;
2. seasonal exact diffuse smoothing, inference, and forecasting integration;
3. the diffuse `L2` recursion for lag-one state covariance and cross-time
   disturbance covariance;
4. robust, profile-likelihood, analytic-derivative, and parameter-uncertainty
   inference;
5. parameter-aware paths, sparse operators, and scalable execution;
6. smooth admissibility parameterization and automatic order selection;
7. exogenous/intervention inputs, GIS adapters, cross-language fixtures, PyPI
   release, parallel execution, and time-varying extensions.

These represent seven major technical workstreams, or roughly 17--21 separately
reviewable subprojects depending on how seasonal and release engineering are
split.

## Merge checklist

Before marking PR #25 ready:

1. pass complete CI on the final code and documentation head;
2. record run number, run ID, test count, total branch coverage, and new-module
   coverage;
3. confirm README, documentation home, roadmap, project status, exact diffuse
   MLE cross-links, and MkDocs navigation remain synchronized;
4. confirm no temporary workflow or generated artifact remains;
5. confirm no unresolved review thread remains;
6. mark ready and squash-merge;
7. start the seasonal exact diffuse design stage from the resulting `main`.
