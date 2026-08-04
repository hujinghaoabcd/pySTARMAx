# Remaining work inventory

## Snapshot

This inventory is based on development version 0.0.30 and PR #30.

Step 30 completes fixed-parameter seasonal exact-diffuse forecast paths and
central simulation intervals on original and transformed scales. After this
merge, the plan contains:

- **6 major technical workstreams**;
- **6 numbered core milestones**;
- approximately **11–15 independently reviewable projects** after ecosystem
  work is split.

This is a delivery inventory, not a claim that every research extension already
has a production-ready derivation.

## Workstream 1: remaining seasonal posterior operation

Status: next implementation candidate.

Add seasonal exact-diffuse conditional simulation smoothing after the 0.0.30
path and interval contracts are stable.

The implementation should:

- reuse the 0.0.26 augmented state without creating another seasonal layout;
- condition complete latent paths on original observations and missing masks;
- retain deterministic observation consistency where posterior variance is zero;
- reject unresolved terminal or smoothing diffuse directions when a proper draw
  cannot be defined;
- reduce exactly to the ordinary exact-diffuse simulation smoother when seasonal
  orders are zero;
- keep dense reference behavior explicit before introducing chunking or sparse
  execution.

## Workstream 2: diffuse cross-time covariance theory

Status: method research required.

15. Derive and independently validate the diffuse `L2` recursion required for
    lag-one smoothed state covariance.
16. Use that recursion to expose cross-time state-disturbance and primitive
    innovation covariance.

This workstream must not substitute a finite large covariance while continuing
to call the method exact diffuse.

## Workstream 3: stronger likelihood and parameter uncertainty

Status: not started.

17. Add robust/sandwich and profile-likelihood inference with explicit boundary,
    rank-deficiency, and singular-curvature policies.
18. Propagate parameter uncertainty into forecast paths, intervals, and selected
    smoother summaries through validated asymptotic draws or bootstrap refits.

Analytic derivatives may be added only with independent checks against the
existing finite-difference objectives.

## Workstream 4: scalable numerical execution

Status: not started.

19. Add sparse spatial and state operators plus memory-aware filtering,
    smoothing, forecasting, and simulation paths.
20. Add chunked or parallel execution with deterministic seed partitioning and
    dense-reference equivalence.

For forecasting, chunked empirical quantiles must define their approximation or
storage contract explicitly. The current dense path arrays remain transparent
moderate-sample references rather than large-data performance claims.

## Workstream 5: model specification automation

Status: split into independent future pull requests.

- smooth admissibility parameterization;
- automatic ordinary and seasonal order selection;
- automatic spatial-lag and weight-set selection;
- model-selection reporting that never mixes conditional and exact-diffuse
  likelihood conventions.

## Workstream 6: exogenous inputs and ecosystem integration

Status: split into independent future pull requests.

- exogenous regressors and intervention variables;
- GIS-oriented spatial-weight and data adapters;
- cross-language numerical reference fixtures;
- public PyPI and signed release automation;
- explicitly time-varying and other extended state-space models.

## Recommended sequence

1. Merge PR #30 after final synchronized CI.
2. Add seasonal exact-diffuse conditional simulation smoothing.
3. Run the diffuse `L2` derivation as a separate theory-and-validation project.
4. Add robust, profile, and parameter-aware uncertainty after posterior
   contracts are stable.
5. Introduce sparse and chunked execution before broad GIS adapters or automatic
   model search.
6. Complete ecosystem and release engineering after public numerical contracts
   stabilize.

## Completion definition

A milestone is complete only when it has:

- an explicit mathematical and scale contract;
- immutable typed public results where applicable;
- analytic or independently constructed numerical references;
- missing-data and rank-deficiency behavior;
- cross-platform tests;
- formatting, linting, typing, strict documentation, and package-build success;
- synchronized README, documentation home, roadmap, project status, method
  guide, example, and handoff;
- no temporary workflows, generated artifacts, or unresolved review threads.
