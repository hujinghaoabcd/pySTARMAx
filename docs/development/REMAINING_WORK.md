# Remaining work inventory

## Snapshot

This inventory is based on development version 0.0.32 and PR #32.

Step 32 adds an exact dense adjacent-time covariance oracle for genuine diffuse
phases. After this merge, the plan contains:

- **5 major technical workstreams**;
- **6 numbered core milestones**, with the recursive part of milestone 15 still
  outstanding;
- approximately **9–13 independently reviewable projects** after ecosystem work
  is split.

This is a delivery inventory, not a claim that every research extension already
has a production-ready derivation.

## Completed exact-diffuse posterior contract

The ordinary and seasonal exact-diffuse workflows now include:

- fixed-interval state and observation smoothing;
- primitive innovation and state-disturbance smoothing;
- observed-information and natural covariance inference;
- original-level and transformed forecast paths and intervals;
- complete conditional simulation smoothing;
- exact dense adjacent-time state and observation covariance;
- transformed-state projections from the same conditional draws;
- explicit rejection of unresolved diffuse directions;
- exact reduction to ordinary posterior operations when diffuse or seasonal
  integration components vanish.

The 0.0.32 adjacent covariance is a dense moderate-sample exact reference. Future
recursive or sparse implementations must agree with it and must reuse the same
state layout and orientation.

## Workstream 1: recursive diffuse cross-time covariance theory

Status: dense oracle complete; recursive theory remains.

15. Derive and independently validate the memory-linear diffuse `L2` recursion
    required for lag-one smoothed state covariance. The implementation must
    reproduce the 0.0.32 dense oracle across genuine diffuse, zero-diffuse,
    missing-data, non-symmetric-transition, ordinary, and seasonal cases.
16. Use the validated recursion to expose arbitrary cross-time state covariance,
    cross-time state-disturbance covariance, and primitive-innovation
    covariance.

This workstream must not substitute a finite large covariance while continuing
to call the method exact diffuse. It requires analytic recursions, closed-form
bridge references, ordinary finite-phase reduction, and independent comparison
against both dense source conditioning and conditional path simulation.

## Workstream 2: stronger likelihood and parameter uncertainty

Status: not started.

17. Add robust/sandwich and profile-likelihood inference with explicit boundary,
    rank-deficiency, and singular-curvature policies.
18. Propagate parameter uncertainty into forecast paths, intervals, and selected
    smoother summaries through validated asymptotic draws or bootstrap refits.

Analytic derivatives may be added only with independent checks against the
existing finite-difference objectives.

## Workstream 3: scalable numerical execution

Status: not started.

19. Add sparse spatial and state operators plus memory-aware filtering,
    smoothing, forecasting, and simulation paths.
20. Add chunked or parallel execution with deterministic seed partitioning and
    dense-reference equivalence.

For forecasting, chunked empirical quantiles must define their approximation or
storage contract explicitly. For simulation smoothing and adjacent covariance,
sparse methods must preserve observed-cell constraints, covariance orientation,
and agreement with dense references. Current dense implementations remain
transparent moderate-sample oracles.

## Workstream 4: model specification automation

Status: split into independent future pull requests.

- smooth admissibility parameterization;
- automatic ordinary and seasonal order selection;
- automatic spatial-lag and weight-set selection;
- model-selection reporting that never mixes conditional and exact-diffuse
  likelihood conventions.

## Workstream 5: exogenous inputs and ecosystem integration

Status: split into independent future pull requests.

- exogenous regressors and intervention variables;
- GIS-oriented spatial-weight and data adapters;
- cross-language numerical reference fixtures;
- public PyPI and signed release automation;
- explicitly time-varying and other extended state-space models.

## Recommended sequence

1. Merge PR #32 after final synchronized CI.
2. Derive the memory-linear diffuse `L2` recursion and compare it directly with
   the 0.0.32 dense adjacent covariance oracle.
3. Add arbitrary cross-time disturbance covariance only after recursive/dense
   equivalence is established.
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