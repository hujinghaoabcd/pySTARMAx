# Remaining work inventory

## Snapshot

This inventory is based on development version 0.0.31 and PR #31.

Step 31 completes the planned seasonal exact-diffuse posterior operations by
adding complete conditional simulation smoothing. After this merge, the plan
contains:

- **5 major technical workstreams**;
- **6 numbered core milestones**;
- approximately **10–14 independently reviewable projects** after ecosystem
  work is split.

This is a delivery inventory, not a claim that every research extension already
has a production-ready derivation.

## Completed seasonal posterior contract

The seasonal exact-diffuse workflow now includes:

- fixed-interval state and observation smoothing;
- primitive innovation and state-disturbance smoothing;
- observed-information and natural covariance inference;
- original-level and transformed forecast paths and intervals;
- complete conditional simulation smoothing;
- transformed-state projections from the same conditional draws;
- explicit rejection of unresolved diffuse directions;
- exact reduction to ordinary posterior operations when seasonal integration is
  zero.

Future work must reuse this state layout and must not introduce a competing
seasonal exact-diffuse convention.

## Workstream 1: diffuse cross-time covariance theory

Status: next theory-and-validation project.

15. Derive and independently validate the diffuse `L2` recursion required for
    lag-one smoothed state covariance.
16. Use that recursion to expose cross-time state-disturbance and primitive
    innovation covariance.

This workstream must not substitute a finite large covariance while continuing
to call the method exact diffuse. The derivation requires analytic recursions,
small closed-form bridge references, ordinary finite-phase reduction, missing
observations, and numerical comparison against conditional source simulations
where identifiable.

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
storage contract explicitly. For simulation smoothing, sparse conditioning must
preserve observed-cell constraints and posterior marginal checks. Current dense
implementations remain transparent moderate-sample references.

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

1. Merge PR #31 after final synchronized CI.
2. Run the diffuse `L2` lag-one covariance derivation as an isolated
   theory-and-validation project.
3. Add cross-time disturbance covariance only after the `L2` recursion is
   independently verified.
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
