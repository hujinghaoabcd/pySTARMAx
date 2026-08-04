# Remaining work inventory

## Snapshot

This inventory is based on `main` version 0.0.28 at
`83656043b916395b95c7a1135625cb223e8a43ce` and PR #29 for development version
0.0.29.

Step 29 completes seasonal exact-diffuse observed-information and natural
innovation-covariance inference. After that merge, the plan contains:

- **6 major technical workstreams**;
- **7 numbered core milestones**;
- approximately **12–16 independently reviewable projects** after ecosystem
  work is split.

This is a delivery inventory, not a claim that every research extension already
has a production-ready derivation.

## Workstream 1: remaining seasonal exact-diffuse posterior operations

Status: next implementation sequence.

14. Add original-level and transformed-scale seasonal exact-diffuse forecast
    paths and interval contracts.

A later independent project may add seasonal conditional simulation smoothing
after forecasting contracts stabilize.

The remaining operations must reuse the 0.0.26 original-level state, 0.0.27
fitted-model contract, 0.0.28 posterior facade, and 0.0.29 likelihood-inference
contract. They must not introduce a second seasonal state convention.

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

The dense implementations remain transparent moderate-sample references rather
than large-data performance claims.

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

1. Merge PR #29 after final synchronized CI.
2. Add seasonal exact-diffuse forecasting uncertainty.
3. Add seasonal conditional simulation smoothing only after the interval
   contracts are stable.
4. Run the diffuse `L2` derivation as a separate theory-and-validation project.
5. Add robust, profile, and parameter-aware uncertainty after posterior
   contracts are stable.
6. Introduce sparse execution before broad GIS adapters or automatic model
   search.
7. Complete ecosystem and release engineering after public numerical contracts
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
