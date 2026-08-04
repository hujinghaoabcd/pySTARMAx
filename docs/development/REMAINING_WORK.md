# Remaining work inventory

## Snapshot

This inventory is based on `main` version 0.0.24 at
`0639f4bd932aa86020215b50b6ef9867bb5ad566` and the active draft PR #25 for
version 0.0.25.

The remaining work is best described as:

- **8 major workstreams** including the active simulation-smoothing stage;
- **20 independently reviewable milestones** under the current roadmap;
- **1 active milestone group** in PR #25;
- **7 later technical workstreams** after PR #25 is merged.

The count is a delivery plan rather than a claim that every research extension
has a known final solution. The diffuse lag-one covariance recursion, robust
boundary inference, and scalable seasonal diffuse implementation still require
method research before their production scope can be fixed.

## Workstream 1: release and status consistency

Status: active inside PR #25.

1. Synchronize README capability and limitation text with versions 0.0.24 and
   0.0.25.
2. Add completed 0.0.24 and 0.0.25 sections to the roadmap and remove delivered
   items from `Next priorities`.
3. Replace the stale 0.0.23 project-status snapshot with PR #24 and PR #25
   validation records.
4. Update exact diffuse MLE cross-links and limitations so forecast intervals
   and simulation smoothing are not still described as unavailable.

These are documentation defects, not missing numerical capabilities, but they
must be closed before 0.0.25 is considered complete.

## Workstream 2: exact diffuse simulation smoothing

Status: active in draft PR #25.

5. Complete all-platform validation of the dense flat-diffuse-coordinate
   simulation smoother.
6. Finalize public API, immutable diagnostics, method guide, example, and Step
   25 handoff.
7. Record test count, total coverage, new-module coverage, and numerical
   agreement with the exact information smoother.
8. Remove temporary artifacts, close review threads, and merge version 0.0.25.

The implemented scope draws complete state and observation paths. Primitive
innovation paths, lag-one covariance, and cross-time disturbance covariance are
not silently included in this milestone.

## Workstream 3: seasonal exact diffuse foundation

Status: not started.

9. Derive and implement ordinary-seasonal integrated state augmentation for
   multiplicative seasonal STARIMA without conditioning away the transformed
   history.
10. Add exact diffuse filtering and original-level likelihood for the seasonal
    augmented state.
11. Add optimizer-facing seasonal exact diffuse MLE with correct factor
    parameter counting and expanded admissibility checks.

## Workstream 4: seasonal exact diffuse posterior operations

Status: blocked by Workstream 3.

12. Extend exact diffuse fixed-interval state and primitive disturbance
    smoothing to the seasonal augmented state.
13. Add seasonal exact diffuse likelihood inference and natural innovation
    covariance inference.
14. Add original-level and transformed-scale seasonal exact diffuse forecasting
    and interval contracts.

## Workstream 5: diffuse cross-time covariance theory

Status: research required.

15. Derive, implement, and independently validate the nontrivial diffuse `L2`
    recursion needed for lag-one smoothed state autocovariance.
16. Use that recursion to expose cross-time state-disturbance and primitive
    innovation covariance without large-variance substitution.

This workstream must not be replaced by approximate diffuse initialization while
being labelled exact.

## Workstream 6: stronger likelihood and parameter uncertainty

Status: not started.

17. Add robust/sandwich and profile-likelihood inference with explicit boundary
    and singular-curvature policies.
18. Propagate parameter uncertainty into forecast paths, intervals, and selected
    smoother summaries through asymptotic draws or bootstrap refitting.

Analytic derivatives can be delivered as part of this workstream if they are
validated against finite-difference references; they are not a prerequisite for
calling the existing observed-information implementation correct.

## Workstream 7: scalable numerical execution

Status: not started.

19. Add sparse spatial/state operators and memory-aware filtering, smoothing,
    forecasting, and simulation paths.
20. Add chunked or parallel execution with reproducibility contracts and dense
    reference equivalence.

The current dense simulation smoother is intentionally a transparent reference,
not a large-data performance claim.

## Workstream 8: model specification and ecosystem integration

Status: not started.

This later workstream contains the next package-level expansion after the 20
core milestones above:

- smooth admissibility parameterization;
- automatic order and spatial-lag selection;
- exogenous regressors and intervention variables;
- GIS-oriented weight/data adapters;
- cross-language reference fixtures;
- public PyPI release automation;
- time-varying and other explicitly extended state-space models.

These items should be split into separate PRs when activated. They are grouped
here because their exact sequence depends on results from the seasonal,
cross-time covariance, and sparse-computation stages.

## Recommended sequence

1. Finish and merge PR #25.
2. Close the release/status consistency defects in the same merge.
3. Implement seasonal exact diffuse state augmentation and filtering before any
   seasonal smoother facade.
4. Run the diffuse `L2` derivation as a separately reviewed research stage.
5. Add robust/parameter-aware inference after the exact posterior contracts are
   stable.
6. Introduce sparse execution before broad GIS adapters or automatic model
   search, so higher-level features do not hard-code dense assumptions.
7. Finish ecosystem and release engineering only after public API boundaries
   are stable.

## Completion definition

A milestone is complete only when it has:

- an explicit mathematical and scale contract;
- immutable typed public results where applicable;
- analytic or independently constructed numerical references;
- missing-data and rank-deficiency behavior;
- cross-platform tests;
- formatting, lint, typing, strict documentation, and package-build success;
- updated README, roadmap, project status, method guide, example, and handoff;
- no temporary workflows, generated artifacts, or unresolved review threads.
