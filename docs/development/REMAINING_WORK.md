# Remaining work inventory

## Snapshot

This inventory is based on `main` version 0.0.25 at
`ac7c049e16889046391e8b324970dbc8d9e8798b` and draft PR #26 for version
0.0.26.

PR #26 completes the fixed-parameter seasonal exact diffuse state augmentation,
filter, and original-level likelihood foundation. After that merge, the
remaining plan contains:

- **7 major technical workstreams**;
- **10 numbered core milestones**;
- roughly **15--19 separately reviewable projects** after ecosystem work is
  split into independent pull requests.

The count is a delivery plan, not a claim that every research extension already
has a closed-form production solution. In particular, diffuse lag-one
covariance, boundary-aware robust inference, and scalable seasonal diffuse
execution still require method research.

## Workstream 1: seasonal exact diffuse maximum likelihood

Status: fixed-parameter foundation active in PR #26; optimizer not started.

11. Add optimizer-facing seasonal exact diffuse MLE with multiplicative ordinary
    and seasonal factor parameters, factor-based parameter counting, original-
    level AIC/BIC, and expanded admissibility enforcement.

The optimizer must rebuild the ordered multiplicative operator expansion,
stationary transformed state, combined original-level seasonal augmentation, and
exact diffuse filter at every candidate. It must not optimize deterministic
expanded cross-lag matrices as independent parameters.

## Workstream 2: seasonal exact diffuse posterior operations

Status: blocked until Workstream 1 stabilizes the fitted model contract.

12. Extend exact diffuse fixed-interval state and primitive disturbance
    smoothing to the seasonal augmented state.
13. Add seasonal exact diffuse observed-information inference and natural
    innovation covariance inference.
14. Add original-level and transformed-scale seasonal exact diffuse forecasts,
    paths, and interval contracts.

All operations should reuse the 0.0.26 original-level state contract rather than
introducing separate incompatible seasonal augmentations.

## Workstream 3: diffuse cross-time covariance theory

Status: research required.

15. Derive, implement, and independently validate the diffuse `L2` recursion
    required for lag-one smoothed state autocovariance.
16. Use that recursion to expose cross-time state-disturbance and primitive
    innovation covariance without large-variance substitution.

This workstream must not replace exact diffuse initialization with an
approximate large finite covariance while retaining the label “exact”.

## Workstream 4: stronger likelihood and parameter uncertainty

Status: not started.

17. Add robust/sandwich and profile-likelihood inference with explicit boundary,
    rank-deficiency, and singular-curvature policies.
18. Propagate parameter uncertainty into forecast paths, intervals, and selected
    smoother summaries through validated asymptotic draws or bootstrap
    refitting.

Analytic derivatives may be delivered here if they are checked against the
existing finite-difference objective references.

## Workstream 5: scalable numerical execution

Status: not started.

19. Add sparse spatial/state operators and memory-aware filtering, smoothing,
    forecasting, and simulation paths.
20. Add chunked or parallel execution with deterministic seed partitioning,
    reproducibility contracts, and dense-reference equivalence.

The dense exact diffuse implementations remain transparent moderate-sample
references, not large-data performance claims.

## Workstream 6: model specification automation

Status: not started; split into independent pull requests.

- smooth admissibility parameterization;
- automatic ordinary and seasonal order selection;
- automatic spatial-lag and weight-set selection;
- comparable conditional versus exact-diffuse model-selection reporting without
  mixing likelihood conventions.

## Workstream 7: exogenous inputs and ecosystem integration

Status: not started; split into independent pull requests.

- exogenous regressors and intervention variables;
- GIS-oriented spatial-weight and data adapters;
- cross-language numerical reference fixtures;
- public PyPI release and signed release automation;
- explicitly time-varying and other extended state-space models.

## Recommended sequence

1. Finish and merge PR #26.
2. Implement optimizer-facing seasonal exact diffuse MLE.
3. Add seasonal smoothing, inference, and forecasting in separate reviewable
   stages.
4. Run the diffuse `L2` derivation as an independent theory-and-validation
   project.
5. Add robust and parameter-aware inference after posterior contracts stabilize.
6. Introduce sparse execution before broad GIS adapters or automatic model
   search so higher-level APIs do not hard-code dense assumptions.
7. Complete ecosystem and release engineering after public numerical contracts
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
