# Step 26 handoff: seasonal exact diffuse state-space foundation

## Repository position

- development version: `0.0.26`;
- branch: `agent/seasonal-exact-diffuse-state-space`;
- pull request: PR #26, `Add seasonal exact diffuse state-space foundation`;
- base: version 0.0.25 on `main` at merge commit
  `ac7c049e16889046391e8b324970dbc8d9e8798b`;
- package exports, version metadata, README, documentation home, roadmap,
  project status, remaining-work inventory, method guide, example, and MkDocs
  navigation are synchronized through 0.0.26;
- one-shot formatting and status-maintenance workflows have self-deleted and are
  absent from the formal PR diff;
- authoritative validation: pending on this ordinary synchronized head.

## Delivered API

```python
specification = build_exact_seasonal_integrated_state_space(
    transformed_model,
    ordinary_integration_order=d,
    seasonal_integration_order=D,
    seasonal_period=s,
)
result = specification.filter(original_level_data)
```

Functional routes:

```python
result = exact_seasonal_integrated_filter(
    original_level_data,
    transformed_model,
    d,
    D,
    s,
)
log_likelihood = exact_seasonal_integrated_loglikelihood(
    original_level_data,
    transformed_model,
    d,
    D,
    s,
)
```

## Mathematical construction

The transformed stationary observation is

\[
x_t=(1-B)^d(1-B^s)^D y_t=Z\beta_t.
\]

For

\[
\delta(B)=1+\delta_1B+\cdots+\delta_mB^m,
\qquad m=d+Ds,
\]

the positive-seasonal-order state stores

\[
[y_t,y_{t-1},\ldots,y_{t-m+1},\beta_t].
\]

Its first block follows

\[
y_t=-\sum_{j=1}^{m}\delta_jy_{t-j}+Z\beta_t.
\]

All `mn` original-level lag coordinates receive exact diffuse covariance. The
transformed state receives its stationary finite mean and covariance. No finite
large diffuse scale is introduced.

When `D=0`, the constructor delegates to the existing ordinary exact diffuse
builder. This preserves exact matrix and likelihood equivalence with the
pre-0.0.26 ordinary route.

## Verification design

The validation suite covers:

1. a closed-form scalar seasonal random walk;
2. explicit `(1-B)(1-B^2)` polynomial and companion matrices;
3. direct recovery of the complete combined-difference residual series;
4. exact `D=0` equivalence with ordinary integration;
5. delayed diffuse completion under missing observations;
6. rejection of a nonstationary transformed subsystem;
7. immutable polynomial and initialization arrays;
8. argument validation;
9. the complete inherited package suite.

## Deliberate boundaries

Step 26 does not provide:

- optimizer-facing seasonal exact diffuse MLE;
- seasonal exact diffuse smoothing and disturbance-smoothing facades;
- seasonal exact diffuse likelihood inference;
- original-level seasonal exact diffuse forecast intervals;
- parameter-aware paths;
- sparse or parallel execution.

The state-space contract in this stage is intended to be reused by all of those
later operations.

## Remaining roadmap after Step 26

The next core milestone is optimizer-facing seasonal exact diffuse MLE. After
that, seasonal posterior operations, the diffuse `L2` lag-one recursion,
parameter-aware inference, sparse execution, model selection, exogenous inputs,
GIS adapters, and release engineering remain.

After Step 26, ten numbered core milestones remain in the current inventory,
with roughly 15--19 separately reviewable projects after ecosystem work is
split.

## Merge checklist

Before marking PR #26 ready:

1. pass complete CI on the final code and documentation head;
2. record run number, run ID, test count, total branch coverage, and new-module
   coverage;
3. confirm analytic seasonal-random-walk and ordinary-equivalence tests pass on
   every supported platform;
4. confirm package exports, version metadata, README, documentation home,
   roadmap, project status, remaining-work inventory, navigation, and example
   are synchronized;
5. confirm no temporary workflow or generated artifact remains;
6. confirm no unresolved review thread remains;
7. mark ready and squash-merge;
8. start optimizer-facing seasonal exact diffuse MLE from the resulting `main`.
