# pySTARMAx project status

## Repository position

- `main` is version `0.0.30` at squash merge commit
  `02bb1a14409e3b679cacb90970c5f44b6314694d`.
- Active branch: `agent/seasonal-exact-diffuse-simulation-smoothing`.
- Active pull request: PR #31,
  `Add seasonal exact diffuse simulation smoothing`.
- Development version: `0.0.31`.
- PR #1 through PR #30 have been squash-merged.

## Public model families

The package exposes:

- conditional least-squares `STARMA`, `STARIMA`, and `SeasonalSTARIMA`;
- stationary Gaussian `KalmanSTARMA`;
- conditional ordinary-integrated `KalmanSTARIMA`;
- conditional multiplicative seasonal `SeasonalKalmanSTARIMA`;
- original-level ordinary exact-diffuse `ExactDiffuseKalmanSTARIMA`;
- fixed-parameter seasonal exact-diffuse state construction through
  `ExactSeasonalIntegratedStateSpace`;
- optimizer-facing original-level seasonal exact-diffuse MLE through
  `SeasonalExactDiffuseKalmanSTARIMA`;
- seasonal exact-diffuse smoothing, inference, forecasting, and complete
  conditional simulation smoothing.

Conditional transformed-data likelihoods and original-level exact-diffuse
likelihoods are separate contracts. Their likelihoods, AIC, BIC, Hessians, and
parameter covariance estimates are not interchangeable.

## Completed numerical foundations

### Spatial and temporal conventions

- ordered spatial weights with identity `W0`;
- supplied orientation preserved for non-symmetric weights;
- positive moving-average sign and matching inverse-MA companion convention;
- explicit ordinary and seasonal differencing histories;
- multiplicative seasonal factors applied on the left of ordinary factors;
- deterministic cross-lag aggregation without spatial-basis projection.

### Estimation and inference

- stationary conditional and Gaussian maximum likelihood;
- ordinary conditional integrated Gaussian estimation;
- multiplicative seasonal conditional Gaussian estimation;
- ordinary and seasonal original-level exact-diffuse MLE;
- scalar, diagonal, and full-Cholesky innovation covariance;
- missing and partially observed measurement rows;
- expanded AR stationarity and positive-sign inverse-MA invertibility checks;
- observed-information inference across fitted Gaussian and exact-diffuse
  families;
- natural innovation-covariance delta-method inference.

### Posterior operations and forecasting

- ordinary Kalman fixed-interval smoothing;
- ordinary and seasonal exact-diffuse fixed-interval smoothing;
- ordinary and seasonal primitive innovation/state-disturbance smoothing;
- ordinary Gaussian and exact-diffuse forecast intervals;
- seasonal exact-diffuse original-level and transformed paths and intervals;
- dense ordinary and seasonal exact-diffuse conditional simulation smoothing;
- pathwise ordinary-seasonal level restoration and transformed-state projection;
- explicit rejection of unresolved diffuse directions.

### Diagnostics and evaluation

- STACF and STPACF variants;
- space-time portmanteau tests;
- admissibility diagnostics with companion matrices and eigenvalues;
- rolling-origin evaluation;
- interval scoring and coverage diagnostics;
- bootstrap prediction workflows.

## Completed in 0.0.31

### Public simulation-smoothing API

The fitted seasonal exact-diffuse estimator exposes:

```python
paths = model.simulate_smoothing_paths(
    n_simulations=1000,
    random_state=42,
)
```

The equivalent functional API is:

```python
from pystarmax import seasonal_exact_diffuse_simulation_smoother

paths = seasonal_exact_diffuse_simulation_smoother(
    filter_result,
    integrated_state_space,
    n_simulations=1000,
    random_state=42,
)
```

Passing explicit data to the fitted method starts a fresh exact-diffuse filter
under the fitted parameters rather than continuing from the training terminal
posterior.

### Exact conditioning contract

The seasonal facade reuses the generic dense exact-diffuse source-conditioning
algorithm. The complete augmented state is represented as a linear function of
initial diffuse coordinates, initial proper finite coordinates, and primitive
innovations. Observed original-level cells are imposed as exact linear
constraints.

Diffuse coordinates are eliminated analytically. The remaining proper Gaussian
source vector is conditioned and sampled. No finite large covariance and no
second seasonal simulation algorithm are introduced.

For

\[
m=d+Ds,
\]

the stationary transformed-state block begins at
`m * n_locations`. The returned transformed state and observation paths are
direct projections of the same complete conditional draws.

### Result contract

`SeasonalExactDiffuseSimulationSmootherResult` exposes immutable:

- complete augmented-state and original-observation paths;
- transformed-state and transformed-observation paths;
- posterior complete-state marginal means and covariances;
- posterior transformed-state marginal means and covariances;
- diffuse rank, conditioning rank, source dimensions, and numerical
  verification discrepancies.

The wrapper verifies its transformed arrays against the complete paths and
posterior blocks during construction.

### Proper-posterior policy

Simulation requires `final_diffuse_rank == 0`. If the observations do not
identify every diffuse direction, the method raises `RuntimeError`. The supplied
seasonal integrated specification must also be the exact state-space object used
by the filter result.

### Independent references

The 0.0.31 suite covers:

- a period-two seasonal-random-walk bridge with closed-form conditional means
  and variances;
- exact reproduction of every observed original-level cell;
- pathwise `x_t = y_t - y_(t-2)` identities;
- multivariate partial-location observation masks;
- exact transformed-state projections;
- exact reduction to the ordinary exact-diffuse simulation smoother when
  seasonal integration is zero;
- fitted training-result reuse and fresh new-data initialization;
- unresolved diffuse-rank and mismatched-state rejection;
- immutable results and public exports.

## Authoritative validation

Initial implementation CI #641 identified only Black formatting in three new or
expanded Python files before later quality steps ran. A temporary branch-only
workflow pinned Black 26.5.1, applied the formatter output, committed it, and
deleted itself. The formal PR diff contains no workflow file.

Authoritative implementation CI #644, run ID `30962744283`, passed on exact head
`91ef1f95aab8371af14899e87c0c4ce9f1813c6d`:

- 259 tests passed;
- total branch coverage: 87.31%;
- `src/pystarmax/seasonal_exact_diffuse_simulation_smoothing.py` branch
  coverage: 88.8%;
- Black, isort, Ruff, and mypy passed;
- diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11–3.14.

A synchronized documentation and merge-gate CI remains required on the final PR
head.

## Numerical and research safeguards

- no exact-diffuse method substitutes a finite large covariance for diffuse
  directions;
- missing observations are skipped, not silently imputed;
- observed cells are imposed as exact conditional constraints;
- non-symmetric spatial operators retain orientation;
- expanded multiplicative cross terms are deterministic, not extra parameters;
- covariance optimizer coordinates and natural covariance elements remain
  distinct;
- singular-curvature inference is explicit rather than silent;
- seasonal transformed paths are projections from complete augmented paths;
- result arrays are immutable;
- dense implementations are moderate-sample references, not large-data claims.

## Current limitations

Version 0.0.31 does not yet provide:

- diffuse lag-one state covariance during a nontrivial diffuse phase;
- cross-time state-disturbance or primitive-innovation covariance;
- robust or sandwich covariance;
- profile-likelihood intervals;
- parameter-uncertainty propagation into forecasts or smoother summaries;
- analytic score or Hessian recursions;
- sparse, chunked, or parallel numerical execution.

The diffuse `L2` recursion is the next theory-and-validation project.

## Remaining plan

After Step 31, the delivery inventory contains:

- 5 major technical workstreams;
- 6 numbered core milestones;
- approximately 10–14 independently reviewable projects.

The next implementation stage is the diffuse lag-one covariance derivation and
independent validation project.

## Merge checklist for PR #31

1. Synchronize README, documentation home, method guide, example, navigation,
   roadmap, inventory, status, and Step 31 handoff.
2. Pass complete CI on the exact final head.
3. Record final run number, run ID, head, test count, and coverage.
4. Confirm the formal diff contains no temporary workflow or generated artifact.
5. Confirm no submitted review or unresolved review thread remains.
6. Mark PR #31 ready and squash-merge version 0.0.31.
7. Create the diffuse lag-one covariance branch from the new `main`.

## Handoff documents

Read these before the next stage:

- `docs/development/STEP_31_SEASONAL_EXACT_DIFFUSE_SIMULATION_SMOOTHING.md`;
- `docs/seasonal_exact_diffuse_simulation_smoothing.md`;
- `docs/development/STEP_30_SEASONAL_EXACT_DIFFUSE_FORECASTING.md`;
- `docs/seasonal_exact_diffuse_forecasting.md`;
- `docs/development/REMAINING_WORK.md`;
- `docs/roadmap.md`.
