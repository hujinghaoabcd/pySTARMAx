# Step 24 handoff: exact diffuse forecast intervals

## Repository position

- development version: `0.0.24`;
- branch: `agent/exact-diffuse-forecast-intervals`;
- pull request: PR #24, `Add exact diffuse forecast intervals`;
- base: version 0.0.23 on `main` at merge commit
  `b7bd9e3b55e856ba43e5da9cb82e8616bd6c132f`;
- final validation: pending on the synchronized implementation and documentation
  head.

## Delivered API

Fitted original-level interval:

```python
interval = fitted_exact_model.predict_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

Fitted highest ordinary-difference interval:

```python
interval = fitted_exact_model.predict_differenced_interval(
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

Low-level original-level path simulation:

```python
paths = simulate_exact_diffuse_forecast_paths(
    fitted_exact_model.filter(),
    steps=12,
    n_simulations=5000,
    random_state=2026,
)
```

Low-level original-level interval:

```python
interval = exact_diffuse_forecast_interval(
    fitted_exact_model.filter(),
    steps=12,
    level=0.95,
    n_simulations=5000,
    random_state=2026,
)
```

The interval result reuses the immutable public `ForecastInterval` contract.

## Terminal propriety contract

The exact diffuse filter retains finite and diffuse covariance components:

\[
P_T(\kappa)=P_{\ast,T}+\kappa P_{\infty,T},
\qquad \kappa\rightarrow\infty.
\]

Forecast simulation requires

\[
\operatorname{rank}(P_{\infty,T\mid T})=0.
\]

A positive final diffuse rank means that the terminal state distribution remains
improper. Step 24 raises rather than replacing unresolved uncertainty with a
large finite variance.

This check is applied before terminal-state drawing for both original-level and
highest-difference intervals.

## Simulation recursion

After diffuse resolution,

\[
\alpha_T\mid Y\sim
\mathcal N(a_{T\mid T},P_{\ast,T\mid T}).
\]

For simulation path `b`,

\[
\alpha_T^{(b)}
=
a_{T\mid T}+L_Tz_0^{(b)},
\]

\[
\alpha_{T+h}^{(b)}
=
c+T\alpha_{T+h-1}^{(b)}+R\eta_{T+h}^{(b)},
\]

where

\[
z_0^{(b)}\sim\mathcal N(0,I),
\qquad
\eta_{T+h}^{(b)}\sim\mathcal N(0,Q).
\]

The original-level path is

\[
y_{T+h}^{(b)}=Z\alpha_{T+h}^{(b)}.
\]

The highest ordinary-difference path uses a projection matrix that selects the
transformed STARMA substate from the same augmented exact state path.

## Why the integrated state is simulated directly

The exact state already contains

\[
[y_t,\Delta y_t,\ldots,\Delta^{d-1}y_t,\beta_t].
\]

Direct propagation therefore preserves:

- terminal uncertainty in all integrated levels;
- finite covariance between integrated blocks and the transformed substate;
- future cumulative innovation effects;
- the same transition and selection matrices used by exact diffuse fitting.

No external inverse-differencing history is required for the original-level
interval.

## Point forecast convention

`ForecastInterval.mean` is the deterministic recursive point forecast. It is
not the Monte Carlo sample mean. Thus the reported center is independent of
seed and simulation count and agrees with:

- `ExactDiffuseKalmanSTARIMA.predict()` on the original level scale;
- `ExactDiffuseKalmanSTARIMA.predict_differenced()` on the highest ordinary-
  difference scale.

## Covariance policy

The final finite covariance is symmetrized and factorized through an
eigendecomposition. Floating-point-scale negative eigenvalues are clipped to
zero. Materially negative eigenvalues raise.

Step 24 adds no diagonal jitter and no arbitrary diffuse scale.

## Validation design

The tests cover:

1. exact equality with ordinary stationary Kalman forecast paths when both
   filters have zero diffuse rank and identical terminal moments;
2. a scalar analytic state process with future means `1.5`, `2.0` and variances
   `5.0`, `6.0`;
3. random-walk variance increasing linearly with forecast horizon;
4. interval mean equality with original-level recursive point forecasts;
5. highest-difference interval mean equality with transformed recursive point
   forecasts;
6. deterministic reproducibility under a fixed integer seed;
7. second-order integrated-state projection;
8. immutable interval arrays;
9. unresolved terminal diffuse-rank refusal;
10. steps, level, simulation count, and result-type validation;
11. the complete inherited package suite.

## First CI finding

Initial CI #509 reached quality, strict MkDocs, distributions, and packaging
success. The only test failure was a reference-array shape mismatch in the
random-walk fixture: a `(3, 1)` simulated mean was compared with a `(3,)`
reference. The reference was corrected to retain its location axis. The failure
did not indicate a numerical implementation defect.

## Deliberate boundaries

Step 24 does not claim:

- parameter-estimation uncertainty in forecast intervals;
- exact diffuse seasonal state augmentation;
- exact diffuse simulation smoothing;
- public analytic horizon-by-horizon or cross-time forecast covariance;
- separate measurement-noise simulation;
- sparse state propagation;
- continuation from a user-supplied terminal filter result through the fitted
  facade;
- predictive distributions integrated over parameter uncertainty.

## Files added or changed

Core:

- `src/pystarmax/exact_diffuse_forecasting.py`;
- `src/pystarmax/exact_diffuse_model.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Validation:

- `tests/test_exact_diffuse_forecasting.py`;
- `tests/test_exact_diffuse_forecast_intervals.py`.

Documentation and example:

- `docs/exact_diffuse_forecast_intervals.md`;
- `examples/exact_diffuse_forecast_intervals.py`;
- `mkdocs.yml`;
- this Step 24 handoff;
- README, documentation home, roadmap, project status, and exact diffuse MLE
  cross-links are to be synchronized before merge.

## Next recommended stage

After Step 24, the highest-value exact diffuse extensions are:

1. exact diffuse simulation smoothing after resolving the required diffuse
   backward-sampling recursions;
2. seasonal exact diffuse state augmentation and likelihood;
3. parameter-aware forecast paths using either asymptotic draws or bootstrap
   refitting;
4. robust or sandwich likelihood inference;
5. sparse state and spatial operators;
6. order selection and exogenous regressors.

Exact diffuse simulation smoothing should remain separate from forecast path
simulation. Forecast paths condition on observations only through the terminal
filter posterior and simulate future disturbances; simulation smoothing samples
latent states and disturbances over the observed interval conditional on the
complete data.

## Merge checklist

Before marking PR #24 ready:

1. obtain a complete all-platform CI success on the final code and documentation
   head;
2. record run number, run ID, test count, total branch coverage, and new-module
   coverage here and in `PROJECT_STATUS.md`;
3. update the PR body with final method scope and validation;
4. confirm no unresolved review threads or generated artifacts remain;
5. mark ready and squash-merge;
6. create the next branch from the resulting `main` merge commit.
