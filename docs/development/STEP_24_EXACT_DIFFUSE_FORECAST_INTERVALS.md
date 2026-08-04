# Step 24 handoff: exact diffuse forecast intervals

## Repository position

- development version: `0.0.24`;
- branch: `agent/exact-diffuse-forecast-intervals`;
- pull request: PR #24, `Add exact diffuse forecast intervals`;
- base: version 0.0.23 on `main` at merge commit
  `b7bd9e3b55e856ba43e5da9cb82e8616bd6c132f`;
- authoritative implementation and documentation validation: GitHub Actions
  CI #518, run ID `30941761879`;
- validation result: 207 tests passed and 87.30% total branch coverage;
- `src/pystarmax/exact_diffuse_forecasting.py` coverage: 87.8%;
- Black, isort, Ruff, mypy, strict MkDocs, source/wheel construction, Twine,
  Ubuntu, Windows, and macOS Python 3.11–3.14 passed.

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

Low-level original-level path simulation and interval:

```python
paths = simulate_exact_diffuse_forecast_paths(
    fitted_exact_model.filter(),
    steps=12,
    n_simulations=5000,
    random_state=2026,
)

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
large finite variance, discarding the unresolved direction, or using only the
finite covariance component.

This check applies to both original-level and highest-difference intervals.

## Simulation recursion

After diffuse resolution,

\[
\alpha_T\mid Y\sim
\mathcal N(a_{T\mid T},P_{\ast,T\mid T}).
\]

For path `b`,

\[
\alpha_T^{(b)}=a_{T\mid T}+L_Tz_0^{(b)},
\]

\[
\alpha_{T+h}^{(b)}
=c+T\alpha_{T+h-1}^{(b)}+R\eta_{T+h}^{(b)},
\]

with

\[
z_0^{(b)}\sim\mathcal N(0,I),
\qquad
\eta_{T+h}^{(b)}\sim\mathcal N(0,Q).
\]

Original-level paths use

\[
y_{T+h}^{(b)}=Z\alpha_{T+h}^{(b)}.
\]

Highest ordinary-difference paths project the same simulated augmented states
through a matrix that selects the transformed STARMA substate.

## Why the integrated state is simulated directly

The exact state already contains

\[
[y_t,\Delta y_t,\ldots,\Delta^{d-1}y_t,\beta_t].
\]

Direct propagation preserves:

- terminal uncertainty in integrated levels and lower differences;
- covariance between integrated blocks and the transformed substate;
- future cumulative innovation effects;
- the same transition, selection, and design conventions used by exact diffuse
  fitting and filtering.

No external inverse-differencing history is needed for the original-level
interval.

## Point forecast convention

`ForecastInterval.mean` is the deterministic recursive forecast, not the Monte
Carlo sample mean. It therefore remains independent of the seed and simulation
count and agrees with:

- `ExactDiffuseKalmanSTARIMA.predict()` on original levels;
- `ExactDiffuseKalmanSTARIMA.predict_differenced()` on the highest ordinary-
  difference scale.

## Covariance policy

The final finite covariance is symmetrized and factorized by eigendecomposition.
Floating-point-scale negative eigenvalues are clipped to zero; materially
negative eigenvalues raise. No diagonal jitter or arbitrary diffuse scale is
introduced.

## Validation scope

The tests cover:

1. exact equality with ordinary stationary Kalman forecast paths when the two
   filters have zero diffuse rank and identical terminal moments;
2. a scalar analytic state process with future means `1.5`, `2.0` and variances
   `5.0`, `6.0`;
3. random-walk variance increasing linearly with forecast horizon;
4. original-level interval mean equality with `predict()`;
5. highest-difference interval mean equality with `predict_differenced()`;
6. deterministic reproducibility under a fixed integer seed;
7. second-order integrated-state projection;
8. immutable public interval arrays;
9. unresolved terminal diffuse-rank refusal;
10. steps, level, simulation count, and result-type validation;
11. the complete inherited package suite.

## CI findings and fixes

Initial CI #509 showed one test-fixture shape mismatch: a `(3, 1)` simulated
random-walk mean was compared with a `(3,)` reference. The reference was fixed
to retain its location axis; the numerical implementation was unchanged.

CI #515 then showed only a Black formatting difference in the runnable example.
The example was formatted. CI #518 passed the synchronized implementation,
tests, example, package metadata, documentation home, navigation, method guide,
and this handoff.

## Final validation

GitHub Actions CI #518, run ID `30941761879`, completed successfully:

- 207 tests passed;
- total branch coverage: 87.30%;
- exact diffuse forecast module coverage: 87.8%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

A documentation-record-only merge-gate CI is required after this handoff update.
No implementation, test, API, metadata, example, or method-guide behavior is
changed after CI #518.

## Deliberate boundaries

Step 24 does not claim:

- parameter-estimation uncertainty in forecast intervals;
- exact diffuse seasonal state augmentation;
- exact diffuse simulation smoothing;
- public analytic horizon-by-horizon or cross-time forecast covariance;
- separate measurement-noise simulation;
- sparse state propagation;
- predictive distributions integrated over parameter uncertainty.

## Files added or changed

Core and metadata:

- `src/pystarmax/exact_diffuse_forecasting.py`;
- `src/pystarmax/exact_diffuse_model.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Validation:

- `tests/test_exact_diffuse_forecasting.py`;
- `tests/test_exact_diffuse_forecast_intervals.py`.

Documentation and example:

- `docs/exact_diffuse_forecast_intervals.md`;
- `docs/index.md`;
- `docs/development/STEP_24_EXACT_DIFFUSE_FORECAST_INTERVALS.md`;
- `examples/exact_diffuse_forecast_intervals.py`;
- `mkdocs.yml`.

## Next recommended stage

The next branch should separate forecast simulation from simulation smoothing.
Recommended order:

1. exact diffuse simulation smoothing after deriving the required backward
   sampling recursion without fabricating unavailable lag-one diffuse moments;
2. seasonal exact diffuse state augmentation and likelihood;
3. parameter-aware forecast paths using asymptotic or bootstrap parameter draws;
4. robust likelihood inference;
5. sparse state and spatial operators;
6. order selection and exogenous regressors.

Forecast paths condition on observations through the terminal filter posterior
and simulate future disturbances. Simulation smoothing instead samples latent
states and disturbances over the observed interval conditional on the complete
data; it must be implemented and documented as a distinct method.

## Merge checklist

1. Run the validation-record-only merge-gate CI.
2. Update the PR body with the final validation record.
3. Confirm no unresolved review thread or temporary artifact remains.
4. Mark PR #24 ready and squash-merge it into `main`.
5. Create the next development branch from the resulting merge commit.
