# Step 27 handoff: seasonal exact diffuse maximum likelihood

## Repository position

- development version: `0.0.27`;
- branch: `agent/seasonal-exact-diffuse-mle`;
- pull request: PR #27, `Add seasonal exact diffuse maximum likelihood`;
- base: version 0.0.26 on `main` at merge commit
  `e626ff330c2e6966d1474883656bf3543f56d84a`;
- implementation validation: CI #576, run ID `30949430144`;
- final synchronized documentation merge-gate: pending on this head.

## Delivered API

```python
from pystarmax import SeasonalExactDiffuseKalmanSTARIMA

model = SeasonalExactDiffuseKalmanSTARIMA(
    ar_order=p,
    integration_order=d,
    ma_order=q,
    seasonal_ar_order=P,
    seasonal_integration_order=D,
    seasonal_ma_order=Q,
    seasonal_period=s,
    covariance_type="scalar",
)
result = model.fit(data, weights)
```

The immutable result type is
`SeasonalExactDiffuseKalmanSTARIMAResult`.

Fitted routes:

```python
model.filter()
model.to_transformed_state_space()
model.to_state_space()
model.admissibility()
model.predict_differenced(steps=h)
model.predict(steps=h)
model.fitted_original()
```

## Optimization contract

The free dynamic parameters remain the ordinary and seasonal factor
coefficients. For each optimizer candidate the implementation:

1. decodes ordinary AR, seasonal AR, ordinary MA, and seasonal MA factors;
2. expands the ordered multiplicative matrix polynomials;
3. aggregates equal temporal lags without spatial-basis projection;
4. builds the stationary transformed state-space model;
5. constructs the 0.0.26 original-level seasonal exact diffuse state;
6. evaluates the exact diffuse likelihood on the original observations.

Expanded cross-lag matrices are deterministic functions of the factor
coefficients. They are never optimized independently and do not contribute
additional AIC/BIC parameters.

## Likelihood and admissibility

The objective is the original-level exact diffuse Gaussian likelihood for

\[
(1-B)^d(1-B^s)^D y_t.
\]

It is not interchangeable with the conditional transformed-data likelihood of
`SeasonalKalmanSTARIMA`.

The full expanded AR and inverse-MA companion matrices determine admissibility.
Candidate violations receive the existing smooth quadratic penalty and the final
candidate is checked again before a result is returned.

Innovation covariance supports scalar, diagonal, and full-Cholesky codecs.

## Verification design

The test suite covers:

1. closed-form seasonal random-walk MLE for drift, variance, and likelihood;
2. combined `(1-B)(1-B^2)` integration;
3. factor parameter counting and deterministic AR cross-lag expansion;
4. exact `D=0`, `P=Q=0` equivalence with ordinary exact diffuse MLE;
5. missing initial observations delaying diffuse completion;
6. fitted state-space identities and original/transformed predictions;
7. immutable result arrays, public exports, and validation errors;
8. the complete inherited package suite.

## Authoritative implementation validation

GitHub Actions CI #576, run ID `30949430144`, validated exact implementation
head `246b2bcf38789adc4caf6b43b7db312d602991c7`.

Results:

- 231 tests passed;
- total branch coverage was 87.10%, above the required 80%;
- `src/pystarmax/seasonal_exact_diffuse_mle.py` branch coverage was 82.9%;
- the closed-form seasonal-random-walk MLE reference passed;
- combined ordinary-seasonal integration passed;
- factor counting and deterministic cross-lag expansion passed;
- exact ordinary-model reduction passed;
- missing-data diffuse-rank behavior passed;
- public API and immutable-result contracts passed;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic-reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

All changes after that exact implementation head are documentation and status
synchronization only. A final merge-gate CI must pass before Ready and squash
merge.

## Deliberate boundaries

Step 27 does not provide seasonal exact diffuse:

- fixed-interval smoothing;
- primitive disturbance smoothing;
- likelihood-Hessian inference;
- forecast intervals;
- conditional simulation smoothing;
- parameter-aware paths;
- sparse or parallel execution.

These operations should reuse the fitted transformed and integrated state-space
objects stored by the 0.0.27 result.

## Remaining roadmap after Step 27

The next core stages are seasonal exact diffuse posterior operations:

1. fixed-interval state and primitive disturbance smoothing;
2. observed-information and natural innovation-covariance inference;
3. original-level and transformed-scale forecasts, paths, and intervals.

After Step 27, the inventory contains six major technical workstreams, nine
numbered core milestones, and approximately 14–18 separately reviewable
projects after ecosystem tasks are split.

## Merge checklist

Before marking PR #27 ready:

1. pass complete CI on the final synchronized head;
2. confirm the closed-form seasonal random-walk and ordinary-equivalence tests
   pass on every supported platform;
3. confirm public exports, metadata, README, documentation home, roadmap,
   project status, remaining-work inventory, method guide, navigation, example,
   and this handoff are synchronized;
4. confirm no temporary workflow or generated artifact remains;
5. confirm no unresolved review thread or review submission remains;
6. mark ready and squash-merge;
7. begin seasonal exact diffuse smoothing from the resulting `main`.
