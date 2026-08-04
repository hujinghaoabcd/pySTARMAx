# Seasonal exact-diffuse likelihood inference

Version 0.0.29 adds observed-information inference for fitted
`SeasonalExactDiffuseKalmanSTARIMA` models.

The inference target is the same original-level exact-diffuse likelihood used by
the estimator. It is not the conditional likelihood of pre-differenced data.

## Public API

```python
inference = model.likelihood_inference()
```

Equivalent functional API:

```python
from pystarmax import infer_seasonal_exact_diffuse_kalman_starima

inference = infer_seasonal_exact_diffuse_kalman_starima(model)
```

The returned `LikelihoodInferenceResult` contains:

- raw optimizer-scale parameter estimates;
- the finite-difference score and observed-information Hessian;
- parameter covariance, standard errors, Wald statistics, and p-values;
- the parameter correlation matrix;
- Hessian eigenvalues, rank, condition number, and pseudoinverse status;
- finite-difference step sizes and objective-evaluation count;
- distances from the fitted expanded AR and inverse-MA admissibility limits.

All public numerical arrays are immutable.

## Likelihood contract

For

\[
x_t=(1-B)^d(1-B^s)^D y_t,
\]

the stationary transformed process has multiplicative ordinary and seasonal
AR/MA factors. At every finite-difference candidate, pySTARMAx performs the full
estimation construction again:

1. decode ordinary and seasonal AR/MA factor coordinates;
2. expand the ordered multiplicative matrix polynomials;
3. decode the scalar, diagonal, or full-Cholesky innovation covariance;
4. build the stationary transformed state-space model;
5. build the original-level seasonal exact-diffuse state;
6. evaluate the original observations with the exact-diffuse filter.

This matters because the state matrices, expanded cross lags, diffuse
observation sequence, and likelihood all depend on the candidate parameter
vector.

No large finite covariance substitutes for diffuse initialization.

## Parameter coordinates

Curvature is evaluated on the raw optimizer scale. The dynamic coordinates are
the free factor parameters retained by the fitted result:

- intercept, when enabled;
- ordinary AR factor coefficients;
- seasonal AR factor coefficients;
- ordinary MA factor coefficients;
- seasonal MA factor coefficients.

Expanded multiplicative cross-lag matrices are deterministic functions of those
factor coordinates. They are not additional parameters in the Hessian, standard
errors, AIC, or BIC.

Covariance coordinates follow the fitted covariance model:

- `scalar`: one log standard deviation;
- `diagonal`: one log standard deviation per location;
- `full`: unconstrained lower-Cholesky coordinates with log diagonal entries.

## Finite-difference curvature

The observed information is

\[
\mathcal I(\hat\psi)
=
\nabla^2[-\ell_{\mathrm{exact}}(\psi)]\big|_{\hat\psi},
\]

where `psi` is the raw optimizer vector.

Central differences are used for scores, diagonal Hessian entries, and mixed
partials. Candidate-specific steps are

\[
h_i=\max(h_{\mathrm{abs}},
          h_{\mathrm{rel}}\max(1,|\hat\psi_i|)).
\]

The defaults are:

```python
relative_step=1e-4
absolute_step=1e-6
```

Smaller steps are not automatically more accurate. Exact-diffuse rank decisions,
optimizer tolerances, floating-point cancellation, and curvature scale should be
considered together. The result retains the actual steps and score so users can
repeat the calculation with alternative settings.

## Admissibility at stencil points

The fitted model can enforce stationarity and positive-sign inverse-MA
invertibility. Every finite-difference candidate is checked using the fully
expanded recursions.

Candidates beyond an enforced boundary receive the same smooth quadratic
penalty convention used by the estimator-facing objective. Invalid covariance,
state-space, or filter candidates receive a deterministic large objective value.
The finite-difference engine rejects stencil values that cross its invalid-value
threshold.

The result also records

```python
inference.stability_boundary_distance
inference.invertibility_boundary_distance
```

Small positive distances warn that local quadratic inference may be sensitive to
boundary geometry.

## Hessian rank policy

By default, the observed-information Hessian must be positive definite and full
rank:

```python
inference = model.likelihood_inference()
```

A singular or indefinite Hessian raises `numpy.linalg.LinAlgError`. This avoids
silently reporting ordinary standard errors when the local quadratic
approximation is unidentified or has negative curvature.

A generalized inverse must be requested explicitly:

```python
inference = model.likelihood_inference(allow_singular=True)
```

Only positive eigenvalue directions above the `rcond` threshold contribute to
the covariance. Zero or negative directions are discarded. The result reports:

```python
inference.rank
inference.positive_definite
inference.used_pseudoinverse
inference.eigenvalues
inference.condition_number
```

Generalized-inverse Wald statistics are diagnostic summaries, not a replacement
for a well-identified likelihood.

## Natural innovation-covariance inference

Optimizer covariance coordinates are numerically convenient but are not the
natural variance and covariance elements. Convert them with the existing delta-
method contract:

```python
natural = inference.innovation_covariance_inference()
```

For a covariance matrix `Q`, the natural coordinates are the lower-triangular
elements:

\[
\operatorname{vech}(Q).
\]

The transformation supplies:

- natural covariance estimates;
- delta-method covariance and standard errors;
- Wald statistics and p-values;
- correlations;
- covariance between dynamic factor parameters and natural covariance elements.

For the scalar seasonal random walk with innovation variance `q` and `n`
identified transitions, the closed-form asymptotic standard error is

\[
\operatorname{se}(\hat q)=\hat q\sqrt{2/n}.
\]

This identity is part of the independent test suite.

## Missing observations

Each candidate likelihood uses the original observation mask. Missing locations
are excluded from that measurement update, and a fully missing time row performs
prediction only. Missing values are never filled before curvature evaluation.

Because missingness can delay diffuse-rank resolution and reduce information,
users should inspect Hessian rank and condition diagnostics rather than assuming
the complete-data information formula applies.

## Ordinary reduction

When all seasonal AR, integration, and MA orders are zero, the seasonal
estimator uses the ordinary exact-integrated state construction. Version 0.0.29
tests require the resulting parameter estimates, Hessian, and covariance to
agree with `ExactDiffuseKalmanSTARIMA` inference for the same fitted model.

## Example

```python
import numpy as np
from pystarmax import SeasonalExactDiffuseKalmanSTARIMA, SpatialWeights

weights = SpatialWeights(
    matrices=(np.eye(1),),
    names=("W0",),
)

model = SeasonalExactDiffuseKalmanSTARIMA(
    ar_order=0,
    integration_order=0,
    ma_order=0,
    seasonal_ar_order=0,
    seasonal_integration_order=1,
    seasonal_ma_order=0,
    seasonal_period=12,
    covariance_type="scalar",
    include_intercept=True,
)
model.fit(level_data, weights)

inference = model.likelihood_inference()
print(inference.summary())

natural = inference.innovation_covariance_inference()
print(natural.summary())
```

## Likelihood-scope warning

Do not compare this Hessian or covariance directly with inference from
`SeasonalKalmanSTARIMA` unless the models use the same observation likelihood.
The latter conditions on transformed-data history; the former evaluates original
levels with exact diffuse initialization.

## Deliberate limitations

Version 0.0.29 does not provide:

- analytic score or Hessian recursions;
- robust or sandwich covariance;
- profile-likelihood intervals;
- parameter-aware forecast or smoother uncertainty;
- seasonal exact-diffuse forecast intervals;
- conditional simulation smoothing;
- diffuse lag-one state covariance;
- sparse or parallel curvature evaluation.

Finite-difference inference can be expensive because a `k`-parameter Hessian
requires

\[
1+2k+4\binom{k}{2}
\]

objective evaluations. Each evaluation rebuilds and filters the complete
seasonal exact-diffuse model. The dense implementation is therefore a transparent
moderate-size reference rather than a large-parameter scalability claim.
