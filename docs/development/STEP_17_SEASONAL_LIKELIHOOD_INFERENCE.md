# Step 17 handoff: seasonal likelihood-Hessian inference

## Repository position

- development version: `0.0.17`;
- branch: `agent/seasonal-likelihood-inference`;
- pull request: PR #17, `Add seasonal likelihood-Hessian inference`;
- base: version 0.0.16 on `main`;
- final validation results must be recorded before merge.

## Delivered API

```python
inference = fitted_seasonal_model.infer(
    relative_step=1e-4,
    absolute_step=1e-6,
    rcond=1e-10,
    allow_singular=False,
)
```

Low-level equivalent:

```python
inference = infer_seasonal_kalman_starima(fitted_seasonal_model)
```

The result is the existing immutable `LikelihoodInferenceResult` rather than a
seasonal-specific duplicate.

## Optimizer coordinates

The first block contains the optional intercept and ordinary/seasonal AR/MA
factor parameters. Cross-lag matrices induced by the multiplicative expansion
are not independent coordinates.

The second block contains innovation-covariance optimizer coordinates from the
shared scalar, diagonal, or full Cholesky codec.

`n_dynamic_params` equals the factor-coordinate count. This split permits the
existing natural covariance delta-method transformation and preserves factor—
covariance cross uncertainty.

## Objective reconstruction

Every finite-difference objective call:

1. splits factor and covariance optimizer coordinates;
2. expands the complete ordered multiplicative AR and positive-sign MA matrix
   polynomials;
3. aggregates equal lags without spatial-basis projection;
4. constructs the arbitrary-lag companion state space;
5. evaluates expanded AR and inverse-MA spectral radii;
6. evaluates the transformed Gaussian Kalman likelihood when admissible.

This avoids approximating seasonal curvature with a different recursion from
the fitted model.

## Boundary rule

A stencil point outside an enabled expanded stationarity or invertibility
region returns the model feasibility penalty. The generic curvature engine uses
an invalid-objective threshold and raises instead of differentiating the
penalty.

This protects the observed information from artificial curvature at hard
feasibility boundaries. The rule applies to ordinary factors, seasonal factors,
and their induced matrix cross terms jointly.

## Finite-difference rule

The implementation reuses central score differences, diagonal second
differences, and four-corner mixed differences from the stationary inference
engine. Steps are parameter scaled:

\[
h_i=\max\left(a,r\max(1,|\vartheta_i|)\right).
\]

For `k` optimizer coordinates the full calculation uses `1 + 2*k**2` objective
evaluations. Large seasonal/full-covariance models can therefore be costly.

## Hessian policy

- symmetrize the numerical Hessian;
- use an eigendecomposition for rank and definiteness diagnostics;
- require positive definite full-rank observed information by default;
- raise on indefinite or rank-deficient curvature;
- permit an explicit positive-eigenspace generalized inverse only with
  `allow_singular=True`;
- mark that result with `used_pseudoinverse=True`.

The diagnostic pseudoinverse must not be presented as ordinary full-rank
inferential covariance.

## Result fields

The shared result provides:

- optimizer estimates and names;
- factor-only and complete optimizer tables;
- Hessian and optimizer covariance;
- standard errors, normal statistics, p-values, and confidence intervals;
- correlation matrix;
- score, finite-difference steps, objective value, and evaluation count;
- eigenvalues, rank, condition number, and positive-definite status;
- pseudoinverse status;
- distances to expanded AR and inverse-MA feasibility limits.

## Natural covariance inference

```python
natural = inference.innovation_covariance_inference()
```

The same analytic scalar, diagonal, and full-Cholesky transformations used by
stationary Kalman STARMA apply because the covariance optimizer codec is shared.
`dynamic_cross_covariance` now relates ordinary/seasonal factor coordinates to
natural innovation variance and covariance elements.

The public result accessors are `natural.parameter_names`, `natural.estimates`,
`natural.table`, and `natural.confidence_intervals()`. The lower-level
`element_names` and `element_table` metadata belong to `natural.transform`; code
and examples should not treat them as direct `InnovationCovarianceInference`
attributes.

## Validation references

Tests include:

1. exact zero-seasonal-order inference equivalence with stationary
   `KalmanSTARMA` under identical data, starts, and finite-difference settings;
2. full-rank pure seasonal AR factor curvature;
3. equality between `.infer()` and `infer_seasonal_kalman_starima()`;
4. natural scalar innovation-variance inference and factor/covariance cross
   covariance;
5. direct expanded non-stationary objective penalty verification;
6. strict singular-Hessian rejection and explicit finite positive-eigenspace
   generalized inverse;
7. fitted-state, rank-threshold, and step validation;
8. the complete inherited package test suite.

## Files introduced or changed

Core:

- `src/pystarmax/seasonal_likelihood_inference.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_seasonal_likelihood_inference.py`.

Documentation and example:

- `docs/seasonal_likelihood_inference.md`;
- `examples/seasonal_likelihood_inference.py`;
- README, documentation home, navigation, seasonal Kalman guide, roadmap, and
  project status;
- Step 17 handoff.

## Current limitations

- central finite differences scale quadratically with optimizer dimension;
- step sensitivity remains possible near flat or highly curved likelihood
  directions;
- conditional transformation history is treated as fixed;
- no robust, profile-likelihood, likelihood-ratio, sandwich, or bootstrap
  seasonal Kalman inference;
- feasibility penalties are not smooth parameterizations;
- original-scale intervals and parameter uncertainty in smoothing are not yet
  propagated;
- dense seasonal state spaces make each objective evaluation expensive.

## Next recommended stage

The next product-facing stage is original-scale Gaussian forecast intervals for
ordinary and seasonal Kalman STARIMA, with future paths simulated on the
transformed scale and inverse-differenced pathwise before quantiles. A deeper
method stage is exact diffuse integrated level-state likelihood and smoothing as
a separate API.
