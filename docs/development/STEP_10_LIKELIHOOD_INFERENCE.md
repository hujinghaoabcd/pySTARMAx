# Step 10 handoff: likelihood-curvature inference

Updated: 2026-08-04

## Objective

Add defensible local asymptotic inference to the stationary Gaussian
`KalmanSTARMA` estimator without changing the existing conditional estimators or
pretending that a numerically problematic likelihood Hessian is reliable.

The completed implementation evaluates the negative Gaussian Kalman log
likelihood around the fitted raw optimizer vector. It exposes the score, observed
Hessian, inverse observed information, coefficient uncertainty, and explicit
curvature diagnostics.

## Public API added

The following symbols are exported from `pystarmax`:

- `FiniteDifferenceCurvature`;
- `LikelihoodInferenceResult`;
- `finite_difference_curvature()`;
- `finite_difference_hessian()`.

A fitted `KalmanSTARMA` model now provides:

```python
inference = model.infer(
    relative_step=1e-4,
    absolute_step=1e-6,
    rcond=1e-10,
    allow_singular=False,
)
```

`model.infer()` requires a successful prior `fit()` call and reuses the fitted
training observations, spatial weights, covariance parameterization,
initialization policy, stability margin, and raw optimizer point.

## Numerical convention

For raw optimizer parameter `x_i`, the perturbation is

```text
h_i = max(absolute_step, relative_step * max(1, abs(x_i)))
```

The implementation uses:

- a two-sided central score;
- a three-point diagonal second derivative;
- a four-corner central mixed derivative;
- explicit Hessian symmetrization;
- function-evaluation accounting;
- rejection of non-finite objective values;
- rejection of values at or above the stationarity-penalty threshold.

For `k` parameters, the full stencil uses

```text
1 + 2k + 2k(k - 1)
```

objective evaluations.

## Observed-information policy

The Hessian of the negative log likelihood is interpreted as observed
information only when the eigenvalue test indicates positive-definite, full-rank
curvature.

The numerical threshold is

```text
threshold = rcond * max(1, max(abs(eigenvalues)))
```

The result reports:

- raw optimizer parameter names and estimates;
- Hessian and inverse observed-information covariance;
- standard errors, z statistics, and two-sided normal p values;
- parameter correlation matrix;
- normal-approximation confidence intervals for intercept and AR/MA terms;
- eigenvalues, numerical rank, and condition number;
- score vector and maximum absolute score;
- finite-difference step vector and evaluation count;
- distance to the transition spectral-radius feasibility boundary;
- whether a pseudoinverse was used.

## Failure and pseudoinverse behavior

Default behavior is deliberately strict:

- an indefinite Hessian raises `numpy.linalg.LinAlgError`;
- a numerically rank-deficient Hessian raises the same error;
- a stencil entering the stationarity penalty raises `ValueError`;
- invalid step sizes and `rcond` values raise `ValueError`.

`allow_singular=True` is an explicit diagnostic route. It retains only strictly
positive eigen-directions above the numerical threshold and sets excluded inverse
eigenvalues to zero. The result is marked with `used_pseudoinverse=True`.

Zero-standard-error directions produce finite diagnostic values rather than
`NaN` or `Inf`. The positive-eigenspace inverse uses masked `numpy.divide`, so
excluded zero or negative eigenvalues are never evaluated as denominators. This
design avoids both non-finite results and divide-by-zero warnings, but the
pseudoinverse route must not be presented as ordinary full-rank maximum-
likelihood inference.

## Parameter scale

Inference is on the raw optimizer scale.

- intercept and AR/MA coefficients are already natural coefficients;
- scalar and diagonal covariance parameters are log standard deviations;
- full covariance parameters are log Cholesky diagonals plus unconstrained lower
  Cholesky entries.

Therefore:

- `coefficient_table` contains only intercept and dynamic coefficients;
- `confidence_intervals()` is restricted to those dynamic terms;
- `optimizer_table` includes covariance-factor parameters but does not transform
  them to variance, covariance, or correlation elements;
- no delta-method covariance transformation is claimed in 0.0.10.

## Main implementation files

- `src/pystarmax/likelihood_inference.py` — finite differences, result object,
  likelihood reconstruction, and observed-information inversion;
- `src/pystarmax/_maximum_likelihood_model.py` — fitted-model `infer()` method;
- `src/pystarmax/__init__.py` — public exports;
- `tests/test_likelihood_inference.py` — analytic and statistical reference tests;
- `tests/test_likelihood_inference_singular.py` — singular-Hessian policy tests;
- `examples/likelihood_inference.py` — complete fitted-model example;
- `docs/likelihood_inference.md` — statistical and user-facing method guide.

## Validation references

Tests cover:

1. an analytic quadratic objective with known gradient and Hessian;
2. exact function-evaluation count for the central stencil;
3. finite-difference input validation and invalid-region rejection;
4. scalar Gaussian white-noise MLE standard errors against closed-form results;
5. incomplete-observation AR(1) curvature, rank, condition, and boundary distance;
6. fit-state and control validation;
7. default failure on rank-deficient observed information;
8. explicit positive-eigenspace pseudoinverse behavior;
9. finite arrays and explicit `used_pseudoinverse` marking.

Final implementation and documentation validation was completed by GitHub Actions
CI run #237:

- 91 tests passed without Python test warnings;
- total branch coverage was 87.84%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic reference regeneration produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11 through 3.14;
- the singular-Hessian pseudoinverse path produced no divide-by-zero warning;
- the PR contained only the 14 formal implementation, test, example,
  documentation, metadata, and navigation files.

The final validation-record edits are documentation only and require one final
CI repeat before merge. Numerical code, tests, exports, package metadata, and CI
configuration are unchanged from run #237.

## Deliberate exclusions

0.0.10 does not include:

- robust or sandwich covariance;
- delta-method covariance-element uncertainty;
- profile likelihood or likelihood-ratio intervals;
- automatic-differentiation Hessians;
- exact diffuse likelihood;
- MA invertibility constraints;
- smooth stationarity reparameterization;
- integrated or multiplicative seasonal MLE inference;
- state or disturbance smoothing;
- sparse state-space computation.

## Next recommended stage

The next stage should address stability and invertibility before adding broader
model families.

Recommended sequence:

1. implement reusable AR transition stability diagnostics independent of fitting;
2. define and test a multivariate/spatial MA invertibility criterion;
3. expose fitted-result stability and invertibility reports;
4. add explicit constrained optimization behavior;
5. evaluate smooth parameterizations that keep finite-difference stencils inside
   the admissible region;
6. test near-boundary simulations and failure messages;
7. only then add delta-method covariance transforms and seasonal/integrated MLE
   wrappers.

## Handoff checklist

Before starting Step 11:

- read `PROJECT_STATUS.md`;
- read `docs/model.md`, `docs/state_space.md`, `docs/maximum_likelihood.md`, and
  `docs/likelihood_inference.md`;
- inspect `KalmanSTARMA._objective()` and `infer_kalman_starma()` together;
- preserve separation between conditional and maximum-likelihood estimators;
- do not differentiate through the large feasibility penalty;
- retain explicit raw-parameter-scale wording;
- add independent numerical tests before changing admissibility constraints.
