# Likelihood-Hessian inference

Version 0.0.10 added observed-likelihood curvature diagnostics for a fitted
`KalmanSTARMA` model. Version 0.0.11 aligns every finite-difference evaluation
with both the autoregressive stationarity and moving-average invertibility
criteria used by fitting.

The implementation evaluates the negative Gaussian Kalman log likelihood around
the optimizer solution using central finite differences. It reports coefficient
uncertainty only when the resulting observed-information matrix is numerically
defensible.

## Basic use

```python
from pystarmax import KalmanSTARMA

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=True,
)
fit = model.fit(data, weights)
inference = model.infer()

print(inference.summary())
print(inference.coefficient_table)
print(inference.confidence_intervals(level=0.95))
print(inference.stability_boundary_distance)
print(inference.invertibility_boundary_distance)
```

`model.infer()` uses the same incomplete training matrix, spatial weights,
initialization policy, covariance parameterization, AR and MA margins,
enforcement flags, and fitted optimizer point as `model.fit()`.

## Finite-difference convention

For a raw optimizer parameter vector `x`, each parameter receives a scale-aware
step

\[
h_i = \max(h_{\mathrm{abs}}, h_{\mathrm{rel}}\max(1, |x_i|)).
\]

The score uses a central two-sided difference. Diagonal Hessian terms use the
three-point stencil, while mixed terms use the four-corner central stencil:

\[
\frac{f(x+h_i e_i+h_j e_j)-f(x+h_i e_i-h_j e_j)
-f(x-h_i e_i+h_j e_j)+f(x-h_i e_i-h_j e_j)}{4h_i h_j}.
\]

The returned `FiniteDifferenceCurvature` records the optimizer point, step
vector, objective value, score, Hessian, and total function evaluations. For
`k` parameters the complete central stencil evaluates the objective
`1 + 2k + 4k(k-1)/2` times.

Low-level functions are public:

```python
from pystarmax import finite_difference_curvature, finite_difference_hessian

curvature = finite_difference_curvature(objective, point)
hessian = finite_difference_hessian(objective, point)
```

They are useful for reproducible numerical-method tests independent of STARMA.

## Observed information

The Hessian of the negative log likelihood is interpreted as observed
information. If it is positive definite and numerically full rank, its inverse
is used as the asymptotic covariance matrix.

`LikelihoodInferenceResult` contains:

- raw optimizer estimates and names;
- observed-information Hessian;
- asymptotic covariance and correlation matrices;
- standard errors, normal z statistics, and two-sided p values;
- normal-approximation confidence intervals for intercept and dynamic AR/MA
  coefficients;
- Hessian eigenvalues, numerical rank, and condition number;
- the finite-difference score and maximum absolute score component;
- finite-difference steps and function-evaluation count;
- distance from the configured AR stationarity boundary;
- distance from the configured inverse-MA invertibility boundary;
- the minimum signed distance to either admissibility boundary.

All numerical arrays are immutable.

## Parameter scale

Inference is computed on the raw optimizer scale.

- The common intercept and AR/MA coefficients are already on their natural
  coefficient scale.
- Scalar and diagonal covariance parameters are log standard deviations.
- Full-covariance parameters are log Cholesky diagonal entries and unconstrained
  lower-triangular Cholesky entries.

Therefore, covariance-parameter standard errors in `optimizer_table` are not
standard errors of covariance matrix elements. The current implementation
deliberately does not apply a delta-method transformation to variances,
correlations, or covariance entries. `coefficient_table` and
`confidence_intervals()` are restricted to the intercept and dynamic
coefficients.

## Rank and definiteness policy

By default, inference fails if the observed-information Hessian is not positive
definite and numerically full rank:

```python
inference = model.infer()  # raises on indefinite or rank-deficient curvature
```

This prevents an indefinite matrix or weakly identified parameter direction from
being silently presented as reliable standard errors.

For diagnosis only, users may explicitly request a positive-eigenvalue
pseudoinverse:

```python
inference = model.infer(allow_singular=True)
print(inference.used_pseudoinverse)
print(inference.rank)
```

The pseudoinverse sets non-positive or numerically negligible eigen-directions to
zero. The result is marked with `used_pseudoinverse=True`; it should not be
interpreted as ordinary full-rank maximum-likelihood covariance.

## Admissibility boundaries

The maximum-likelihood estimator uses explicit AR and inverse-MA spectral-radius
feasibility boundaries. Finite-difference stencils reconstruct the same two
criteria and reject an objective value that enters either enabled penalty region.
The artificial penalty surface is never treated as likelihood curvature.

The AR distance is

\[
d_{AR} = 1 - \text{stability_margin} - \rho(C_{AR}),
\]

and the MA distance is

\[
d_{MA} = 1 - \text{invertibility_margin} - \rho(C_{MA}^{-1}).
\]

`minimum_admissibility_distance` is `min(d_AR, d_MA)`. A small positive value
indicates that the central stencil may be sensitive to one of the feasibility
boundaries.

If a stencil crosses a boundary, reduce `relative_step` and `absolute_step`, or
treat local asymptotic inference as unavailable. A smaller step can avoid an
artificial crossing but cannot repair genuinely boundary-adjacent estimation or
weak identification.

See [Stationarity and invertibility](admissibility.md) for the companion-matrix
sign convention and public eigensystem diagnostics.

## Step-size controls

```python
inference = model.infer(
    relative_step=2e-4,
    absolute_step=1e-6,
    rcond=1e-10,
)
```

- `relative_step` controls scale-proportional perturbations;
- `absolute_step` prevents zero or tiny raw parameters from receiving a
  vanishing perturbation;
- `rcond` defines the eigenvalue threshold used for numerical rank and
  positive-definiteness decisions.

Step-size sensitivity should be examined when the Hessian condition number is
large, the optimizer score is not close to zero, the model is near either
admissibility boundary, or covariance parameters are weakly identified.

## Statistical scope

Included through 0.0.11:

- central finite-difference score and Hessian;
- observed-information covariance;
- dynamic coefficient standard errors and normal tests;
- rank, definiteness, condition, score, and dual-boundary diagnostics;
- explicit pseudoinverse diagnostics;
- complete and partially missing stationary observations through the fitted
  Kalman likelihood;
- rejection of AR-stationarity and MA-invertibility penalty points.

Not yet included:

- delta-method uncertainty for variance, covariance, correlation, spectral
  radius, polynomial roots, or transformed Cholesky quantities;
- robust or sandwich covariance;
- profile likelihood or likelihood-ratio confidence intervals;
- bootstrap uncertainty for `KalmanSTARMA`;
- smooth stationarity/invertibility parameterization;
- exact diffuse likelihood;
- integrated and multiplicative seasonal maximum-likelihood inference;
- sparse or automatic-differentiation Hessians.

These exclusions are explicit. A numerically invertible Hessian alone does not
establish model adequacy, Gaussian correctness, strong identification, or a safe
distance from the admissibility boundaries.
