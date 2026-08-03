# Likelihood-Hessian inference

Version 0.0.10 introduced observed-likelihood curvature diagnostics for a fitted
`KalmanSTARMA` model. Version 0.0.11 aligned every finite-difference evaluation
with both the AR stationarity and MA invertibility criteria used during fitting.
Version 0.0.12 added a separate analytic delta-method layer for natural
innovation variance and covariance elements.

## Basic use

```python
from pystarmax import KalmanSTARMA

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
    include_intercept=True,
)
model.fit(data, weights)
inference = model.infer()

print(inference.summary())
print(inference.coefficient_table)
print(inference.optimizer_table)
print(inference.confidence_intervals(level=0.95))
print(inference.stability_boundary_distance)
print(inference.invertibility_boundary_distance)
```

`model.infer()` reuses the fitted observation matrix, spatial weights,
initialization policy, covariance parameterization, stationarity and
invertibility margins, enforcement flags, and optimizer point.

## Finite-difference convention

For a raw optimizer vector \(x\), parameter \(i\) receives

\[
h_i=\max\{h_{\mathrm{abs}},
           h_{\mathrm{rel}}\max(1,|x_i|)\}.
\]

The gradient uses a central two-sided difference. Diagonal Hessian elements use
the three-point stencil. Mixed elements use

\[
\frac{
 f(x+h_i e_i+h_j e_j)
-f(x+h_i e_i-h_j e_j)
-f(x-h_i e_i+h_j e_j)
+f(x-h_i e_i-h_j e_j)
}{4h_i h_j}.
\]

`FiniteDifferenceCurvature` records the optimizer point, step vector, objective
value, gradient, Hessian, and total function evaluations. For \(k\) parameters,
the full stencil evaluates the objective

\[
1+2k+2k(k-1)
\]

times.

Low-level functions are public:

```python
from pystarmax import finite_difference_curvature, finite_difference_hessian

curvature = finite_difference_curvature(objective, point)
hessian = finite_difference_hessian(objective, point)
```

## Observed information

The Hessian of the negative Gaussian Kalman log likelihood is interpreted as the
observed-information matrix. If it is positive definite and numerically full
rank, its inverse is used as the asymptotic covariance matrix.

`LikelihoodInferenceResult` stores:

- raw optimizer parameter names and estimates;
- the observed-information Hessian;
- optimizer-scale covariance and correlation matrices;
- standard errors, normal z statistics, and two-sided p values;
- coefficient confidence intervals;
- Hessian eigenvalues, numerical rank, and condition number;
- the finite-difference gradient and maximum absolute score;
- step sizes and function-evaluation count;
- signed distances from the AR and inverse-MA feasibility boundaries;
- the minimum signed distance to either boundary;
- covariance type and fitted number of locations.

All numerical arrays are defensive, contiguous, and read-only.

## Parameter scale

The observed-information result is on the raw optimizer scale.

- intercept, AR, and MA entries are already natural coefficients;
- scalar and diagonal covariance entries are log standard deviations;
- full covariance entries are log Cholesky diagonals and unrestricted
  strict-lower Cholesky elements.

Therefore `optimizer_table` covariance standard errors are not standard errors
of covariance matrix elements. Use the analytic natural-scale layer:

```python
natural = inference.innovation_covariance_inference()

print(natural.element_table)
print(natural.standard_error_matrix)
print(natural.dynamic_cross_covariance)
```

`coefficient_table` and `confidence_intervals()` remain restricted to the
intercept and dynamic AR/MA coefficients. See
[Innovation covariance inference](covariance_inference.md) for the covariance
Jacobian and statistical policy.

## Rank and definiteness policy

Inference fails by default if the observed-information Hessian is indefinite or
numerically rank deficient:

```python
inference = model.infer()
```

For explicit diagnosis, a positive-eigenspace pseudoinverse is available:

```python
inference = model.infer(allow_singular=True)
print(inference.used_pseudoinverse)
print(inference.rank)
```

Non-positive and numerically negligible eigen-directions are excluded. The
result is marked `used_pseudoinverse=True`; it must not be presented as ordinary
full-rank likelihood covariance.

## Admissibility boundaries

Fitting uses explicit AR and inverse-MA spectral-radius feasibility regions.
The finite-difference objective reconstructs the same criteria. A stencil point
inside an enabled artificial penalty region is rejected rather than treated as
likelihood curvature.

The signed distances are

\[
d_{AR}=1-\text{stability margin}-\rho(C_{AR}),
\]

\[
d_{MA}=1-\text{invertibility margin}-\rho(C_{MA}^{-1}).
\]

`minimum_admissibility_distance` is the smaller distance. A small positive value
warns that local curvature may be sensitive to an admissibility boundary.
Reducing finite-difference steps can avoid an artificial crossing, but it cannot
repair genuinely boundary-adjacent estimation or weak identification.

See [Stationarity and invertibility](admissibility.md).

## Step-size controls

```python
inference = model.infer(
    relative_step=2e-4,
    absolute_step=1e-6,
    rcond=1e-10,
)
```

- `relative_step` supplies scale-proportional perturbations;
- `absolute_step` prevents zero or tiny parameters from receiving a vanishing
  perturbation;
- `rcond` determines numerical Hessian rank and positive-definiteness.

Step-size sensitivity should be checked when the Hessian condition number is
large, the score is not close to zero, the fitted model is near either
admissibility boundary, or covariance parameters are weakly identified.

## Natural covariance inference

For raw covariance parameters \(\eta\), natural elements \(g(\eta)\), analytic
Jacobian \(J\), and optimizer covariance \(V\), version 0.0.12 uses

\[
\operatorname{Var}\{g(\hat\eta)\}
\approx J V_{\eta\eta}J^\top.
\]

Cross covariance with the dynamic coefficient block \(\beta\) is retained:

\[
\operatorname{Cov}\{\hat\beta,g(\hat\eta)\}
\approx V_{\beta\eta}J^\top.
\]

The transformation covers scalar, diagonal, and full Cholesky covariance
parameterizations. It reports only free natural covariance elements and does not
duplicate a shared scalar variance by location.

## Statistical scope through 0.0.13

Included:

- central finite-difference score and Hessian;
- observed-information covariance;
- dynamic coefficient standard errors, normal tests, and intervals;
- analytic natural-scale scalar, diagonal, and full-Cholesky covariance
  transformations;
- dynamic/covariance-element cross covariance;
- rank, definiteness, condition, score, and dual-boundary diagnostics;
- explicit positive-eigenspace pseudoinverse status;
- complete and partially missing stationary observations;
- rejection of AR and inverse-MA penalty points.

Not included:

- delta-method inference for correlations, spectral radii, or polynomial roots;
- robust or sandwich covariance;
- profile-likelihood or likelihood-ratio intervals;
- bootstrap likelihood inference for `KalmanSTARMA`;
- smooth stationarity/invertibility parameterization;
- exact diffuse likelihood;
- integrated or multiplicative seasonal Kalman MLE;
- sparse or automatic-differentiation Hessians;
- parameter-uncertainty propagation into filtered or smoothed states.

A numerically invertible Hessian does not by itself establish Gaussian
correctness, model adequacy, strong identification, or a safe distance from the
admissibility boundaries.
