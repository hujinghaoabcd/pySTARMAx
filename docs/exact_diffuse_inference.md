# Exact diffuse likelihood inference

Version 0.0.23 adds observed-information inference for fitted
`ExactDiffuseKalmanSTARIMA` models.

The inference objective is the same original-level exact diffuse likelihood used
for estimation. It is not the conditional likelihood of a differenced series,
and it does not replace diffuse state uncertainty with an arbitrary large
variance.

## Public API

```python
inference = fitted_model.likelihood_inference()
print(inference.summary())
```

The low-level entry point is:

```python
from pystarmax import infer_exact_diffuse_kalman_starima

inference = infer_exact_diffuse_kalman_starima(fitted_model)
```

Important numerical controls are:

```python
inference = fitted_model.likelihood_inference(
    relative_step=1e-4,
    absolute_step=1e-6,
    rcond=1e-10,
    allow_singular=False,
)
```

## Objective reconstruction

Let the unconstrained optimizer coordinate vector be

\[
\vartheta=(\theta^\top,\lambda^\top)^\top,
\]

where \(\theta\) contains the intercept and STARMA coefficients and
\(\lambda\) contains the innovation-covariance coordinates.

For every finite-difference candidate, the implementation reconstructs:

1. the intercept, AR coefficients, and positive-sign MA coefficients;
2. the scalar, diagonal, or full-Cholesky innovation covariance;
3. the transformed stationary STARMA state space;
4. the ordinary integrated original-level state space;
5. the exact finite/diffuse initial covariance decomposition;
6. the complete exact diffuse Kalman likelihood on the observed level series.

Therefore the numerical Hessian approximates

\[
\mathcal I(\widehat\vartheta)
=
\left.
\frac{\partial^2[-\ell_D(\vartheta)]}
{\partial\vartheta\partial\vartheta^\top}
\right|_{\widehat\vartheta},
\]

where \(\ell_D\) is the exact diffuse original-level log likelihood.

## Adaptive central finite differences

The implementation reuses the package finite-difference curvature engine. For
coordinate \(j\), the step is

\[
h_j=\max(h_{\mathrm{abs}},
          h_{\mathrm{rel}}\max(1,|\widehat\vartheta_j|)).
\]

Diagonal and mixed Hessian entries use central differences. If a symmetric
candidate is inadmissible, the engine reduces the step before failing. The
result stores:

- the step used for every coordinate;
- the gradient at the fitted point;
- the symmetrized Hessian;
- the number of objective evaluations;
- the objective value at the fitted point.

For \(k\) parameters, a fully central evaluation requires
\(1+2k+2k(k-1)\) function evaluations. The two-parameter random-walk reference
therefore requires nine evaluations.

## Admissibility boundary

Every candidate uses the same transformed-dynamics rules as the fitted model:

- the AR spectral radius must remain below the fitted stability limit when
  stationarity enforcement is enabled;
- the package positive-sign inverse-MA spectral radius must remain below the
  fitted invertibility limit when invertibility enforcement is enabled;
- invalid candidates receive the same large smooth boundary penalty convention
  used by the optimizer-facing model.

The inference result records the fitted AR and inverse-MA boundary distances.

## Covariance policy

The default policy requires the observed-information Hessian to be full rank
and positive definite. Then

\[
\widehat{\operatorname{Var}}(\widehat\vartheta)
=
\mathcal I(\widehat\vartheta)^{-1}.
\]

If this condition fails, the default call raises `LinAlgError` and reports the
rank and minimum eigenvalue. This prevents an unidentified direction from being
silently reported as ordinary Wald inference.

A diagnostic positive-eigenspace generalized inverse is available only through
an explicit opt-in:

```python
inference = fitted_model.likelihood_inference(allow_singular=True)
```

Such a result sets `used_pseudoinverse=True` and
`positive_definite=False`. It is a numerical diagnostic rather than evidence
that all model parameters are identified.

## Returned inference

The method returns the existing immutable `LikelihoodInferenceResult`, which
contains:

- optimizer-coordinate parameter names and estimates;
- Hessian and covariance matrices;
- standard errors, z statistics, two-sided normal p values, and correlations;
- Hessian eigenvalues, rank, condition number, and positive-definite status;
- numerical gradient, finite-difference steps, and function-evaluation count;
- stability and invertibility boundary distances.

Confidence intervals and individual Wald tests are available through the result
methods already used by stationary and seasonal likelihood inference.

## Natural innovation covariance inference

The optimizer covariance coordinates are not the natural entries of
\(\Sigma_\eta\). The result can transform them with the existing analytic
Jacobian:

```python
natural = inference.innovation_covariance_inference()
print(natural.estimates)
print(natural.standard_errors)
print(natural.dynamic_cross_covariance)
```

For transformation \(g(\vartheta)\), the first-order delta method is

\[
\operatorname{Var}[g(\widehat\vartheta)]
\approx
J_g
\widehat{\operatorname{Var}}(\widehat\vartheta)
J_g^\top.
\]

The implementation supports scalar, diagonal, and full-Cholesky covariance
parameterizations and retains cross covariance between dynamic coefficients and
natural covariance entries.

## Scalar random-walk reference

For

\[
y_t=y_{t-1}+\mu+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,\sigma^2),
\]

suppose \(n\) increments contribute to the Gaussian part of the exact diffuse
likelihood. In coordinates \((\mu,\log\sigma)\), the observed information at
the MLE is

\[
\mathcal I=
\begin{bmatrix}
 n/\widehat\sigma^2 & 0\\
 0 & 2n
\end{bmatrix}.
\]

Consequently,

\[
\operatorname{SE}(\widehat\mu)
=
\sqrt{\widehat\sigma^2/n},
\]

and the delta-method standard error of the natural variance is

\[
\operatorname{SE}(\widehat\sigma^2)
=
\widehat\sigma^2\sqrt{2/n}.
\]

The test suite compares the numerical Hessian, raw-coordinate covariance, and
natural variance standard error with these expressions.

## Missing observations

Missing cells remain `NaN` throughout candidate reconstruction. Each objective
evaluation applies the exact diffuse filter's existing rules:

- missing locations perform no measurement update;
- fully missing rows contribute no observation likelihood but propagate the
  state;
- missing initial levels delay diffuse-rank reduction;
- no imputation occurs before curvature evaluation.

Inference therefore reflects the actual incomplete original-level likelihood.

## Validation references

Tests cover:

1. the scalar random-walk closed-form Hessian;
2. the closed-form optimizer-coordinate covariance;
3. the natural innovation variance delta-method standard error;
4. equality of the curvature objective and the fitted exact diffuse objective;
5. missing-data finite inference and immutable arrays;
6. strict rejection of a singular Hessian;
7. explicit positive-eigenspace generalized inverse diagnostics;
8. fit, finite-difference-step, and `rcond` validation;
9. the complete inherited package suite.

Core CI #497 reported 197 passing tests, 87.27% total branch coverage, and
88.8% coverage for `exact_diffuse_inference.py`.

## Scope and limitations

Available in 0.0.23:

- observed-information Hessian for ordinary exact diffuse STARIMA MLE;
- raw optimizer-coordinate covariance and Wald summaries;
- scalar, diagonal, and full-Cholesky natural innovation covariance delta
  inference;
- dynamic/natural-covariance cross uncertainty;
- missing-data-aware objective reconstruction;
- strict and explicitly diagnostic singular-Hessian policies.

Not yet available:

- analytic score or Hessian recursions;
- exact diffuse robust or sandwich covariance;
- parameter uncertainty propagated into forecasts, state smoothing, or
  disturbance smoothing;
- seasonal exact diffuse inference;
- simulation-based likelihood inference;
- automatic profile-likelihood intervals;
- sparse or parameter-aware derivative paths.
