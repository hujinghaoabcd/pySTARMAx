# Seasonal likelihood-Hessian inference

Version 0.0.17 adds observed-information inference for the multiplicative factor
parameters and innovation-covariance optimizer parameters fitted by
`SeasonalKalmanSTARIMA`.

The result contract is the same immutable `LikelihoodInferenceResult` used by
the stationary Gaussian estimator. Seasonal factor estimates therefore receive
standard errors, normal-reference tests, confidence intervals, correlations,
score diagnostics, Hessian rank and eigenvalues, condition number, and natural
innovation-covariance delta-method inference without introducing a second
incompatible inference API.

## Parameter vector

For `K` spatial weights, the dynamic factor vector contains

\[
\mathbf 1_c+K(p+P+q+Q)
\]

entries in this order:

1. optional transformed-scale intercept;
2. ordinary AR factor coefficients;
3. seasonal AR factor coefficients;
4. ordinary MA factor coefficients;
5. seasonal MA factor coefficients.

The covariance optimizer block follows the existing codec:

- scalar covariance: one log standard deviation;
- diagonal covariance: one log standard deviation per location;
- full covariance: log-Cholesky diagonal entries and unrestricted lower-factor
  entries.

Multiplicative cross-lag matrices are deterministic functions of factor
parameters. They do not appear as independent Hessian coordinates.

## Objective

For each raw optimizer vector, inference reconstructs:

1. ordinary and seasonal factor matrices from the supplied spatial weights;
2. the complete ordered multiplicative AR and positive-sign MA expansion;
3. the arbitrary-lag companion state space;
4. the innovation covariance from the optimizer codec;
5. the transformed Gaussian Kalman log likelihood.

The objective is the negative conditional likelihood

\[
f(\vartheta)
=-\ell\left(
\vartheta;
(1-B)^d(1-B^s)^D y
\mid\mathcal H_{d+Ds}
\right).
\]

The inference therefore describes the conditional transformed likelihood. It is
not exact diffuse inference for an integrated seasonal level-state model.

## Central finite differences

For coordinate `i` and step `h_i`, the score approximation is

\[
\frac{\partial f}{\partial\vartheta_i}
\approx
\frac{f(\vartheta+h_i e_i)-f(\vartheta-h_i e_i)}{2h_i}.
\]

Diagonal Hessian elements use

\[
\frac{\partial^2 f}{\partial\vartheta_i^2}
\approx
\frac{f(\vartheta+h_i e_i)-2f(\vartheta)
+f(\vartheta-h_i e_i)}{h_i^2}.
\]

Mixed elements use the four-corner stencil

\[
\frac{\partial^2f}{\partial\vartheta_i\partial\vartheta_j}
\approx
\frac{
 f_{++}-f_{+-}-f_{-+}+f_{--}
}{4h_ih_j}.
\]

The step is

\[
h_i=\max\left(
\texttt{absolute\_step},
\texttt{relative\_step}\max(1,|\vartheta_i|)
\right).
\]

For `k` raw optimizer parameters, a complete central Hessian requires
`1 + 2*k**2` objective evaluations. Seasonal models can therefore be expensive
when many factor coefficients or full-covariance elements are present.

## Expanded admissibility boundary

Every finite-difference point is expanded through the complete multiplicative
matrix polynomial. The implementation then evaluates:

- the spectral radius of the complete expanded AR companion;
- the spectral radius of the inverse recursion for the complete expanded
  positive-sign MA companion.

When an enabled stencil point enters the stationarity or invertibility penalty
region, the finite-difference engine rejects the stencil. Penalty curvature is
not interpreted as statistical likelihood curvature.

This rule applies to ordinary, seasonal, and induced cross-lag behavior. It is
not sufficient to inspect each factor coefficient independently.

## Observed information

The symmetrized Hessian is

\[
H=\frac{1}{2}(\widehat H+\widehat H^\top).
\]

When `H` is positive definite and full rank, the optimizer-scale covariance is

\[
V_\vartheta=H^{-1}.
\]

Standard errors, normal-reference statistics, and correlations follow from
`V_theta`.

A non-positive-definite or rank-deficient Hessian raises by default. Setting
`allow_singular=True` explicitly retains only eigenvectors with positive
eigenvalues above the configured threshold and forms a generalized inverse.
The result records `used_pseudoinverse=True`; this path is diagnostic and should
not be reported as ordinary full-rank inference without qualification.

## Usage

```python
from pystarmax import SeasonalKalmanSTARIMA

model = SeasonalKalmanSTARIMA(
    ar_order=1,
    integration_order=0,
    ma_order=0,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=0,
    seasonal_period=12,
    covariance_type="full",
)
model.fit(level_series, weights)

inference = model.infer(
    relative_step=1e-4,
    absolute_step=1e-6,
    rcond=1e-10,
)

print(inference.coefficient_table)
print(inference.optimizer_table)
print(inference.confidence_intervals(level=0.95))
print(inference.eigenvalues)
print(inference.condition_number)
print(inference.minimum_admissibility_distance)
```

The coefficient table contains transformed-scale multiplicative factor
parameters. The optimizer table additionally includes covariance-codec
parameters.

## Natural innovation covariance inference

Because the seasonal result uses the same raw covariance codec and
`LikelihoodInferenceResult`, natural covariance inference is available directly:

```python
natural = inference.innovation_covariance_inference()

print(natural.element_table)
print(natural.covariance_matrix)
print(natural.standard_error_matrix)
print(natural.dynamic_cross_covariance)
```

For a full covariance `Sigma = L @ L.T`, the analytic Jacobian uses

\[
d\Sigma=dL\,L^\top+L\,dL^\top.
\]

`dynamic_cross_covariance` retains covariance between ordinary/seasonal factor
estimates and natural innovation-covariance elements.

Natural intervals remain first-order unbounded normal approximations. Negative
lower endpoints are not silently clipped, and an ordinary Wald test of
`variance = 0` is not supplied because zero is a boundary null.

## Diagnostics

The result includes:

- finite-difference score at the optimum;
- per-coordinate steps;
- function-evaluation count and objective value;
- Hessian eigenvalues and numerical rank;
- positive-definite status;
- condition number of the retained eigenspace;
- explicit pseudoinverse status;
- distances to the expanded AR-stationarity and MA-invertibility limits.

A small gradient does not by itself validate the Hessian. Inspect boundary
distance, eigenvalues, condition number, step sensitivity, optimizer
convergence, and substantive parameter identifiability together.

## Low-level function

```python
from pystarmax import infer_seasonal_kalman_starima

inference = infer_seasonal_kalman_starima(fitted_model)
```

The package-root `SeasonalKalmanSTARIMA` class provides `.infer()` as the normal
entry point. The low-level function is useful for explicit workflows and tests.

## Validation references

The implementation is checked through:

- zero-seasonal-order equivalence with stationary `KalmanSTARMA` inference under
  identical data, starts, and finite-difference settings;
- a fitted pure seasonal AR factor with full-rank observed information;
- equality between the fitted method and low-level function;
- natural scalar innovation-variance delta-method inference and factor/covariance
  cross covariance;
- direct verification that an expanded non-stationary point receives the
  feasibility penalty rather than a likelihood value;
- strict default rejection of a singular Hessian and finite explicit
  positive-eigenspace pseudoinverse results;
- not-fitted, step-size, and rank-threshold validation.

## Current limitations

- finite differences can be expensive and step sensitive for large factor or
  full-covariance models;
- inference remains conditional on ordinary-seasonal transformation history;
- feasibility penalties are not a smooth parameterization;
- no robust, sandwich, profile-likelihood, likelihood-ratio, or bootstrap
  seasonal Kalman inference is available;
- original-scale forecast intervals and parameter-uncertainty propagation
  through smoothing remain unavailable;
- dense seasonal companion states can make each likelihood evaluation costly.
