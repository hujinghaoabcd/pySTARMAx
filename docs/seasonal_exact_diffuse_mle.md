# Seasonal exact diffuse maximum likelihood

Version 0.0.27 adds optimizer-facing maximum likelihood for multiplicative
seasonal STARIMA models on the **original observation scale**. The estimator
combines the factorized seasonal parameterization already used by
`SeasonalKalmanSTARIMA` with the original-level exact diffuse state introduced
in version 0.0.26.

## Model

Let

\[
x_t=(1-B)^d(1-B^s)^D y_t.
\]

The stationary transformed process follows the multiplicative matrix-polynomial
model

\[
\Phi_s(B^s)\Phi(B)x_t
= c + \Theta_s(B^s)\Theta(B)\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal N(0,\Sigma).
\]

The package convention multiplies each seasonal factor on the **left** of its
ordinary factor. For the autoregressive side,

\[
(I-S(B^s))(I-A(B))
\]

therefore produces ordinary terms, seasonal terms, and cross terms of the form
\(-S_jA_i\) at lag \(js+i\). The corresponding moving-average cross terms are
\(+N_jM_i\).

The cross-lag matrices are derived quantities. They are not projected back onto
the supplied spatial-weight basis and are not optimized as independent
parameters.

## Exact diffuse original-level likelihood

For each optimizer candidate, pySTARMAx performs the following operations:

1. decode the ordinary and seasonal AR/MA factor coefficients;
2. expand the ordered multiplicative matrix polynomials;
3. build the stationary transformed state-space model;
4. construct the original-level state for
   \((1-B)^d(1-B^s)^D\);
5. initialize all integration directions with exact diffuse covariance;
6. run the exact diffuse filter on the original observations;
7. return the negative original-level log likelihood to the optimizer.

No finite large-variance approximation is used. The transformed state receives
its stationary finite distribution, while the integration coordinates remain
exact diffuse directions until the observations identify them.

## Estimator

```python
from pystarmax import SeasonalExactDiffuseKalmanSTARIMA

model = SeasonalExactDiffuseKalmanSTARIMA(
    ar_order=1,
    integration_order=0,
    ma_order=0,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=0,
    seasonal_period=12,
    covariance_type="scalar",
)
result = model.fit(data, weights)
```

The ordinary order is `(p, d, q)` and the seasonal order is `(P, D, Q, s)`.
The covariance choices are:

- `"scalar"`: one innovation variance;
- `"diagonal"`: one variance per location;
- `"full"`: a positive-definite covariance represented internally by a
  Cholesky parameterization.

## Parameter counting

The optimizer dimension is

\[
\mathbb 1_{\{c\}}
+(p+P+q+Q)L+k_\Sigma,
\]

where \(L\) is the number of supplied spatial-weight matrices and
\(k_\Sigma\) is the covariance-codec dimension. Expanded cross-lag matrices do
not add free parameters.

Consequently,

\[
\mathrm{AIC}=-2\ell+2k,
\qquad
\mathrm{BIC}=-2\ell+k\log N,
\]

use the factor-parameter count and the number of observed original-level cells
reported by the exact diffuse filter.

## Admissibility

The estimator expands the full multiplicative operators before checking:

- the spectral radius of the expanded AR companion;
- the spectral radius of the positive-sign inverse-MA companion.

Candidates outside the requested stability or invertibility margin receive a
smooth quadratic penalty. The final candidate is rejected if it remains outside
the admissible region.

```python
admissibility = model.admissibility()
print(admissibility.summary())
```

## Results and state spaces

The immutable result exposes both factor parameters and expanded operators:

```python
result.ar_parameters
result.seasonal_ar_parameters
result.ma_parameters
result.seasonal_ma_parameters
result.ar_lags
result.ar_matrices
result.ma_lags
result.ma_matrices
result.innovation_covariance
```

It also retains the exact objects used at the optimum:

```python
result.transformed_state_space
result.integrated_state_space
result.filter_result
```

The fitted estimator offers:

```python
model.to_transformed_state_space()
model.to_state_space()
model.filter()
model.predict_differenced(steps=6)
model.predict(steps=6)
model.fitted_original()
```

`predict()` propagates the complete original-level augmented state. It does not
reconstruct levels from an external terminal-history object. This allows the
same route to support missing values in the fitting sample after the diffuse
phase has been resolved.

## Missing observations

`NaN` observations are skipped by the exact diffuse filter. They do not receive
silent imputations and they do not artificially reduce the diffuse rank. Missing
initial observations can therefore delay the end of the diffuse phase.

At least one finite value must remain in every transformed-data column used to
construct optimizer starting values.

## Likelihood scope is not interchangeable

`SeasonalExactDiffuseKalmanSTARIMA` and `SeasonalKalmanSTARIMA` estimate related
factorized dynamics, but their likelihoods condition on different information:

- `SeasonalExactDiffuseKalmanSTARIMA` evaluates the original levels and includes
  exact diffuse contributions for unidentified integration directions;
- `SeasonalKalmanSTARIMA` removes the differencing history and evaluates a
  conditional transformed-data likelihood.

Their log likelihoods, AIC values, and BIC values must not be mixed in one model
selection table unless every candidate uses the same likelihood convention.

## Exact reductions

Two reductions are intentional validation contracts:

1. when `seasonal_integration_order=0` and the seasonal AR/MA orders are zero,
   the estimator reduces to the ordinary exact diffuse MLE;
2. when all integration orders are zero, the exact diffuse state contains no
   diffuse directions and reduces to stationary initialization of the expanded
   seasonal state.

## Current boundaries

Version 0.0.27 does not yet expose seasonal exact diffuse:

- fixed-interval smoothing;
- primitive disturbance smoothing;
- observed-information likelihood inference;
- Gaussian forecast intervals;
- simulation smoothing;
- parameter-uncertainty propagation;
- sparse or parallel execution.

Those operations should reuse the fitted transformed and original-level state
objects delivered by this estimator rather than introduce a second seasonal
state convention.
