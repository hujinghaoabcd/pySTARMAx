# Innovation covariance delta-method inference

Version 0.0.12 adds natural-scale uncertainty for the innovation covariance of a
fitted `KalmanSTARMA` model. This layer is deliberately separate from the raw
optimizer-scale observed-information result returned by `KalmanSTARMA.infer()`.

## Why a transformation is required

The Gaussian optimizer does not estimate covariance elements directly. It uses
unconstrained parameters that guarantee a valid covariance matrix:

- `scalar`: one log standard deviation;
- `diagonal`: one log standard deviation per location;
- `full`: log diagonal entries and unrestricted strict-lower entries of a
  Cholesky factor.

Let the raw covariance parameters be \(\eta\), and let \(g(\eta)\) denote the
free natural variance or covariance elements. The first-order delta method uses

\[
\operatorname{Var}\{g(\hat\eta)\}
\approx
J\,\operatorname{Var}(\hat\eta)\,J^\top,
\qquad
J = \left.\frac{\partial g}{\partial \eta^\top}\right|_{\hat\eta}.
\]

The implementation also propagates the cross covariance between dynamic
coefficients and covariance parameters. If \(\beta\) denotes the intercept,
AR, and MA block, then

\[
\operatorname{Cov}\{\hat\beta,g(\hat\eta)\}
\approx
\operatorname{Cov}(\hat\beta,\hat\eta)J^\top.
\]

## Scalar covariance

For a shared raw log standard deviation \(\eta\),

\[
\sigma = \exp(\eta),
\qquad
\sigma^2 = \exp(2\eta),
\qquad
\frac{\partial\sigma^2}{\partial\eta}=2\sigma^2.
\]

Only one shared variance is reported. The result does not duplicate the same
parameter once per location.

## Diagonal covariance

For location \(i\),

\[
\sigma_i^2 = \exp(2\eta_i).
\]

The Jacobian is diagonal with entry \(2\sigma_i^2\). The natural result reports
one variance per location.

## Full Cholesky covariance

The optimizer constructs a lower-triangular matrix \(L\). Its diagonal entries
are exponentiated and its strict-lower entries are unrestricted. The innovation
covariance is

\[
\Sigma = LL^\top.
\]

For a perturbation \(dL\),

\[
d\Sigma = dL\,L^\top + L\,dL^\top.
\]

For a raw log-diagonal parameter, the corresponding differential of \(L_{ii}\)
is \(L_{ii}\,d\eta_i\). For a strict-lower parameter, the differential is one
in its Cholesky position. pySTARMAx evaluates this analytic Jacobian and reports
the unique lower-triangle elements of \(\Sigma\) in row-major lower-triangle
order.

## Usage

```python
from pystarmax import KalmanSTARMA

model = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
)
model.fit(observations, weights)

raw_inference = model.infer(relative_step=1e-4)
natural = raw_inference.innovation_covariance_inference()

print(natural.element_table)
print(natural.covariance_matrix)
print(natural.standard_error_matrix)
print(natural.confidence_intervals(level=0.95))
print(natural.dynamic_cross_covariance)
```

The lower-triangle standard errors are also expanded into a symmetric
matrix-shaped view for convenient reporting. The underlying inferential vector
still contains only the free natural covariance elements.

## Statistical policy

The returned intervals are first-order, unbounded normal approximations. A
negative lower endpoint for a variance is not silently clipped: clipping would
change the stated approximation and conceal weak curvature or small-sample
behavior.

An ordinary Wald test of `variance = 0` is not reported because zero variance is
a boundary null under the positive covariance parameterization. Profile
likelihood, likelihood-ratio mixtures, bootstrap inference, and bounded
transformations remain separate future work.

If the source likelihood inference used an explicit positive-eigenspace
pseudoinverse, the natural result preserves that diagnostic status. Delta-method
output from a singular observed-information matrix should be treated as
exploratory rather than routine publication-quality inference.

## Public objects

- `innovation_covariance_transform()` returns the natural elements and analytic
  Jacobian for supplied raw covariance parameters;
- `delta_method_covariance()` applies a general Jacobian covariance propagation;
- `innovation_covariance_delta_inference()` transforms a
  `LikelihoodInferenceResult`;
- `LikelihoodInferenceResult.innovation_covariance_inference()` is the fitted
  convenience route;
- `InnovationCovarianceTransform` and `InnovationCovarianceInference` store
  immutable read-only arrays.
