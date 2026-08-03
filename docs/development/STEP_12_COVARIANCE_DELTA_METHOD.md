# Step 12 handoff: innovation covariance delta-method inference

## Completed scope

Version 0.0.12 added a natural-scale inferential layer for the innovation
covariance of a fitted `KalmanSTARMA` model.

Implemented objects and entry points:

- `InnovationCovarianceTransform`;
- `InnovationCovarianceInference`;
- `innovation_covariance_transform()`;
- `delta_method_covariance()`;
- `innovation_covariance_delta_inference()`;
- `LikelihoodInferenceResult.innovation_covariance_inference()`.

The transformation supports scalar, diagonal, and full Cholesky covariance
parameterizations. It preserves dynamic-coefficient/covariance-element cross
covariance and reports only statistically free natural covariance elements.

## Mathematical contract

For raw covariance parameters \(\eta\), natural covariance elements
\(g(\eta)\), analytic Jacobian \(J\), and optimizer covariance \(V\):

\[
\operatorname{Var}\{g(\hat\eta)\}\approx JV_{\eta\eta}J^\top,
\]

\[
\operatorname{Cov}\{\hat\beta,g(\hat\eta)\}
\approx V_{\beta\eta}J^\top.
\]

For a full covariance, \(\Sigma=LL^\top\) and
\(d\Sigma=dL L^\top + L dL^\top\). Log-Cholesky diagonal derivatives include
the current positive diagonal value.

## Statistical decisions

- scalar covariance reports one shared variance, not repeated location entries;
- diagonal covariance reports one variance per location;
- full covariance reports unique lower-triangle elements;
- first-order normal intervals are not clipped at zero;
- ordinary variance-equals-zero Wald tests are not reported because zero is a
  boundary null;
- a source likelihood pseudoinverse remains explicitly marked.

## Validation

The step includes scalar and diagonal analytic references, a hand-derived
2-location full-Cholesky Jacobian, a 3-location finite-difference comparison,
direct covariance propagation, cross-covariance, immutability, validation, and a
fitted scalar white-noise chain-rule standard-error test.

During Step 13 integration, two inherited defects were found and corrected:

1. `infer_kalman_starma()` used an undefined local variable when setting
   `n_locations`; it now uses the fitted innovation covariance dimension.
2. covariance-element indices now have an explicit variable-length tuple type,
   allowing current mypy versions to type all covariance branches correctly.

## Remaining work

- standard-deviation and correlation transformations;
- bounded, profile-likelihood, likelihood-ratio, or bootstrap intervals;
- sandwich or other robust covariance estimators;
- simultaneous confidence regions;
- parameter uncertainty in filtered or smoothed state estimates.
