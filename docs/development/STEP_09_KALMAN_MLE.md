# Step 09: Kalman maximum-likelihood estimation

## Goal

Add direct Gaussian maximum-likelihood estimation over the version 0.0.8
state-space core while preserving the existing conditional estimator as a
separate, reproducible baseline.

## Delivered API

- `KalmanSTARMA`
- `KalmanSTARMAResult`
- `CovarianceType`
- `fit()` with complete or partially missing observations
- `filter()` for the training sample or a new incomplete matrix
- `to_state_space()`
- recursive `predict()` from the final filtered state

## Parameterization

Dynamic parameters follow temporal-major, spatial-minor order. Every temporal AR
or MA order includes all matrices in the supplied `SpatialWeights` collection.
The common intercept is optional.

Innovation covariance supports:

- scalar shared variance;
- diagonal location-specific variances;
- full positive-definite covariance through a lower-triangular Cholesky factor
  with exponentiated diagonal elements.

AIC and BIC count dynamic parameters and every optimized covariance parameter.

## Missing observations

The objective receives the original observation matrix. At each time, the
Kalman update uses only finite locations. A completely missing time row performs
prediction only and contributes zero to the likelihood.

Location-mean filling is used only to generate conditional starting values. It is
never used inside the likelihood or returned filtered series.

## Optimization

The estimator uses L-BFGS-B. Automatic starts come from the existing conditional
STARMA estimator when dynamic terms are present. Scalar, diagonal, and Cholesky
diagonal covariance parameters are represented on a log-standard-deviation
scale.

Stationary initialization rejects candidates whose transition spectral radius
reaches `1 - stability_margin`. The objective assigns a large continuous
feasibility penalty outside that boundary, and the final candidate is checked
again before constructing the result.

This strategy is transparent but is not a smooth stationarity
reparameterization. MA invertibility is not yet constrained.

## Verification

Focused tests cover:

- exact scalar white-noise Gaussian MLE for the mean and variance;
- scalar AR(1) recovery with systematically missing observations;
- conditional-mean recursion from the final filtered state;
- diagonal covariance recovery;
- full covariance and off-diagonal recovery;
- positive-definite full covariance construction;
- filtering new partial and fully missing rows;
- start-value, missing-data, constructor, and pre-fit validation;
- immutability and information-criterion parameter counts.

GitHub Actions CI run #210 validated the numerical implementation:

- 85 tests passed;
- total branch coverage was 87.84%, above the configured 80% threshold;
- Black, isort, Ruff, mypy, exact diagnostic fixture, and strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu and macOS passed on Python 3.11-3.14, with the Windows matrix completing
  in the same final workflow.

## Explicit exclusions

Version 0.0.9 does not yet include:

- likelihood-Hessian standard errors;
- profile likelihood or likelihood-ratio intervals;
- smooth stationarity or invertibility parameterization;
- exact diffuse likelihood;
- state or disturbance smoothing;
- integrated or multiplicative seasonal maximum-likelihood wrappers;
- Kalman predictive intervals;
- sparse state matrices.

## Next step

Add numerical likelihood curvature and inference diagnostics, including a stable
finite-difference Hessian, covariance and standard-error reporting, condition
and rank diagnostics, and explicit stationarity/invertibility checks. Smooth
constrained parameterization should follow only after the unconstrained reference
route and independent fixtures are stable.
