# Roadmap

## Implemented in 0.0.1--0.0.7: conditional modelling and evaluation

- immutable spatial-weight collections and higher-order constructors;
- STAR ordinary least squares and STARMA iterative conditional least squares;
- classical STACF, nested Yule--Walker STPACF, regression analogues, and
  residual portmanteau tests;
- arbitrary ordinary and seasonal differencing with reversible terminal states;
- conditional `STARIMA(p,d,q)` and multiplicative
  `(p,d,q)x(P,D,Q)_s` estimation;
- ordered matrix-polynomial expansion without cross-term basis projection;
- deterministic and stochastic simulation;
- aligned original-scale fitted values;
- conditional future-innovation and parameter-refitting bootstrap intervals;
- expanding and fixed-window rolling-origin interval evaluation;
- coverage, width, Winkler score, MAE, RMSE, and horizon summaries.

## Implemented in 0.0.8--0.0.9: state space and stationary Kalman MLE

- explicit stationary STARMA companion state space;
- preservation of non-symmetric spatial-weight orientation;
- stationary, known, and approximate diffuse initialization;
- Gaussian filtering and log likelihood with partial-location and fully missing
  rows;
- immutable state, covariance, innovation, mask, and likelihood results;
- independent `KalmanSTARMA` maximum-likelihood estimator;
- scalar, diagonal, and full Cholesky innovation covariance;
- conditional-estimator automatic starts;
- optimizer, likelihood, covariance, AIC, BIC, filtering, state-space, and
  recursive mean-prediction diagnostics.

## Implemented in 0.0.10--0.0.12: admissibility and inference

- central finite-difference score and observed-information Hessian;
- parameter-scaled steps and mixed-partial stencil;
- coefficient covariance, standard errors, normal tests, intervals,
  correlations, rank, eigenvalues, condition number, and score diagnostics;
- strict indefinite/rank-deficient behavior and explicit positive-eigenspace
  diagnostic pseudoinverse;
- reusable AR and positive-sign inverse-MA companion diagnostics;
- complex eigenvalues, spectral radii, limits, and signed boundary distances;
- independent stationarity and invertibility start shrinkage, penalties, and
  final validation;
- scalar, diagonal, and analytic full-Cholesky natural covariance
  delta-method inference;
- dynamic-coefficient/covariance-element cross covariance and explicit boundary
  inference policy.

## Implemented in 0.0.13--0.0.14: smoothing and original innovations

- Rauch--Tung--Striebel fixed-interval state smoothing;
- smoothed state and observation means and covariances;
- retained smoothing gains and lag-one state covariance;
- state-equation disturbance posterior moments;
- rank-deficient prediction diagnostics and positive-eigenspace inverse policy;
- exact conditional-Gaussian mapping from `w_t = R eta_t` to original
  location-level innovations;
- covariance-weighted map `Q R.T (R Q R.T)+`;
- posterior innovation means and marginal covariances;
- retained unresolved selection-nullspace covariance;
- process rank, pseudoinverse use, and state-support residual diagnostics;
- fitted filtering/smoothing routes for training or new incomplete data.

## Implemented in 0.0.15: conditional ordinary-integrated Kalman STARIMA

- public `KalmanSTARIMA(p,d,q)` wrapper around the stationary Gaussian core;
- ordinary finite differencing before likelihood evaluation;
- explicit conditional likelihood
  `L(Delta^d y_(d+1:T) | y_(1:d))`;
- explicit separation from exact diffuse integrated level-state likelihood;
- immutable integrated result metadata;
- distinct transformed-scale and original-scale methods;
- arbitrary non-negative integration order through `DifferencingState`;
- pathwise recursive inverse differencing;
- terminal-anchor validation and refusal of indefensible level forecasts;
- missing-value propagation through the ordinary finite-difference stencil;
- inherited filtering, smoothing, innovation smoothing, admissibility, and
  observed-information inference on the transformed scale;
- exact `d=0` equivalence and analytic random-walk/inverse-difference tests.

## Implemented in 0.0.16: multiplicative seasonal Kalman STARIMA

- public `SeasonalKalmanSTARIMA(p,d,q)x(P,D,Q)_s`;
- Gaussian Kalman likelihood on the combined ordinary-seasonally transformed
  process;
- explicit conditional likelihood on the removed transformation history;
- ordinary and seasonal AR/MA factor parameters rather than independent expanded
  cross-lag parameters;
- ordered matrix products `-S_r @ A_i` for AR cross lags and
  `+N_u @ M_j` for positive-sign MA cross lags;
- direct use of arbitrary cross-lag matrices without spatial-basis projection;
- aggregation of equal temporal lags and dense arbitrary-lag companion
  construction;
- stationarity and inverse-MA spectral radii on the complete expanded
  recursions;
- independent factor-block start shrinkage, feasibility penalties, and final
  admissibility checks;
- scalar, diagonal, and full Cholesky innovation covariance;
- factor-based AIC/BIC parameter counting that excludes deterministic cross
  terms;
- combined ordinary-seasonal missing-value propagation without imputation;
- transformed-scale filtering, RTS smoothing, original innovation smoothing,
  and forecasts;
- pathwise seasonal-then-ordinary original-scale reconstruction;
- terminal ordinary-anchor and seasonal-history validation;
- aligned original-scale fitted values using the complete combined differencing
  polynomial;
- zero-seasonal equivalence, multiplicative sign, pure seasonal AR, seasonal
  random-walk, missing propagation, and new-data validation tests;
- method guide, runnable example, navigation, README, status, and Step 16
  handoff.

## Next priorities

1. Observed-information Hessian and natural innovation covariance inference for
   `SeasonalKalmanSTARIMA` factor parameters.
2. Original-scale Gaussian forecast intervals for ordinary and seasonal Kalman
   STARIMA, with pathwise inverse differencing before quantiles.
3. Exact diffuse integrated level-state likelihood and smoothing as a separate
   API from the conditional transformed likelihoods.
4. Sparse spatial weights and sparse arbitrary-lag state matrices for large
   seasonal periods and networks.
5. Cross-time innovation-disturbance covariance and conditional simulation
   smoothing.
6. Smooth stationarity/invertibility parameterization for optimization and
   inference.
7. Automatic ordinary and seasonal order selection using diagnostics and
   information criteria.
8. Exogenous regressors and intervention variables.
9. GeoPandas, libpysal, NetworkX, and OSMnx adapters.
10. Cross-language estimator fixtures and the first PyPI pre-release.
11. Optional parallel bootstrap and rolling-origin execution.
12. Profile likelihood, sandwich, conformal, generalized, location-varying, and
    time-varying extensions.

## Research safeguards for future work

- Distinguish conditional transformed likelihoods from exact diffuse integrated
  likelihoods.
- Count optimized multiplicative factor parameters, not deterministic expanded
  cross-lag matrices.
- Preserve ordered matrix products and never project cross terms back onto a
  spatial-weight basis without an explicit approximation model.
- Keep transformed-scale filtering, smoothing, inference, and innovation
  results explicitly labelled.
- Never reconstruct original-scale forecasts without finite ordinary anchors
  and seasonal histories.
- Let missing observations propagate through the complete differencing stencil;
  do not impute levels before likelihood evaluation.
- Preserve the package positive MA sign in inverse-recursion diagnostics.
- Distinguish state-equation disturbances from original location innovations.
- Preserve `Var(eta_t | R eta_t)` for non-injective selection maps.
- Do not treat feasibility penalties as smooth parameterizations.
- Do not report singular observed-information inverses without explicit status.
- Preserve non-symmetric spatial-matrix orientation in every extension.
