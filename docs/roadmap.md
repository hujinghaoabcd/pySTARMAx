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

## Implemented in 0.0.17: seasonal likelihood-Hessian inference

- package-root seasonal inference facade and low-level
  `infer_seasonal_kalman_starima()`;
- central finite-difference score and observed-information Hessian for ordinary
  and seasonal factor parameters plus covariance optimizer coordinates;
- objective reconstruction through the complete multiplicative matrix expansion
  and arbitrary-lag state space at every stencil point;
- rejection of expanded AR-stationarity and positive-sign MA-invertibility
  penalty points;
- shared immutable coefficient/optimizer tables, covariance, standard errors,
  normal tests, intervals, correlation, score, rank, eigenvalue, condition, and
  boundary diagnostics;
- strict full-rank positive-definite Hessian policy and explicit diagnostic
  positive-eigenspace generalized inverse;
- scalar, diagonal, and full natural innovation covariance delta-method inference
  with seasonal factor/covariance cross uncertainty;
- zero-seasonal equivalence, pure seasonal factor, penalty, singular-Hessian,
  and validation tests;
- method guide, example, navigation, README, status, and Step 17 handoff.

## Implemented in 0.0.18: Gaussian Kalman forecast intervals

- future state paths initialized from the final filtered Gaussian posterior;
- future location-level process innovations drawn from the fitted covariance;
- positive-semidefinite final-state covariance factorization with numerical
  eigenvalue tolerance;
- fixed-parameter stationary `KalmanSTARMA.predict_interval()`;
- transformed-scale `predict_differenced_interval()` for ordinary and seasonal
  Kalman STARIMA;
- original-scale `predict_interval()` for ordinary integration and combined
  ordinary-seasonal integration;
- pathwise inverse differencing before quantiles rather than inverse-transforming
  marginal lower and upper bounds;
- support for arbitrary ordinary integration order and rolling seasonal cycles;
- deterministic recursive means retained as interval centers;
- reproducible NumPy random-generator or integer-seed behavior;
- transformed-scale availability with explicit refusal of original-scale
  intervals when terminal anchors or seasonal histories are incomplete;
- analytic state-variance, ordinary random-walk, seasonal-cycle, reproducibility,
  and validation tests;
- method guide, runnable example, navigation, README, status, and Step 18
  handoff.

## Implemented in 0.0.19: exact diffuse filtering and ordinary level states

- separate exact diffuse filtering without changing approximate
  `initialization="diffuse"` semantics;
- initial covariance decomposition `P_* + kappa P_inf`;
- sequential scalar diffuse and ordinary Gaussian updates;
- immutable finite/diffuse covariance, innovation, mask, likelihood, and rank
  diagnostics;
- missing and partial-location observations without artificial rank reduction;
- deterministic zero-variance agreement and contradiction handling;
- ordinary integrated augmentation `[y, Delta y, ..., Delta^(d-1)y, beta]`;
- stationary finite initialization for the transformed STARMA state and diffuse
  initialization only for integration directions;
- first- and second-order analytic integrated likelihood references;
- large-variance-limit, `d=0`, matrix-orientation, missing-data, and
  cross-platform validation;
- mathematical guide, runnable example, navigation, README, status, and Step 19
  handoff.

## Implemented in 0.0.20: exact diffuse ordinary STARIMA MLE

- optimizer-facing `ExactDiffuseKalmanSTARIMA(p,d,q)`;
- original-level exact diffuse likelihood rather than conditional differenced
  likelihood;
- transformed and integrated state-space reconstruction at every candidate;
- stationary factor/covariance starts, L-BFGS-B bounds, and transformed
  admissibility enforcement;
- immutable result with likelihood, AIC/BIC, optimizer, diffuse-phase,
  state-space, and admissibility metadata;
- fitted filtering, state-space access, and original/highest-difference point
  forecasts;
- closed-form random-walk and second-order MLE references plus `d=0`
  stationary equivalence;
- method guide, runnable example, navigation, README, status, and Step 20
  handoff.

## Implemented in 0.0.21: exact diffuse fixed-interval smoothing

- exact diffuse backward information recursions for `r`, `r_inf`, `N`, `N1`,
  and `N2`;
- smoothed state means from finite and diffuse covariance components;
- finite posterior state covariance with all ordinary/diffuse cross terms;
- smoothed observation means and covariance;
- forward scalar finite/diffuse covariance reconstruction diagnostics;
- PSD stabilization diagnostics without a production diffuse scale;
- missing-row, leading-missing, and partial-location smoothing;
- stationary zero-diffuse equivalence with ordinary RTS smoothing;
- random-walk bridge and large-variance-limit references;
- fitted `ExactDiffuseKalmanSTARIMA.smooth()` for training and new data;
- explicit omission of lag-one, disturbance, and simulation smoothing until the
  full diffuse autocovariance recursion is implemented;
- method guide, runnable example, navigation, README, status, and Step 21
  handoff.

## Next priorities

1. Exact diffuse lag-one state covariance and disturbance smoothing.
2. Exact diffuse observed-information and natural covariance inference.
3. Exact diffuse simulation smoothing.
4. Seasonal ordinary-seasonal diffuse state augmentation and smoothing.
5. Forecast intervals for the exact diffuse estimator.
6. Parameter-aware paths, sparse state matrices, and cross-time innovation
   covariance.
7. Smooth admissibility parameterization and automatic order selection.
8. Exogenous regressors, GIS adapters, cross-language fixtures, PyPI release,
   parallel execution, robust inference, and time-varying extensions.

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
- Inverse-difference complete simulated paths before original-scale quantiles;
  never transform horizon-wise marginal bounds as though horizons were
  independent.
- Distinguish final filtered-state uncertainty, future innovation uncertainty,
  and parameter uncertainty.
- Let missing observations propagate through the complete differencing stencil;
  do not impute levels before likelihood evaluation.
- Preserve the package positive MA sign in inverse-recursion diagnostics.
- Distinguish state-equation disturbances from original location innovations.
- Preserve `Var(eta_t | R eta_t)` for non-injective selection maps.
- Do not treat feasibility penalties as smooth parameterizations.
- Do not report singular observed-information inverses without explicit status.
- Preserve non-symmetric spatial-matrix orientation in every extension.
