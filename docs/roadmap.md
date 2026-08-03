# Roadmap

## Implemented in 0.0.1

- immutable spatial-weight collections;
- higher-order lattice and contiguity construction;
- STAR ordinary least squares;
- STARMA iterative conditional least squares;
- recursive forecasts and deterministic simulation;
- STACF, regression diagnostic, residual portmanteau test;
- typed results, tests, documentation, and CI.

## Implemented in 0.0.2

- public space-time covariance with explicit past/future weight orientation;
- corrected classical STACF for non-symmetric spatial weights;
- classical nested Yule--Walker STPACF as the default;
- retained regression STPACF analogue behind an explicit method option;
- exact-rational independent reference fixture and regeneration check;
- singular-system solver policy and diagnostic validation.

## Implemented in 0.0.3

- arbitrary-order ordinary differencing;
- immutable end-of-sample differencing state;
- recursive original-scale forecast inversion;
- compositional `STARIMA(p,d,q)` wrapper;
- explicit stationary-scale forecasts;
- integrated-process simulation;
- zero-order compatibility with STARMA.

## Implemented in 0.0.4

- reversible seasonal differencing;
- combined ordinary-seasonal differencing state;
- factorized `(p,d,q)x(P,D,Q)_s` specification;
- ordered matrix-polynomial expansion;
- nonlinear conditional least squares for seasonal factors;
- stationary and integrated seasonal simulation;
- zero seasonal-order compatibility.

## Implemented in 0.0.5

- combined differencing-polynomial coefficients;
- aligned original-scale one-step fitted values;
- immutable `ForecastInterval` results;
- conditional innovation simulation for all public model families;
- pathwise ordinary-seasonal inversion before quantiles;
- stable simulation from singular fitted innovation covariance.

## Implemented in 0.0.6

- residual direct bootstrap of complete centered innovation vectors;
- Gaussian parametric bootstrap;
- same-length conditional pseudo-series;
- refitting for every accepted replication;
- parameter-only and combined predictive intervals;
- recursive integrated-scale reconstruction;
- linear and multiplicative seasonal bootstrap routes;
- convergence, retry, and reproducibility controls.

## Implemented in 0.0.7

- Winkler interval scores;
- immutable aggregate interval metrics;
- expanding and fixed-window rolling-origin evaluation;
- conditional and bootstrap interval dispatch;
- coverage, signed gap, absolute error, width, score, MAE, and RMSE;
- overall and horizon-specific summaries;
- deterministic per-origin random states.

## Implemented in 0.0.8

- explicit stationary STARMA state-space representation;
- companion construction from temporal-by-spatial AR and MA coefficients;
- preservation of non-symmetric weight orientation;
- stationary, known, and approximate diffuse initialization;
- fixed-parameter Gaussian filtering and log likelihood;
- partial-location and fully missing-row handling;
- immutable state, covariance, innovation, mask, and likelihood results;
- numerical jitter and covariance safeguards;
- conversion from fitted conditional STAR and STARMA estimators.

## Implemented in 0.0.9

- independent `KalmanSTARMA` maximum-likelihood estimator;
- complete and incomplete observation matrices;
- scalar, diagonal, and full Cholesky innovation covariance;
- conditional-estimator automatic starting values;
- explicit AR spectral-radius feasibility checks;
- immutable optimizer, likelihood, covariance, and filter diagnostics;
- AIC and BIC with complete parameter counting;
- fitted filtering, state-space access, and recursive mean prediction.

## Implemented in 0.0.10

- central finite-difference likelihood score and Hessian;
- parameter-scaled relative and absolute steps;
- four-corner mixed-partial stencil;
- rejection of stationarity-penalty stencil points;
- observed-information covariance;
- standard errors, normal tests, and coefficient intervals;
- dynamic and full optimizer tables;
- Hessian eigenvalues, rank, condition number, and score diagnostics;
- strict indefinite/rank-deficient behavior;
- explicit positive-eigenspace pseudoinverse for diagnosis.

## Implemented in 0.0.11

- reusable temporal-lag operator composition;
- immutable AR and inverse-MA companion diagnostics;
- complex eigenvalues, spectral radii, limits, and signed distances;
- positive-MA-sign inverse recursion using `[-B1, ..., -Bq]`;
- zero-order AR and MA diagnostics;
- joint `STARMAAdmissibility` results;
- automatic AR and MA start shrinkage;
- dual stationarity and invertibility feasibility penalties;
- independent enforcement flags and margins;
- final hard validation of both fitted polynomial blocks;
- likelihood-Hessian rejection of either enabled penalty region.

## Implemented in 0.0.12

- scalar shared-variance natural inference;
- diagonal location-variance natural inference;
- analytic full-Cholesky covariance-element Jacobian;
- first-order delta covariance `J V J.T`;
- dynamic-coefficient/covariance-element cross covariance;
- immutable natural estimates, standard errors, correlations, intervals, and
  matrix-shaped standard errors;
- one scalar variance rather than duplicated location entries;
- explicit boundary-null and unbounded-normal-interval policy;
- analytic, finite-difference, propagation, fitted-model, validation, and
  immutability tests.

## Implemented in 0.0.13

- Rauch--Tung--Striebel fixed-interval state smoothing;
- smoothed state and observation means and covariances;
- retained smoothing gains;
- lag-one state cross covariance;
- state-equation disturbance means and conditional covariances;
- positive-eigenspace pseudoinverse for rank-deficient predictions;
- per-transition prediction rank and pseudoinverse diagnostics;
- fitted `KalmanSTARMA.smooth()` for training or new incomplete data;
- scalar Gaussian bridge and direct joint-conditioning references;
- final-state, covariance-reduction, disturbance-recovery, edge, and
  immutability tests;
- correction of inherited 0.0.12 location metadata and covariance index typing;
- covariance and smoothing method guides, runnable example, and Step 12/13
  handoffs.

## Next priorities

1. Original location-level innovation disturbance smoothing using an explicit
   conditional Gaussian derivation, including posterior covariance and
   selection-matrix null-space behavior.
2. Integrated and multiplicative seasonal state-space and Kalman MLE wrappers.
3. Smooth stationarity/invertibility parameterization for optimization and
   inference.
4. Sparse spatial weights and sparse state matrices for large networks.
5. Automatic order selection using STACF, STPACF, and information criteria.
6. Exogenous regressors and intervention variables.
7. GeoPandas, libpysal, NetworkX, and OSMnx adapters.
8. Cross-language estimator fixtures and the first PyPI pre-release.
9. Optional parallel bootstrap and rolling-origin execution.
10. Advanced bootstrap, profile-likelihood, sandwich, and conformal methods.
11. Generalized, location-varying, and time-varying STARMA extensions.

## Research safeguards for future work

- Do not call state-equation disturbances original location innovations.
- Do not recover innovations with a naive pseudoinverse of the selection matrix
  without deriving conditional covariance and null-space behavior.
- Do not treat feasibility penalties as a smooth parameterization.
- Do not report singular observed-information inverses without explicit status.
- Do not silently clip variance interval endpoints or impute missing likelihood
  observations.
- Preserve non-symmetric spatial-matrix orientation in every extension.
