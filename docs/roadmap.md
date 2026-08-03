# Roadmap

## Implemented in 0.0.1

- immutable spatial-weight collections;
- higher-order lattice/contiguity construction;
- STAR ordinary least squares;
- STARMA iterative conditional least squares;
- recursive forecasts and deterministic simulation;
- STACF, regression diagnostic, residual portmanteau test;
- typed result objects, tests, documentation, and CI.

## Implemented in 0.0.2

- public space-time covariance with explicit past/future weight orientation;
- corrected classical STACF for non-symmetric spatial weights;
- classical nested Yule-Walker STPACF as the default;
- retained regression STPACF analogue behind an explicit method option;
- exact-rational independent reference fixture and regeneration check;
- singular-system solver policy and additional diagnostic validation tests.

## Implemented in 0.0.3

- ordinary temporal differencing for arbitrary non-negative order;
- immutable end-of-sample differencing state;
- recursive original-scale forecast inversion;
- compositional `STARIMA(p, d, q)` wrapper over the STARMA estimator;
- explicit stationary-scale forecasts through `predict_differenced()`;
- integrated-process simulation with optional initial difference state;
- zero-order compatibility tests against the existing STARMA API.

## Implemented in 0.0.4

- reversible seasonal differencing for arbitrary `D` and period `s`;
- combined ordinary-seasonal differencing state;
- factorized `(p,d,q)x(P,D,Q)_s` model specification;
- ordered matrix-polynomial expansion with constrained cross terms;
- nonlinear conditional least-squares estimation for seasonal factors;
- stationary and integrated seasonal simulation;
- compatibility routes for zero seasonal AR/MA orders.

## Implemented in 0.0.5

- combined differencing-polynomial coefficients;
- aligned original-scale one-step fitted values;
- immutable `ForecastInterval` results;
- conditional innovation simulation for STAR, STARMA, STARIMA, and seasonal STARIMA;
- pathwise ordinary-seasonal inverse differencing before interval quantiles;
- stable simulation from singular fitted innovation covariance matrices;
- reproducible interval tests across linear and nonlinear seasonal cores.

## Implemented in 0.0.6

- residual direct bootstrap using centered complete innovation vectors;
- Gaussian parametric bootstrap from fitted contemporaneous covariance;
- same-length conditional pseudo-series generation;
- model refitting for every accepted bootstrap replication;
- parameter-only and combined parameter/future-innovation intervals;
- recursive pseudo-series restoration for ordinary and seasonal integration;
- linear and multiplicative seasonal bootstrap routes;
- explicit convergence, retry, and reproducibility controls;
- focused tests for all public model families.

## Implemented in 0.0.7

- central Winkler interval scores;
- immutable aggregate interval metrics;
- immutable rolling-origin forecast arrays;
- expanding-window and fixed-length rolling-window evaluation;
- conditional-innovation and bootstrap interval dispatch through one API;
- empirical coverage, signed coverage gap, absolute coverage error, and width;
- mean interval score, point-forecast MAE, and RMSE;
- overall and forecast-horizon-specific summaries;
- deterministic per-origin random-state derivation;
- examples and documentation for calibration diagnostics.

## Implemented in 0.0.8

- explicit linear Gaussian state-space representation for stationary STARMA;
- companion construction from temporal-by-spatial AR and MA coefficients;
- preservation of non-symmetric spatial-weight orientation;
- stationary initialization from the unconditional mean and Lyapunov covariance;
- user-supplied known initialization;
- explicit approximate diffuse initialization;
- Gaussian Kalman filtering and fixed-parameter log likelihood;
- partial-location and fully missing-row observation handling;
- immutable predicted/filtered state, covariance, innovation, and mask results;
- numerical Cholesky jitter and covariance-stability safeguards;
- fitted `STAR` and `STARMA` state-space conversion and filtering methods;
- scalar closed-form and missing-observation reference tests.

## Next priorities

1. Direct maximum-likelihood optimization over the Kalman likelihood.
2. Scalar, diagonal, and full innovation-covariance parameterizations.
3. Likelihood-Hessian uncertainty and optimizer diagnostics.
4. Stationarity and invertibility checks with constrained parameterization.
5. Integrated and multiplicative seasonal state-space wrappers.
6. Sparse spatial matrices and large-network computation.
7. Automatic order selection using STACF/STPACF and information criteria.
8. Exogenous regressors, interventions, and generalized/location-varying STARMA.
9. GeoPandas, libpysal, NetworkX, and OSMnx adapters.
10. Cross-language estimator fixtures and a first PyPI pre-release.
11. Optional parallel bootstrap and rolling-origin execution behind stable APIs.
12. Block, wild, predictive-residual, studentized, and bias-corrected bootstrap methods.
13. Automated interval recalibration or conformal post-processing after benchmark evidence.
