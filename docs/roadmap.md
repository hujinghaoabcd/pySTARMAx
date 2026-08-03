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

## Implemented in 0.0.9

- independent `KalmanSTARMA` maximum-likelihood estimator;
- Gaussian likelihood optimization through the state-space filter;
- complete and partially missing observation matrices;
- scalar shared innovation variance;
- diagonal location-specific innovation variance;
- full positive-definite Cholesky innovation covariance;
- conditional-estimator automatic starting values;
- explicit spectral-radius stationarity feasibility checks;
- immutable optimizer, likelihood, covariance, and filter diagnostics;
- AIC and BIC with complete covariance-parameter counting;
- fitted filtering, state-space access, and recursive mean prediction;
- scalar closed-form, AR recovery, and covariance recovery tests.

## Implemented in 0.0.10

- central finite-difference likelihood score and Hessian;
- parameter-scaled relative and absolute finite-difference steps;
- four-corner mixed-partial Hessian stencil;
- rejection of stencils that enter the stationarity penalty region;
- observed-information covariance on the raw optimizer scale;
- standard errors, z statistics, normal-approximation p values, and confidence intervals;
- separate dynamic-coefficient and full optimizer-parameter tables;
- Hessian eigenvalues, numerical rank, condition number, and maximum score diagnostics;
- explicit distance from the fitted solution to the stationarity feasibility boundary;
- default refusal to invert indefinite or rank-deficient observed information;
- explicit diagnostic positive-eigenspace pseudoinverse with result marking;
- finite outputs for zero-standard-error pseudoinverse directions;
- analytic quadratic, Gaussian white-noise, missing-data AR(1), and singular-Hessian tests;
- full documentation, example, citation metadata, and MkDocs navigation.

## Next priorities

1. Explicit stationarity and MA invertibility checks and constrained fitting.
2. Smooth stability/invertibility parameterization for optimization and inference.
3. Delta-method transforms for innovation covariance elements.
4. Integrated and multiplicative seasonal state-space/MLE wrappers.
5. State and disturbance smoothing.
6. Sparse spatial matrices and large-network computation.
7. Automatic order selection using STACF/STPACF and information criteria.
8. Exogenous regressors and intervention variables.
9. GeoPandas, libpysal, NetworkX, and OSMnx adapters.
10. Cross-language estimator fixtures and a first PyPI pre-release.
11. Optional parallel bootstrap and rolling-origin execution behind stable APIs.
12. Advanced bootstrap, profile-likelihood, sandwich, and conformal calibration.
13. Generalized, location-varying, and time-varying STARMA extensions.
