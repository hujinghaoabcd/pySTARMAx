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
- classical nested Yule–Walker STPACF as the default;
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

## Next priorities

1. Seasonal and multiplicative STARIMA operators.
2. Original-scale fitted-value reconstruction and forecast intervals.
3. Exact state-space/Kalman likelihood and missing observations.
4. Diagonal and full contemporaneous innovation covariance models.
5. Sparse spatial matrices and large-network computation.
6. Automatic order selection using STACF/STPACF and information criteria.
7. Exogenous regressors, interventions, and generalized/location-varying STARMA.
8. Time-varying lag and time-varying coefficient extensions.
9. GeoPandas, libpysal, NetworkX, and OSMnx adapters.
10. Cross-language estimator fixtures and rolling-origin evaluation.
