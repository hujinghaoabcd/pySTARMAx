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

## Implemented in 0.0.4

- reversible seasonal differencing for arbitrary `D` and period `s`;
- combined ordinary-seasonal differencing state;
- factorized `(p,d,q)x(P,D,Q)_s` model specification;
- ordered matrix-polynomial expansion with constrained cross terms;
- nonlinear conditional least-squares estimation for seasonal factors;
- stationary and integrated seasonal simulation;
- compatibility routes for zero seasonal AR/MA orders.

## Next priorities

1. Original-scale fitted-value reconstruction and forecast intervals.
2. Exact state-space/Kalman likelihood and missing observations.
3. Diagonal and full contemporaneous innovation covariance models.
4. Sparse spatial matrices and large-network computation.
5. Automatic order selection using STACF/STPACF and information criteria.
6. Exogenous regressors, interventions, and generalized/location-varying STARMA.
7. Time-varying lag and time-varying coefficient extensions.
8. GeoPandas, libpysal, NetworkX, and OSMnx adapters.
9. Cross-language estimator fixtures and rolling-origin evaluation.
