# Multiplicative seasonal Kalman STARIMA

Version 0.0.16 adds `SeasonalKalmanSTARIMA(p,d,q)x(P,D,Q)_s`, a Gaussian
Kalman maximum-likelihood estimator for the ordinary-seasonally transformed
process.

The estimator preserves the multiplicative factor structure. Ordinary and
seasonal coefficients are optimized as factors; the induced cross-lag matrices
are constructed from ordered matrix products and are not treated as independent
parameters or projected back onto the spatial-weight basis.

## Transformed process

Let

\[
x_t=(1-B)^d(1-B^s)^D y_t,
\]

where `d` is the ordinary integration order, `D` is the seasonal integration
order, and `s` is the seasonal period. The transformed process satisfies

\[
\Phi_s(B^s)\Phi(B)x_t
=
c+\Theta_s(B^s)\Theta(B)\eta_t,
\qquad
\eta_t\sim\mathcal N(0,\Sigma).
\]

With positive moving-average signs,

\[
\Phi(B)=I-\sum_{i=1}^{p}A_iB^i,
\qquad
\Phi_s(B^s)=I-\sum_{r=1}^{P}S_rB^{rs},
\]

\[
\Theta(B)=I+\sum_{j=1}^{q}M_jB^j,
\qquad
\Theta_s(B^s)=I+\sum_{u=1}^{Q}N_uB^{us}.
\]

The factor matrices are spatial-weight combinations:

\[
A_i=\sum_k\phi_{ik}W_k,
\quad
S_r=\sum_k\Phi_{rk}W_k,
\quad
M_j=\sum_k\theta_{jk}W_k,
\quad
N_u=\sum_k\Theta_{uk}W_k.
\]

## Multiplicative expansion

The autoregressive product expands as

\[
\Phi_s(B^s)\Phi(B)
=
I-\sum_iA_iB^i-\sum_rS_rB^{rs}
+\sum_{r,i}S_rA_iB^{rs+i}.
\]

Moving all non-identity terms to the right-hand side gives the recursion

\[
x_t
=
\sum_iA_ix_{t-i}
+\sum_rS_rx_{t-rs}
-\sum_{r,i}S_rA_ix_{t-rs-i}
+\cdots.
\]

The cross-lag AR coefficient is therefore `-S_r @ A_i`. Matrix order is
preserved. For non-commuting spatial operators, `S_r @ A_i` cannot be replaced
by `A_i @ S_r`.

The moving-average product expands as

\[
\Theta_s(B^s)\Theta(B)
=
I+\sum_jM_jB^j+\sum_uN_uB^{us}
+\sum_{u,j}N_uM_jB^{us+j}.
\]

The MA cross-lag coefficient is `+N_u @ M_j`. Because the package uses a
positive MA sign, inverse-MA admissibility uses the companion top row formed
from the negatives of the complete expanded MA recursion.

## No basis projection

A product such as `S_r @ A_i` is generally not representable by the original
finite collection `(W0, W1, ...)`. The estimator therefore stores and uses the
expanded matrices directly. It does not regress or project them back onto the
spatial-weight basis.

This has two consequences:

1. the state-space transition can contain arbitrary matrix coefficients at
   ordinary, seasonal, and cross lags;
2. AIC and BIC count only the optimized ordinary and seasonal factor
   coefficients plus covariance parameters. Cross-lag matrices are deterministic
   functions of those factors and are not independent parameters.

## Conditional likelihood

For an original sample with `T` rows, the combined transformation removes

\[
o=d+Ds
\]

initial rows. The estimator evaluates

\[
\ell_c(\vartheta;y_{1:T})
=
\ell\left(
\vartheta;
(1-B)^d(1-B^s)^D y_{o+1:T}
\mid \mathcal H_o
\right),
\]

where \(\mathcal H_o\) is the ordinary and seasonal history required by the
transformation.

The reported likelihood, AIC, and BIC describe this transformed conditional
model. The API is not an exact diffuse likelihood for an integrated seasonal
level-state model. `initialization="diffuse"` remains the existing
large-variance approximation for the stationary expanded STARMA state.

## Arbitrary-lag state space

After multiplicative expansion, non-zero AR and MA matrices may occur at lags
such as `1`, `s`, and `s + 1`, with gaps between them. The estimator builds a
dense companion representation up to the maximum expanded lag:

- absent lags receive zero matrices;
- AR matrices fill the observation-state companion top row;
- MA matrices fill the innovation-history portion of the top row;
- the current innovation enters both the current observation state and the first
  innovation-history block when MA terms are present;
- the design matrix selects the current transformed observation.

The state dimension therefore depends on the maximum expanded AR and MA lags,
not only on the number of non-zero terms.

## Fitting

```python
from pystarmax import SeasonalKalmanSTARIMA

model = SeasonalKalmanSTARIMA(
    ar_order=1,
    integration_order=1,
    ma_order=1,
    seasonal_ar_order=1,
    seasonal_integration_order=1,
    seasonal_ma_order=1,
    seasonal_period=24,
    covariance_type="full",
    include_intercept=False,
    enforce_stationarity=True,
    enforce_invertibility=True,
)
result = model.fit(level_series, weights)

print(result.summary())
print(model.admissibility().summary())
```

`result.params` contains only factor coefficients and the optional intercept.
The factor arrays are available separately as:

- `ar_parameters`;
- `seasonal_ar_parameters`;
- `ma_parameters`;
- `seasonal_ma_parameters`.

The complete expanded matrices and their lags are exposed as `ar_lags`,
`ar_matrices`, `ma_lags`, and `ma_matrices`.

## Stationarity and invertibility

The estimator evaluates spectral radii on the complete expanded recursions:

- AR stationarity uses the arbitrary-lag companion with the expanded AR
  matrices in its top row;
- MA invertibility uses the same construction with the negatives of the
  expanded positive-sign MA matrices.

```python
admissibility = model.admissibility()
print(admissibility.ar_spectral_radius)
print(admissibility.ma_inverse_spectral_radius)
print(admissibility.stationary)
print(admissibility.invertible)
```

Automatic starting values are shrunk toward zero separately for the AR-factor
and MA-factor blocks until the expanded recursions lie inside conservative
starting regions. During optimization, candidates outside an enabled region
receive explicit feasibility penalties. The final candidate is checked again.
As in the stationary estimator, this is not a smooth bijective
parameterization.

## Scale boundaries

The following methods operate on the transformed process
`(1-B)^d(1-B^s)^D y_t`:

```python
model.filter()
model.smooth()
model.smooth_innovation_disturbances()
model.to_state_space()
model.predict_differenced(steps=12)
model.fitted_differenced()
```

The original scale is available through:

```python
model.predict(steps=12)
model.fitted_original()
```

`predict()` first reverses seasonal differencing pathwise using the stored
seasonal histories, then reverses ordinary differencing using the stored lower-
order terminal anchors. For seasonal order one, each future transformed value
updates the corresponding position in a rolling queue of the last `s` values.

`fitted_original()` aligns transformed one-step means to the original sample.
The first `d + D*s` rows are unavailable. Later values are restored only when
every observed lag required by the combined differencing polynomial is finite.

## Missing observations

Missing original cells are not imputed. They propagate through both
transformations:

- ordinary differencing propagates a missing value across its local finite-
  difference stencil;
- one seasonal difference propagates a missing value to the transformed rows
  involving both `y_t` and `y_(t+s)`;
- repeated ordinary or seasonal differencing expands this affected set.

The transformed Kalman filter then applies the existing partial-location rules.
Finite locations participate in the update, missing locations are omitted, and
a fully missing transformed row performs prediction only.

The result reports `original_missing_cells` and `transformed_missing_cells`.

## New data, smoothing, and innovations

```python
filtered = model.filter(new_levels)
smoothed = model.smooth(new_levels)
innovation_result = model.smooth_innovation_disturbances(new_levels)
```

A new matrix must include its own ordinary and seasonal history. If it contains
`T_new` level rows, the transformed filter has
`T_new - d - D*s` state times.

RTS state smoothing and covariance-weighted original innovation smoothing reuse
the same fixed-parameter routines as stationary STARMA. Their outputs remain on
the transformed scale.

## Terminal histories

Original-scale forecasting requires finite:

- ordinary terminal anchors for orders `0` through `d - 1`;
- the last `s` values at each seasonal-difference level below order `D`.

If any required value is non-finite, transformed-scale estimation, point
forecasts, and `predict_differenced_interval()` remain available, but
`predict()` and `predict_interval()` raise instead of interpolating or carrying
a value forward. `original_scale_forecast_available` records this state.

## Gaussian forecast intervals

```python
transformed_interval = model.predict_differenced_interval(
    steps=24,
    level=0.95,
    n_simulations=5000,
    random_state=42,
)
original_interval = model.predict_interval(
    steps=24,
    level=0.95,
    n_simulations=5000,
    random_state=42,
)
```

Future paths draw the final filtered state and future location-level
innovations under fixed fitted parameters. Original-scale bounds reverse the
combined differencing state for every complete simulated path, including
rolling seasonal histories, before taking quantiles. See
[Gaussian Kalman forecast intervals](kalman_forecast_intervals.md).

## Result and parameter counting

`SeasonalKalmanSTARIMAResult` contains:

- factor parameters and names;
- raw optimizer parameters and names;
- expanded lag matrices;
- innovation covariance;
- ordinary and seasonal orders;
- log likelihood, AIC, BIC, finite observation count, and parameter count;
- convergence and optimizer diagnostics;
- expanded AR and inverse-MA spectral radii and limits;
- original/transformed row and missing-cell counts;
- original-scale forecast availability;
- the training `KalmanFilterResult`.

For `K` spatial weights, the dynamic factor count is

\[
\mathbf 1_{c}
+K(p+P+q+Q),
\]

not the number of expanded temporal lags. Covariance parameters are added
according to scalar, diagonal, or full Cholesky covariance type.

## Observed-information inference

```python
inference = model.infer(
    relative_step=1e-4,
    absolute_step=1e-6,
    rcond=1e-10,
)
```

Version 0.0.17 reconstructs every finite-difference point through the complete
factor expansion and arbitrary-lag state space. An enabled expanded
stationarity/invertibility penalty point invalidates the stencil. The shared
`LikelihoodInferenceResult` provides factor and optimizer tables, standard
errors, intervals, score and Hessian diagnostics, and natural innovation
covariance delta-method inference.

See [Seasonal likelihood inference](seasonal_likelihood_inference.md).

## Validation references

The implementation is checked through:

- exact zero-seasonal-order equivalence with `KalmanSTARMA` under identical
  starts;
- direct scalar verification of AR and MA multiplicative cross-lag signs;
- pure seasonal AR estimation and expanded companion admissibility;
- seasonal random-walk reconstruction from a rolling cycle history;
- seasonal missing-value propagation and finite likelihood counts;
- transformed interval availability and original interval refusal with incomplete terminal seasonal history;
- combined ordinary-seasonal new-data filter and smoother lengths;
- aligned original fitted values using the complete combined lag polynomial;
- constructor, dimensionality, infinity, sample-offset, and fitted-state
  validation.

## Current limitations

- likelihood is conditional on the ordinary-seasonal transformation history,
  not exact diffuse on a level-state representation;
- observed-information inference uses central finite differences and can be
  expensive or step sensitive for large seasonal/full-covariance models;
- forecast intervals condition on fitted parameters and use Monte Carlo quantiles;
- original-scale filtered and smoothed level-state distributions are not
  returned;
- state and innovation smoothing treat parameters as fixed;
- matrices are dense and the state dimension can grow rapidly with `s`, `P`,
  and `Q`;
- exogenous regressors and interventions are unsupported.
