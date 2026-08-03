# Step 16 handoff: multiplicative seasonal Kalman STARIMA

## Repository position

- development version: `0.0.16`;
- branch: `agent/seasonal-kalman-starima`;
- pull request: PR #16, `Add multiplicative seasonal Kalman STARIMA`;
- base: version 0.0.15 on `main`;
- authoritative implementation/documentation validation: GitHub Actions CI
  #360, run ID `30858747303`;
- validation result: 143 tests passed and 87.04% total branch coverage;
- `src/pystarmax/seasonal_maximum_likelihood.py` coverage: 87.6%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and Ubuntu/Windows/
  macOS Python 3.11--3.14 all passed.

## Delivered API

```python
model = SeasonalKalmanSTARIMA(
    ar_order=p,
    integration_order=d,
    ma_order=q,
    seasonal_ar_order=P,
    seasonal_integration_order=D,
    seasonal_ma_order=Q,
    seasonal_period=s,
    covariance_type="full",
)
result = model.fit(level_data, weights)
```

Transformed-scale methods:

```python
model.filter()
model.smooth()
model.smooth_innovation_disturbances()
model.to_state_space()
model.predict_differenced(steps)
model.fitted_differenced()
model.admissibility()
```

Original-scale methods:

```python
model.predict(steps)
model.fitted_original()
```

Public result types:

- `SeasonalKalmanSTARIMAResult`;
- `SeasonalKalmanAdmissibility`.

## Multiplicative convention

For the combined transformation

\[
x_t=(1-B)^d(1-B^s)^D y_t,
\]

the fitted factor model is

\[
\Phi_s(B^s)\Phi(B)x_t
=
c+\Theta_s(B^s)\Theta(B)\eta_t.
\]

The factor polynomials are

\[
\Phi(B)=I-\sum_iA_iB^i,
\qquad
\Phi_s(B^s)=I-\sum_rS_rB^{rs},
\]

\[
\Theta(B)=I+\sum_jM_jB^j,
\qquad
\Theta_s(B^s)=I+\sum_uN_uB^{us}.
\]

Expansion yields:

- ordinary AR coefficient `+A_i` at lag `i`;
- seasonal AR coefficient `+S_r` at lag `r*s`;
- AR cross coefficient `-S_r @ A_i` at lag `r*s + i`;
- ordinary MA coefficient `+M_j` at lag `j`;
- seasonal MA coefficient `+N_u` at lag `u*s`;
- MA cross coefficient `+N_u @ M_j` at lag `u*s + j`.

The seasonal factor remains on the left. Matrix product order is not commuted.

## Cross-lag matrix policy

Products of spatial-weight combinations need not belong to the original
`SpatialWeights` span. The implementation therefore:

- constructs factor matrices from the supplied weights;
- forms ordered cross products directly;
- aggregates matrices only when temporal lags coincide;
- stores arbitrary expanded matrices in the result;
- does not regress or project cross terms back onto `(W0, W1, ...)`.

Do not add a basis-projection shortcut without an explicit approximation model,
loss function, and diagnostics.

## Factor parameter count

For `K` stored spatial weights, the optimized dynamic count is

\[
\mathbf 1_c+K(p+P+q+Q).
\]

The expanded cross-lag matrices are deterministic functions of these factors.
They are not independent optimizer parameters and must not be counted in AIC or
BIC. Covariance parameters are added through the existing scalar, diagonal, or
full Cholesky codec.

## Arbitrary-lag companion state

The expanded recursion can contain gaps, for example lags `1`, `s`, and
`s+1`. The state-space builder:

1. creates dense matrix arrays through the maximum AR and MA lag;
2. inserts zero matrices for absent lags;
3. places expanded AR matrices in the observation-state companion top row;
4. places expanded MA matrices in the innovation-history part of that row;
5. shifts observation and innovation histories with identity blocks;
6. injects the current innovation into the current observation state and the
   first innovation-history block when MA terms exist;
7. selects the current transformed observation with the design matrix.

The result's `ar_order` and `ma_order` in the internal `StateSpaceModel` are the
maximum expanded lags, not the factor orders `p`, `P`, `q`, and `Q`.

## Expanded admissibility

AR stationarity uses the complete expanded companion top row. Positive-sign MA
invertibility uses the negatives of the complete expanded MA matrices.

`SeasonalKalmanAdmissibility` stores:

- AR and inverse-MA spectral radii;
- stability and invertibility limits;
- expanded AR and MA lags and matrices;
- separate stationarity/invertibility decisions;
- joint admissibility.

Starting factor blocks are shrunk toward zero independently for AR and MA when
necessary. Optimization uses explicit feasibility penalties, followed by final
hard checks. This remains a feasibility policy, not a smooth bijection.

## Conditional likelihood

The combined transformation offset is

\[
o=d+Ds.
\]

The evaluated likelihood is

\[
\ell_c(\vartheta;y_{1:T})
=
\ell\left(
\vartheta;(1-B)^d(1-B^s)^D y_{o+1:T}
\mid\mathcal H_o
\right).
\]

The removed ordinary and seasonal history is conditioned on. The API must not
be described as exact diffuse integration on the original level process.
`initialization="diffuse"` remains the approximate large-variance initialization
for the stationary expanded transformed state.

## Missing-observation policy

Missing original cells are not imputed. Ordinary and seasonal differences are
formed with normal arithmetic, so `NaN` propagates through the complete stencil.
For one seasonal difference, a missing `y_t` affects terms involving both
`y_t-y_(t-s)` and `y_(t+s)-y_t` when those rows exist.

The transformed filter then:

- uses finite locations in each measurement update;
- omits missing transformed locations;
- performs prediction only for fully missing transformed rows;
- counts only finite transformed observations in the likelihood.

The result records both original and transformed missing-cell counts.

## Original-scale reconstruction

`CombinedDifferencingState.inverse_forecast()` reverses the transformations in
the opposite order from fitting:

1. seasonal differencing is reversed using rolling seasonal histories;
2. ordinary differencing is reversed using terminal lower-order anchors.

Original-scale prediction requires finite ordinary anchors and every stored
seasonal history. If any are unavailable, transformed forecasts remain valid but
`predict()` raises.

`fitted_original()` uses the complete combined differencing polynomial. It
returns an array aligned to the original sample, leaves the first `d + D*s`
rows unavailable, and reconstructs later values only when every required
observed lag is finite.

## Smoothing and original innovations

RTS smoothing and conditional-Gaussian original innovation smoothing reuse the
existing fixed-parameter routines on the expanded transformed state space:

```python
state_result = model.smooth(new_levels)
innovation_result = model.smooth_innovation_disturbances(new_levels)
```

These results remain on the transformed process. They do not provide a smoothed
original level-state distribution.

## Validation references

Tests include:

1. exact zero-seasonal-order equivalence with `KalmanSTARMA` under identical
   starts;
2. direct scalar expansion showing AR coefficients `[a, A, -A*a]` and MA
   coefficients `[b, B, B*b]` at lags `[1, s, s+1]`;
3. pure seasonal AR estimation near a known coefficient;
4. stationarity and invertibility on the complete expanded companions;
5. pure seasonal random-walk original-scale reconstruction from a rolling
   cycle;
6. seasonal missing propagation and finite observation counts;
7. transformed forecast availability and original forecast refusal with an
   incomplete terminal cycle;
8. combined ordinary-seasonal new-data filter, smoother, and innovation lengths;
9. aligned original fitted values with missing combined lag history;
10. period, order, sample-offset, dimensionality, infinity, and fitted-state
    validation;
11. the complete inherited package test suite.

Authoritative CI #360 reported:

- 143 tests passed;
- total branch coverage: 87.04%;
- seasonal Kalman module coverage: 87.6%;
- Black, isort, Ruff, mypy, strict MkDocs, distributions, and diagnostic fixture
  regeneration passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

The validation-record-only head changes only this handoff and
`PROJECT_STATUS.md`. It receives a final merge-gate CI before PR #16 is marked
ready and merged.

## Files introduced or changed

Core:

- `src/pystarmax/seasonal_maximum_likelihood.py`;
- `src/pystarmax/__init__.py`;
- `CITATION.cff`.

Tests:

- `tests/test_seasonal_maximum_likelihood.py`.

Documentation and example:

- `docs/seasonal_maximum_likelihood.md`;
- `examples/seasonal_kalman_starima.py`;
- README, documentation home, navigation, seasonal guide, roadmap, and project
  status;
- Step 16 handoff.

## Current limitations

- no exact diffuse seasonal level-state likelihood;
- no seasonal-factor observed-information or natural covariance inference;
- no original-scale Gaussian forecast intervals;
- no original-scale filtered/smoothed level-state distribution;
- no parameter-uncertainty propagation through smoothing;
- no cross-time original innovation covariance or simulation smoother;
- dense arbitrary-lag companions can be large;
- no exogenous regressors or interventions.

## Next recommended stage

The next focused stage should add observed-information inference for seasonal
factor parameters, including a finite-difference Hessian that rejects expanded
stationarity/invertibility penalty points and a natural covariance
transformation using the existing covariance codec. An alternative product
stage is original-scale Gaussian forecast intervals with pathwise combined
inverse differencing.
