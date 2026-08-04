# State-space filtering and missing observations

pySTARMAx provides an explicit linear Gaussian state-space representation for
stationary STARMA models. It can be built directly from coefficients or from a
fitted conditional or maximum-likelihood estimator.

Version 0.0.8 introduced fixed-parameter filtering with incomplete observation
matrices. Version 0.0.9 added direct Gaussian Kalman maximum likelihood. Version
0.0.13 added fixed-interval smoothing. Version 0.0.19 adds a separate exact
diffuse kernel and ordinary integrated level-state constructor documented in
[Exact diffuse filtering](exact_diffuse.md).

## Model convention

For

\[
z_t=c+\sum_{i=1}^{p}A_i z_{t-i}
    +\varepsilon_t
    +\sum_{j=1}^{q}M_j\varepsilon_{t-j},
\qquad
\varepsilon_t\sim\mathcal N(0,\Sigma),
\]

operators are assembled from the ordered spatial weights:

\[
A_i=\sum_{k=0}^{\lambda}\phi_{ik}W_k,
\qquad
M_j=\sum_{k=0}^{\lambda}\theta_{jk}W_k.
\]

No symmetry or commutativity is assumed. A non-symmetric row-standardized matrix
keeps exactly the orientation supplied to simulation, fitting, filtering, and
forecasting.

The state equation is

\[
\alpha_t=d+T\alpha_{t-1}+R\eta_t,
\qquad
\eta_t\sim\mathcal N(0,\Sigma),
\]

with observation equation

\[
y_t=Z\alpha_t.
\]

The state retains the required observation lags followed by MA innovation lags.
For `p=0`, one current observation block remains so pure MA and white-noise
specifications still have an observation state. The same location innovation
enters the current observation block and the first innovation-history block.

## Direct construction

```python
import numpy as np
from pystarmax import (
    SpatialWeights,
    build_starma_state_space,
    kalman_filter,
    lattice_weights,
)

weights = SpatialWeights.from_adjacency(
    lattice_weights(2, 2),
    max_order=1,
)

state_space = build_starma_state_space(
    ar_parameters=np.array([[0.45, 0.15]]),
    ma_parameters=np.array([[0.20, 0.05]]),
    weights=weights,
    innovation_covariance=np.eye(4),
    intercept=0.0,
)

filtered = kalman_filter(observations, state_space)
print(filtered.log_likelihood)
print(filtered.filtered_observations)
```

Parameter arrays use shape `(temporal order, number of spatial weights)`. A
zero-order block is an empty matrix with the correct spatial width, for example
`np.empty((0, len(weights)))`.

## Filtering a fitted conditional estimator

```python
from pystarmax import STARMA

conditional = STARMA(ar_order=1, ma_order=1)
conditional.fit(complete_training_data, weights)

state_space = conditional.to_state_space()
filtered = conditional.filter_state_space()
```

This route maps conditional estimates and the residual covariance into the
state-space representation. It does not replace the conditional estimator or
rewrite its historical likelihood fields.

## Filtering a fitted maximum-likelihood estimator

```python
from pystarmax import KalmanSTARMA

mle = KalmanSTARMA(
    ar_order=1,
    ma_order=1,
    covariance_type="full",
)
mle.fit(training_data, weights)

state_space = mle.to_state_space()
filtered_training = mle.filter()
filtered_new = mle.filter(new_incomplete_data)
```

The maximum-likelihood model stores its fitted spatial weights, coefficients,
intercept, innovation covariance, observation matrix, and initialization policy.
Calling `filter()` without data evaluates the training matrix; supplying data
evaluates a new sequence under fixed fitted parameters.

## Missing observations

Missing cells are represented by `NaN`:

```python
incomplete = complete_training_data.copy()
incomplete[10, 2] = np.nan
incomplete[25, :] = np.nan

filtered = kalman_filter(incomplete, state_space)
```

At each time step the measurement equation is reduced to observed locations.

- A partially missing row updates the state using only observed locations.
- A fully missing row performs state prediction but no measurement update.
- A fully missing row contributes zero to the Gaussian log likelihood.
- Missing cells are never mean-filled, interpolated, or converted to zero.

`KalmanFilterResult.observed_mask` records every cell used by the likelihood.
Innovation and innovation-covariance entries outside observed cells remain
`NaN`.

## Initialization

### Stationary

```python
filtered = kalman_filter(data, state_space, initialization="stationary")
```

Stationary initialization requires transition spectral radius below one. The
unconditional state mean solves

\[
\mu_\alpha=(I-T)^{-1}d,
\]

and the covariance solves

\[
P=TPT^\top+R\Sigma R^\top.
\]

### Known moments

```python
filtered = kalman_filter(
    data,
    state_space,
    initialization="known",
    initial_state=initial_mean,
    initial_covariance=initial_covariance,
)
```

Both moments must match the complete state dimension.

### Approximate diffuse

```python
filtered = kalman_filter(
    data,
    state_space,
    initialization="diffuse",
    diffuse_scale=1e6,
)
```

This uses zero initial mean and a large diagonal covariance. It is an explicit
large-variance approximation, not an exact diffuse likelihood. The chosen scale
can affect early likelihood contributions and should be reported.

For exact initialization, use `exact_diffuse_filter()` with separate finite and
diffuse covariance components, or `build_exact_integrated_state_space()` for an
ordinary integrated level process. The exact route reports diffuse-rank paths and
contains no arbitrary `diffuse_scale`.

## Filter result

`KalmanFilterResult` stores immutable read-only arrays:

- `predicted_state` and `predicted_covariance` before measurement updates;
- `filtered_state` and `filtered_covariance` after updates;
- observation-scale `innovations`;
- observed-location `innovation_covariance` matrices;
- `observed_mask`;
- per-time `log_likelihood_contributions`;
- recorded diagonal `jitter` when a nearly singular innovation matrix requires
  stabilization;
- total `log_likelihood`, `n_observations`, and initialization metadata.

Convenience properties return predicted and filtered observation-scale means.
The complete predicted and filtered moment sequences are also the input to the
RTS smoother.

## Numerical safeguards

The filter:

- symmetrizes propagated covariance matrices;
- uses Cholesky solves instead of explicit innovation-matrix inversion;
- adds scale-aware recorded diagonal jitter only when required;
- clips only numerically tiny negative filtered-covariance eigenvalues;
- raises on materially indefinite covariance;
- validates the location innovation covariance as positive semidefinite.

These safeguards address floating-point error; they do not turn an invalid
statistical model into a valid one.

## Fixed-interval smoothing

```python
from pystarmax import kalman_smoother

smoothed = kalman_smoother(filtered)
```

The smoother returns full-interval state and observation moments, smoothing
gains, lag-one state covariance, state-equation disturbance moments, and
per-transition numerical rank diagnostics. See
[Fixed-interval state smoothing](smoothing.md) for formulas and interpretation.

## Current scope

Included through 0.0.19:

- auditable stationary and arbitrary-lag STARMA state-space construction;
- stationary, known, and approximate diffuse initialization;
- separate exact diffuse filtering with finite/diffuse covariance components;
- fixed-parameter Gaussian filtering and likelihood with partial-location and
  fully missing rows;
- stationary, ordinary-integrated, and multiplicative seasonal Kalman wrappers;
- RTS fixed-interval state smoothing and original innovation smoothing;
- exact diffuse ordinary integrated level-state construction for fixed
  transformed parameters;
- Gaussian forecast paths and pathwise original-scale intervals;
- rank, pseudoinverse, support, and diffuse-phase diagnostics.

Not included:

- optimizer-facing exact diffuse STARIMA maximum likelihood;
- exact diffuse smoothing;
- seasonal diffuse state augmentation;
- parameter-uncertainty propagation into filtering, smoothing, or Kalman paths;
- cross-time original innovation covariance and simulation smoothing;
- sparse state matrices for large location systems.
