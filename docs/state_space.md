# State-space filtering and missing observations

Version 0.0.8 introduces an explicit linear Gaussian state-space representation
for stationary `STAR` and `STARMA` models. The implementation is independent of
the existing conditional least-squares estimator: it can be constructed directly
from coefficient matrices, or from coefficients already fitted by `STAR` or
`STARMA`.

## Model convention

Let

\[
z_t = c + \sum_{i=1}^{p} A_i z_{t-i}
      + \varepsilon_t
      + \sum_{j=1}^{q} M_j \varepsilon_{t-j},
\qquad
\varepsilon_t \sim \mathcal N(0, \Sigma).
\]

Each temporal operator is assembled from the ordered spatial-weight collection:

\[
A_i = \sum_{k=0}^{\lambda} \phi_{ik} W_k,
\qquad
M_j = \sum_{k=0}^{\lambda} \theta_{jk} W_k.
\]

No symmetry or commutativity is assumed. In particular, a non-symmetric
row-standardized matrix is used in the same orientation as in the simulation,
conditional estimation, and forecasting modules.

The companion state contains the required observation lags followed by moving-
average innovation lags. For `p=0`, one current observation block is retained so
that pure MA and white-noise specifications still have an observation state.
The same innovation vector enters the current observation block and the first
innovation-history block.

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

Parameter arrays use shape `(temporal order, number of spatial weights)`. An
empty AR or MA order is represented by an empty array with the correct spatial
width, for example `np.empty((0, len(weights)))`.

## Filtering a fitted estimator

```python
from pystarmax import STARMA

model = STARMA(ar_order=1, ma_order=1)
conditional_result = model.fit(complete_training_data, weights)

state_space = model.to_state_space()
kalman_result = model.filter_state_space()
```

`fit()` remains the established conditional estimator. `to_state_space()` maps
its fitted coefficients and contemporaneous residual covariance into the state-
space representation. `filter_state_space()` then evaluates those fixed
parameters with the Kalman filter. It does not silently replace the estimator or
rewrite the existing `STARMAResult` likelihood fields.

This separation is intentional: users can compare the historical conditional
baseline with a fixed-parameter Gaussian state-space evaluation before direct
maximum-likelihood optimization is introduced.

## Missing observations

Missing cells are represented by `NaN`:

```python
incomplete = complete_training_data.copy()
incomplete[10, 2] = np.nan       # one location is missing
incomplete[25, :] = np.nan       # every location is missing

filtered = model.filter_state_space(incomplete)
```

At every time step, the measurement equation is reduced to the locations that
are actually observed.

- A partially missing row updates the state with only its observed locations.
- A fully missing row performs the state prediction but no measurement update.
- A fully missing row contributes zero to the Gaussian log likelihood.
- Missing values are not mean-filled, interpolated, or converted to zeros.

`KalmanFilterResult.observed_mask` records the exact cells used in the
likelihood. Innovations and innovation-covariance entries that do not correspond
to observed cells remain `NaN`.

## Initialization

Three initialization policies are available.

### Stationary

```python
filtered = kalman_filter(data, state_space, initialization="stationary")
```

This is the default. It requires the transition spectral radius to be strictly
below one. The unconditional mean solves

\[
\mu_\alpha = (I-T)^{-1} d,
\]

and the state covariance solves the discrete Lyapunov equation

\[
P = TPT^\top + R\Sigma R^\top.
\]

The resulting likelihood is the stationary Gaussian likelihood for the supplied
fixed parameters.

### Known

```python
filtered = kalman_filter(
    data,
    state_space,
    initialization="known",
    initial_state=initial_mean,
    initial_covariance=initial_covariance,
)
```

Both arrays must be supplied and must match the complete state dimension.

### Approximate diffuse

```python
filtered = kalman_filter(
    data,
    state_space,
    initialization="diffuse",
    diffuse_scale=1e6,
)
```

This uses a zero initial mean and a large diagonal covariance. It is an explicit
large-variance approximation, not an exact diffuse Kalman likelihood. The chosen
scale can affect early likelihood contributions and should be reported in
reproducible analyses.

## Result arrays

`KalmanFilterResult` is frozen and stores read-only arrays:

- `predicted_state` and `predicted_covariance` before each measurement update;
- `filtered_state` and `filtered_covariance` after each update;
- `innovations` on the observation scale;
- `innovation_covariance` for the locations observed at each time;
- `observed_mask`;
- per-time `log_likelihood_contributions`;
- any diagonal `jitter` used to stabilize a nearly singular innovation matrix;
- total `log_likelihood` and scalar `n_observations`.

Convenience properties return predicted and filtered observation-scale values.

## Numerical safeguards

The filter:

- symmetrizes propagated covariance matrices;
- uses Cholesky solves rather than explicit matrix inversion;
- adds a recorded, scale-aware diagonal jitter only when required;
- clips only numerically tiny negative filtered-covariance eigenvalues;
- raises if a materially indefinite covariance is encountered;
- validates the supplied innovation covariance as positive semidefinite.

These safeguards address floating-point error. They do not turn an invalid
statistical specification into a valid one.

## Statistical scope of 0.0.8

Included:

- auditable STARMA companion construction;
- stationary, known, and approximate diffuse initialization;
- fixed-parameter Gaussian Kalman likelihood;
- complete and partially missing observation sequences;
- mapping from fitted stationary `STAR` and `STARMA` models;
- full contemporaneous innovation covariance in filtering.

Not yet included:

- direct maximization of the Kalman likelihood;
- constrained covariance parameterization during optimization;
- likelihood-Hessian standard errors;
- exact diffuse initialization;
- smoothing or disturbance smoothing;
- integrated and multiplicative seasonal wrapper methods;
- Kalman-based forecasting intervals;
- sparse state matrices for large location systems.

Those omissions are explicit next-stage work rather than hidden approximations.
