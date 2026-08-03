# STARMA stationarity and invertibility

Version 0.0.11 adds reusable matrix-polynomial diagnostics for autoregressive
stationarity and moving-average invertibility. The same definitions are used by
the public diagnostic functions, `KalmanSTARMA` starting values, the likelihood
objective, final fitted-result validation, and finite-difference likelihood
inference.

## Model sign convention

pySTARMAx writes the stationary STARMA model as

\[
z_t = c + \sum_{i=1}^{p} A_i z_{t-i}
      + \varepsilon_t
      + \sum_{j=1}^{q} B_j \varepsilon_{t-j},
\]

where each temporal-lag matrix is assembled from the ordered spatial weights:

\[
A_i = \sum_k \phi_{ik} W_k,
\qquad
B_j = \sum_k \theta_{jk} W_k.
\]

The supplied orientation of every `W_k` is preserved. Symmetry and
commutativity are not assumed.

## Autoregressive stationarity

For AR order `p > 0`, define the block companion matrix

\[
C_{AR} =
\begin{bmatrix}
A_1 & A_2 & \cdots & A_p \\
I   & 0   & \cdots & 0 \\
0   & I   & \cdots & 0 \\
\vdots & & \ddots & \vdots
\end{bmatrix}.
\]

The implemented criterion is

\[
\rho(C_{AR}) < 1 - m_{AR},
\]

where `m_AR` is the configured stability margin. The signed boundary distance is

\[
d_{AR} = 1 - m_{AR} - \rho(C_{AR}).
\]

A positive value is inside the requested region; zero or a negative value is on
or beyond the boundary.

## Moving-average invertibility

Because the package uses a **positive** MA sign,

\[
r_t = \varepsilon_t + \sum_{j=1}^{q} B_j\varepsilon_{t-j},
\]

the inverse innovation recursion is

\[
\varepsilon_t = r_t - \sum_{j=1}^{q} B_j\varepsilon_{t-j}.
\]

The relevant inverse-recursion companion matrix therefore has top block row

\[
[-B_1, -B_2, \ldots, -B_q].
\]

The implemented invertibility criterion is

\[
\rho(C_{MA}^{-1}) < 1 - m_{MA},
\]

with signed distance

\[
d_{MA} = 1 - m_{MA} - \rho(C_{MA}^{-1}).
\]

For scalar MA(1), this reduces to the familiar condition
`abs(theta_1) < 1 - margin`. For scalar higher-order models, the companion
criterion is equivalent to requiring the roots of

\[
1 + \theta_1 z + \cdots + \theta_q z^q
\]

to lie outside the reciprocal spectral boundary.

## Public diagnostics

```python
from pystarmax import (
    autoregressive_diagnostics,
    moving_average_diagnostics,
    starma_admissibility,
)

ar = autoregressive_diagnostics(
    ar_parameters,
    weights,
    margin=1e-6,
)
ma = moving_average_diagnostics(
    ma_parameters,
    weights,
    margin=1e-6,
)
joint = starma_admissibility(
    ar_parameters,
    ma_parameters,
    weights,
    stability_margin=1e-6,
    invertibility_margin=1e-6,
)

print(ar.spectral_radius)
print(ar.eigenvalues)
print(ma.companion_matrix)
print(joint.summary())
```

`PolynomialAdmissibility` stores immutable copies of:

- the composed temporal-lag operators;
- the block companion matrix;
- complex companion eigenvalues;
- spectral radius;
- requested limit;
- signed boundary distance;
- the final admissibility decision.

`STARMAAdmissibility` combines AR and MA diagnostics and exposes
`stationary`, `invertible`, `admissible`, and `minimum_distance`.

Low-level scalar helpers are also available:

```python
from pystarmax import (
    autoregressive_spectral_radius,
    moving_average_inverse_spectral_radius,
)

ar_radius = autoregressive_spectral_radius(ar_parameters, weights)
ma_radius = moving_average_inverse_spectral_radius(ma_parameters, weights)
```

Zero-order AR or MA polynomials are represented by empty companion matrices and
have spectral radius zero.

## Spatial-weight coercion

The diagnostics follow the package's established weight convention:

- an existing `SpatialWeights` object is used unchanged;
- a single NumPy matrix is treated as a non-identity spatial matrix and the
  identity lag is prepended;
- a matrix sequence is converted with `SpatialWeights.from_matrices()`.

For unambiguous coefficient widths and names, passing `SpatialWeights` directly
is recommended.

## Kalman maximum-likelihood constraints

`KalmanSTARMA` now enables both criteria by default:

```python
model = KalmanSTARMA(
    ar_order=2,
    ma_order=2,
    enforce_stationarity=True,
    stability_margin=1e-6,
    enforce_invertibility=True,
    invertibility_margin=1e-6,
)
result = model.fit(data, weights)
```

The fitting route applies the criteria at three stages:

1. automatic conditional-estimator starts are iteratively shrunk into the
   configured AR and MA regions;
2. likelihood candidates outside either enabled region receive a large
   feasibility penalty;
3. the optimizer's final candidate is decoded and hard-checked before a result
   is returned.

This remains a feasibility-penalty strategy, not a smooth unconstrained
parameterization. A user-provided `start_params` vector is validated for shape
and finiteness but is not silently modified.

`enforce_stationarity=False` and `enforce_invertibility=False` are explicit
research controls. Diagnostics are still calculated and stored when a constraint
is disabled. A returned result can therefore be marked non-stationary or
non-invertible while also recording that the corresponding constraint was not
enforced.

## Fitted diagnostics

```python
result = model.fit(data, weights)
diagnostic = model.admissibility()

print(result.stationary)
print(result.invertible)
print(result.admissible)
print(result.stability_boundary_distance)
print(result.invertibility_boundary_distance)
print(diagnostic.eigenvalues if hasattr(diagnostic, "eigenvalues") else diagnostic.summary())
```

`KalmanSTARMAResult` records:

- AR spectral radius and limit;
- inverse-MA spectral radius and limit;
- whether each constraint was enabled;
- signed distances to both boundaries;
- joint admissibility.

The full companion matrices and eigenvalues are available from
`model.admissibility()`.

## Likelihood-Hessian inference

`KalmanSTARMA.infer()` evaluates finite-difference stencils under the same two
feasibility rules as fitting. A stencil point entering either enabled penalty
region is rejected instead of differentiating the artificial penalty surface.

`LikelihoodInferenceResult` now reports:

- `stability_boundary_distance`;
- `invertibility_boundary_distance`;
- `minimum_admissibility_distance`.

Curvature close to either boundary should be interpreted cautiously. Reducing
the finite-difference step may avoid a boundary crossing, but it cannot repair
weak identification or genuinely boundary-adjacent inference.

## Numerical scope and limitations

The companion criterion is exact for the finite-dimensional matrix-polynomial
representation implemented here. Version 0.0.11 does not yet provide:

- a smooth bijective stationarity/invertibility parameterization;
- projection of arbitrary user starts onto the nearest admissible model;
- root-by-root confidence intervals;
- delta-method uncertainty for spectral radii or polynomial roots;
- multiplicative seasonal factor admissibility;
- integrated-model maximum-likelihood wrappers;
- sparse or iterative eigenvalue computation for very large networks.

Dense eigenvalue decomposition is currently used. The result API is designed so
that sparse computation can be introduced later without changing the public
statistical definitions.
