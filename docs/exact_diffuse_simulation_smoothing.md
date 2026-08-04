# Exact diffuse simulation smoothing

Version 0.0.25 adds dense conditional simulation of complete latent state paths
for an already filtered exact diffuse linear Gaussian model.

The implementation is deliberately separate from forecast simulation. Forecast
paths begin at the terminal filtering posterior and simulate future states.
Simulation smoothing instead draws the complete state trajectory over the
observed sample conditional on all finite observations.

## Public API

Low-level use starts from an `ExactDiffuseFilterResult`:

```python
from pystarmax import exact_diffuse_simulation_smoother

paths = exact_diffuse_simulation_smoother(
    exact_filter_result,
    n_simulations=2000,
    random_state=2026,
)
```

A fitted exact diffuse STARIMA model exposes the same operation:

```python
paths = fitted_model.simulate_smoothing_paths(
    n_simulations=2000,
    random_state=2026,
)

new_paths = fitted_model.simulate_smoothing_paths(
    new_level_observations,
    n_simulations=2000,
    random_state=2026,
)
```

Supplying new data starts a new exact diffuse initialization, matching the
existing `filter()`, `smooth()`, and disturbance-smoothing contracts. It does
not continue from the training sample's terminal posterior.

## Coordinate representation

Let the first predicted state be represented by

\[
\alpha_1 = a_1 + B_d\delta + B_f z_0,
\]

where

- \(\delta\) contains flat diffuse coordinates;
- \(z_0\sim\mathcal N(0,I)\) contains finite Gaussian coordinates;
- \(B_dB_d^\top=P_{\infty,1}\);
- \(B_fB_f^\top=P_{\ast,1}\).

Later process innovations are represented as

\[
\eta_t=L_Q z_t,
\qquad z_t\sim\mathcal N(0,I),
\qquad L_QL_Q^\top=Q.
\]

Stacking the finite initial coordinates and later innovation coordinates gives
one proper Gaussian source vector \(z\). Every state can then be written as

\[
\alpha_t=b_t+D_t\delta+G_tz.
\]

No arbitrary finite diffuse scale is introduced.

## Observation constraints

For every observed location-time cell,

\[
y_{t,i}=Z_i\alpha_t,
\]

so the complete observed sample imposes linear equality constraints

\[
A_d\delta+A_fz=r.
\]

When the final diffuse rank is zero, the observed sample identifies every
initial diffuse direction. The implementation obtains a full-column
pseudoinverse of \(A_d\), eliminates \(\delta\), and projects the remaining
constraints into the left null space of \(A_d\):

\[
N_dA_fz=N_dr.
\]

The proper source vector is therefore a standard normal vector conditioned on
linear equalities. If

\[
Cz=d,
\]

an SVD gives a conditional mean and an orthonormal null-space basis. Conditional
samples are drawn only in that supported subspace and mapped back to complete
state paths.

## Why this is exact diffuse rather than a large-variance approximation

The diffuse coordinates use a flat prior and are removed analytically after the
observations identify them. The algorithm never substitutes

\[
P_{\ast,1}+\kappa P_{\infty,1}
\]

for a large finite value of \(\kappa\). It also does not depend on an unavailable
exact diffuse lag-one covariance recursion.

The method is exact for the package's current linear Gaussian state-space
contract with deterministic observation equation and a proper posterior after
the final observation.

## Returned result

`ExactDiffuseSimulationSmootherResult` contains immutable:

- `state_paths` with shape `(simulation, time, state)`;
- `observation_paths` with shape `(simulation, time, location)`;
- analytically reconstructed posterior state means and marginal covariances;
- the existing exact diffuse information-smoother result;
- identified diffuse rank;
- conditioning rank;
- proper and posterior source-space dimensions;
- maximum observation-constraint residual;
- maximum mean and covariance discrepancies against the information smoother;
- numerical `rcond` and tolerance.

Observed cells are reproduced up to floating-point error. Missing cells retain
their conditional uncertainty.

## Independent consistency check

The dense source-coordinate representation independently produces

\[
E(\alpha_t\mid Y)
\]

and

\[
\operatorname{Var}(\alpha_t\mid Y).
\]

Before returning paths, the implementation compares these quantities with the
existing exact diffuse information smoother. Material disagreement raises
instead of silently returning paths from an inconsistent posterior.

This comparison is stronger than checking Monte Carlo sample moments because it
compares two deterministic constructions of the same marginal posterior.

## Closed-form random-walk bridge

For

\[
y_t=y_{t-1}+\eta_t,
\qquad \eta_t\sim\mathcal N(0,\sigma^2),
\]

with observed endpoints \(y_1=a\), \(y_3=b\) and missing middle level,

\[
y_2\mid y_1=a,y_3=b
\sim
\mathcal N\left(
\frac{a+b}{2},
\frac{\sigma^2}{2}
\right).
\]

The validation suite checks the simulated middle-state distribution against
this reference while requiring the two observed endpoints to be reproduced
exactly.

## Numerical policy

- covariance factors use symmetric eigendecomposition;
- floating-point-scale negative eigenvalues are excluded;
- materially indefinite covariance raises;
- SVD ranks use the explicit `rcond` argument;
- observation-support residuals are checked explicitly;
- unresolved final diffuse rank raises;
- all returned numerical arrays are read-only;
- an integer seed or NumPy `Generator` gives reproducible draws.

## Complexity

This is intentionally a dense reference implementation. It constructs source
loadings across the complete sample and performs SVDs on observation-constraint
matrices. Memory and time can therefore grow quickly with sample length, state
dimension, location count, and innovation rank.

The dense algorithm prioritizes a transparent, independently verifiable exact
posterior over large-scale execution. Sparse or sequential simulation smoothing
remains a later task.

## Deliberate boundaries

Version 0.0.25 does not provide:

- primitive innovation paths or state-disturbance paths;
- exact diffuse lag-one state covariance;
- cross-time disturbance covariance tensors;
- seasonal exact diffuse augmentation;
- parameter-estimation uncertainty;
- separate measurement-noise draws;
- sparse or parallel conditional simulation;
- continuation from the terminal posterior of another data segment.

Primitive innovation marginal moments remain available from
`exact_diffuse_disturbance_smoother()`. Forecast simulation remains available
from the exact diffuse forecast-interval API. These are distinct inferential
objects and are not re-labelled as simulation smoothing.
