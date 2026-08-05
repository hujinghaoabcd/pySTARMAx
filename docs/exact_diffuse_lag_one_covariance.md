# Exact-diffuse lag-one covariance

Version 0.0.32 adds an exact dense reference for adjacent-time posterior state
covariance under exact diffuse initialization.

For a state-space model

\[
\alpha_{t+1}=c+T\alpha_t+R\eta_t,
\qquad
Y_t=Z\alpha_t,
\]

the returned array follows the existing ordinary smoother convention:

\[
C_t=
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y_{1:T}),
\qquad t=0,\ldots,T-2.
\]

Therefore:

```python
result.lag_one_covariance[t]
```

has left index at time `t` and right index at time `t + 1`. This orientation is
important for non-symmetric transitions.

## Public APIs

For an existing exact-diffuse filter result:

```python
from pystarmax import exact_diffuse_lag_one_covariance

result = exact_diffuse_lag_one_covariance(filter_result)
```

The fitted ordinary and seasonal exact-diffuse estimators expose the same
operation:

```python
ordinary_result = ordinary_model.smooth_lag_one_covariance()
seasonal_result = seasonal_model.smooth_lag_one_covariance()
```

Passing explicit data to a fitted method starts a fresh exact-diffuse filter
under the fitted parameters:

```python
new_result = ordinary_model.smooth_lag_one_covariance(new_level_data)
```

It does not continue from the training terminal posterior.

## Exact source representation

The dense reference reuses the same source representation as conditional
simulation smoothing. Stack the complete state path and write

\[
\alpha=b+D\delta+G\xi,
\]

where:

- `b` is the deterministic propagated state path;
- `delta` contains the flat initial diffuse coordinates;
- `xi` contains proper standard-normal coordinates for the finite initial
  covariance and all process innovations;
- `D` and `G` are the corresponding path-loading matrices.

Every finite observed cell imposes an exact linear constraint. After stacking
those constraints:

\[
A\delta+B\xi=c.
\]

A proper posterior exists only when all diffuse directions are identified by
the observed sample. Then `A` has full column rank and

\[
\delta=A^+(c-B\xi).
\]

Substitution gives

\[
\alpha=
\left(b+DA^+c\right)
+
\left(G-DA^+B\right)\xi.
\]

The remaining equality constraints on `xi` are imposed by projecting onto the
left-null space of `A`. Let `V` span the posterior null space of those proper
constraints. The conditional state loading is

\[
H=\left(G-DA^+B\right)V.
\]

Consequently, for any two times,

\[
\operatorname{Cov}(\alpha_t,\alpha_s\mid Y_{1:T})
=H_tH_s^{\mathsf T}.
\]

Version 0.0.32 evaluates only adjacent pairs `s = t + 1` and the marginal
blocks needed for independent verification.

## No large-variance approximation

The method never replaces the diffuse covariance with

\[
\kappa I
\]

for a large finite `kappa`. Diffuse coordinates remain algebraic flat
directions until the observations identify them. The result is also not a Monte
Carlo estimate: conditional simulation is used only as an independent test
reference.

If

```python
filter_result.final_diffuse_rank != 0
```

then the function raises `RuntimeError` because a proper finite conditional
source distribution has not been defined.

## Result contract

`ExactDiffuseLagOneCovarianceResult` contains immutable arrays:

```python
result.lag_one_covariance
result.observation_lag_one_covariance
result.state_disturbance_mean
result.state_disturbance_covariance
```

Shapes are:

```text
lag_one_covariance:             (T - 1, state_dim, state_dim)
observation_lag_one_covariance: (T - 1, n_locations, n_locations)
state_disturbance_mean:         (T - 1, state_dim)
state_disturbance_covariance:   (T - 1, state_dim, state_dim)
```

For one observation time, all four transition-indexed arrays have length zero.

The result also records:

- identified diffuse rank;
- equality-conditioning rank;
- proper and posterior source dimensions;
- source-support residual;
- discrepancies against the independent information-form state smoother;
- discrepancy against independent exact state-disturbance smoothing;
- any numerical positive-semidefinite covariance correction.

## Observation-scale covariance

For time-invariant design matrix `Z`, the adjacent observation covariance is

\[
\operatorname{Cov}(Y_t,Y_{t+1}\mid Y_{1:T})
=ZC_tZ^{\mathsf T}.
\]

This is returned as `observation_lag_one_covariance`. It includes all model
locations, including cells that were missing in the observed sample.

## State-disturbance reconstruction

Define the state-equation disturbance

\[
u_t=
\alpha_{t+1}-c-T\alpha_t.
\]

Using

\[
P_t=\operatorname{Var}(\alpha_t\mid Y_{1:T})
\]

and

\[
C_t=\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y_{1:T}),
\]

its covariance is

\[
\operatorname{Var}(u_t\mid Y_{1:T})
=
P_{t+1}+TP_tT^{\mathsf T}
-C_t^{\mathsf T}T^{\mathsf T}
-TC_t.
\]

The implementation reconstructs this quantity and compares it with the
independent information-form exact-diffuse disturbance smoother. A material
disagreement raises `LinAlgError`. This check is sensitive to both time indexing
and matrix orientation.

## Ordinary and seasonal models

The routine operates on `ExactDiffuseFilterResult`, so the same calculation
applies to:

- ordinary exact-diffuse STARIMA states;
- complete seasonal exact-diffuse augmented states;
- fixed state-space models supplied directly by users.

For a seasonal model, `lag_one_covariance` describes the complete augmented
state, including the ordinary-seasonal inverse-differencing companion and the
stationary transformed-state block. It is not restricted to original
observations.

## Independent validation

The 0.0.32 tests use five separate references:

1. **Diffuse local-level closed form.** For a random walk observed only at the
   final time, the smoothed marginal and adjacent covariances are available
   analytically.
2. **Ordinary RTS reduction.** With zero diffuse covariance, the exact routine
   matches the existing Rauch–Tung–Striebel lag-one covariance exactly,
   including for a non-symmetric transition and partial observations.
3. **Conditional-path Monte Carlo.** Empirical adjacent cross-covariances from
   the independently implemented exact-diffuse simulation smoother converge to
   the analytic dense result.
4. **Seasonal augmented-state Monte Carlo.** The same comparison is performed
   for a period-two seasonal integrated state.
5. **Disturbance identity.** State-disturbance covariance reconstructed from
   adjacent moments matches direct exact information-form disturbance
   smoothing.

## Computational boundary

This implementation materializes complete path loadings and dense equality
constraints. It is intended as a transparent moderate-sample numerical
reference and as an oracle for later recursive implementations.

Version 0.0.32 does not yet provide:

- the memory-linear exact diffuse `L2` recursion;
- arbitrary non-adjacent state covariance;
- cross-time primitive-innovation covariance;
- sparse or chunked source conditioning;
- parameter-uncertainty propagation.

A future recursive implementation must agree with this dense reference before
it can replace it as the scalable execution path.
