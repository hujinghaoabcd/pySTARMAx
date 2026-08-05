# Step 32 handoff: exact-diffuse lag-one covariance

## Development position

- development version: `0.0.32`;
- branch: `agent/exact-diffuse-lag-one-covariance`;
- pull request: PR #32, `Add exact diffuse lag-one covariance reference`;
- base: version 0.0.31 on `main` at merge commit
  `6d4f0eda336f121315769a6910606381676cfe56`.

## Scope

Step 32 establishes an exact dense reference for

\[
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y_{1:T})
\]

under exact diffuse initialization. It covers ordinary and seasonal augmented
state spaces because both are represented by the same `StateSpaceModel` and
`ExactDiffuseFilterResult` contracts.

The public functional entry point is:

```python
result = exact_diffuse_lag_one_covariance(filter_result)
```

The fitted ordinary and seasonal estimators expose:

```python
result = model.smooth_lag_one_covariance()
```

## Exact dense construction

The complete state path is written as

\[
\alpha=b+D\delta+G\xi,
\]

where `delta` contains flat diffuse coordinates and `xi` contains proper
standard-normal coordinates. Exact observations impose

\[
A\delta+B\xi=c.
\]

When every diffuse direction is identified, `A` has full column rank. The
diffuse coordinates are eliminated analytically and the remaining proper
coordinates are conditioned on their equality constraints. If `V` spans the
remaining posterior source space, the conditional state loading is

\[
H=(G-DA^+B)V,
\]

so adjacent covariance is evaluated exactly as

\[
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y)=H_tH_{t+1}^{\mathsf T}.
\]

No finite large-variance initialization and no Monte Carlo approximation are
used by the public covariance routine.

## Validation design

The implementation is independently checked through:

1. a diffuse local-level model with a closed-form final-anchor covariance;
2. exact reduction to the ordinary RTS lag-one covariance when the diffuse
   covariance is zero, including a non-symmetric transition matrix;
3. Monte Carlo cross-covariances from the already validated exact-diffuse
   conditional simulation smoother;
4. the same path comparison for a period-two seasonal augmented state;
5. reconstruction of state-disturbance covariance from lag-one moments and
   comparison with the independent exact information disturbance smoother.

## Current boundary

This stage provides a transparent moderate-sample dense exact reference. It does
not yet claim the memory-linear diffuse `L2` recursion, arbitrary cross-time
covariance, sparse execution, or parameter-uncertainty propagation.

The complete API inventory, CI record, documentation synchronization, and next
stage plan will be finalized after implementation CI.
