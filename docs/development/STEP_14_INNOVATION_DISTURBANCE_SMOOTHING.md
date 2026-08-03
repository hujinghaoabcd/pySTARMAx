# Step 14 handoff: original innovation disturbance smoothing

## Repository position

- development version: `0.0.14`;
- branch: `agent/innovation-disturbance-smoothing`;
- pull request: PR #14, `Add original innovation disturbance smoothing`;
- base: version 0.0.13 on `main`;
- initial numerical validation: 126 tests passed with 87.06% total branch
  coverage before documentation expansion;
- final quality, documentation, distribution, coverage, and operating-system
  matrix results must be recorded before merge.

## Delivered API

### Low-level

```python
result = innovation_disturbance_smoother(
    state_smoother_result,
    rcond=1e-10,
)
```

### Fitted maximum-likelihood model

```python
result = fitted_model.smooth_innovation_disturbances()
result_new = fitted_model.smooth_innovation_disturbances(new_incomplete_data)
```

### Result object

`InnovationDisturbanceResult` stores:

- `innovation_mean`;
- `innovation_covariance`;
- `conditioning_map`;
- `unresolved_covariance`;
- `process_support_projector`;
- `process_rank`;
- `used_pseudoinverse`;
- `mean_support_residual`;
- `covariance_support_residual`;
- the originating `smoother_result` and `model`.

All public arrays are defensive copies and read-only.

## Mathematical derivation

For

\[
w_t=R\eta_t,
\qquad
\eta_t\sim\mathcal N(0,Q),
\]

define

\[
S=RQR^\top,
\qquad
A=QR^\top S^+.
\]

Then

\[
\eta_t\mid w_t
\sim
\mathcal N\left(
Aw_t,
Q-AS A^\top
\right).
\]

Given RTS state-disturbance posterior moments

\[
\mu_{w,t}=E(w_t\mid y_{1:T}),
\qquad
V_{w,t}=\operatorname{Var}(w_t\mid y_{1:T}),
\]

the law of total expectation and covariance gives

\[
E(\eta_t\mid y_{1:T})=A\mu_{w,t},
\]

\[
\operatorname{Var}(\eta_t\mid y_{1:T})
=Q-AS A^\top+A V_{w,t}A^\top.
\]

This is a conditional Gaussian derivation. It is not equivalent to applying an
unweighted pseudoinverse of `selection` to the state-disturbance mean.

## Null-space policy

`unresolved_covariance = Q - A S A.T` is retained. It represents innovation
variation that cannot be learned from the state disturbance. This term can be
positive when the selection map is non-injective. Silently discarding it would
produce overconfident and generally incorrect posterior uncertainty.

When a scalar innovation is duplicated into several state coordinates, the
process covariance is rank deficient even though the original innovation is
uniquely recoverable. The implementation uses the positive eigenspace of
`S = R Q R.T`, recovers the scalar innovation, and records pseudoinverse use.

## Numerical support diagnostics

Exact state-disturbance posterior moments belong to the range of `S`. The result
reports:

```text
mean_support_residual[t]
    = ||mu_w[t] - Pi_S mu_w[t]||_2

covariance_support_residual[t]
    = ||V_w[t] - Pi_S V_w[t] Pi_S||_F
```

These diagnostics expose floating-point leakage or inconsistent externally
constructed smoother results. The conditioning map uses only the supported
component and does not invent information from unsupported directions.

## Time convention

State-smoother transition index `t` describes

```text
alpha_(t+1) - state_intercept - transition @ alpha_t
```

and therefore innovation result index `t` corresponds to `eta_(t+1)`. A sample
with `T` stored state times produces `T - 1` innovation disturbances. No result
is produced for the disturbance entering the first stored state from the
initial distribution.

## Validation references

Tests include:

1. a non-injective selection with correlated two-dimensional innovations and a
   closed-form unresolved variance;
2. duplicated selection with one uniquely recoverable scalar innovation and a
   rank-deficient process covariance;
3. explicit support-residual diagnostics using deliberately inconsistent state
   moments;
4. scalar AR(1) equality between state and original innovation disturbances;
5. moving-average companion-state reconstruction of both state-injection
   copies;
6. fitted-model smoothing of a new incomplete observation matrix;
7. one-time-point shapes, immutability, and argument validation;
8. the full inherited package test suite on all supported platforms.

## Files introduced or changed

Core:

- `src/pystarmax/innovation_smoothing.py`;
- `src/pystarmax/maximum_likelihood.py`;
- `src/pystarmax/__init__.py`.

Tests:

- `tests/test_innovation_smoothing.py`.

Documentation and examples:

- `docs/innovation_smoothing.md`;
- `examples/innovation_smoothing.py`;
- README, index, navigation, state-smoothing guide, roadmap, and project status;
- Step 13 and Step 14 handoffs;
- `CITATION.cff` version metadata.

## Current limitations

- the map is fixed-parameter and does not propagate coefficient or innovation
  covariance estimation uncertainty;
- posterior covariance is marginal by time; cross-time innovation covariance is
  not exposed;
- the initialization disturbance before the first stored state is unavailable;
- exact diffuse and simulation smoothing are unavailable;
- integrated and seasonal Kalman MLE wrappers are unavailable;
- all matrices are dense.

## Next recommended stage

The highest-value continuation is integrated and multiplicative seasonal
state-space/Kalman maximum-likelihood support. A narrower smoothing continuation
would add cross-time innovation-disturbance covariance and a conditional
simulation smoother.

Do not replace this conditional Gaussian implementation with
`pinv(selection) @ state_disturbance_mean`, and do not remove
`unresolved_covariance` for non-injective selection maps.
