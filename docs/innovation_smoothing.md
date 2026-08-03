# Original innovation disturbance smoothing

Version 0.0.14 maps Rauch--Tung--Striebel state-disturbance moments back to the
original location-level STARMA innovations without treating the state selection
matrix as if it were necessarily invertible.

## Problem statement

The state equation is

\[
\alpha_t=d+T\alpha_{t-1}+R\eta_t,
\qquad
\eta_t\sim\mathcal N(0,Q),
\]

and the fixed-interval state smoother supplies posterior moments of

\[
w_t=R\eta_t.
\]

The selection matrix \(R\) can be rectangular or rank deficient. In the STARMA
companion representation with moving-average states, the same innovation may be
copied into more than one state block. In a general state-space model, some
innovation directions may not enter the state at all. Therefore
`pinv(R) @ state_disturbance_mean` is not a valid general disturbance smoother:
it ignores the covariance metric \(Q\), does not retain unidentified
uncertainty, and provides no coherent posterior covariance.

## Conditional Gaussian map

Define the process covariance

\[
S=RQR^\top
\]

and the covariance-weighted conditioning map

\[
A=QR^\top S^+.
\]

Here \(S^+\) is an ordinary inverse when \(S\) is full rank and a
positive-eigenspace Moore--Penrose inverse when it is rank deficient.

The conditional distribution of the original innovation given the state
disturbance is

\[
E(\eta_t\mid w_t)=Aw_t,
\]

\[
\operatorname{Var}(\eta_t\mid w_t)
=Q-AS A^\top.
\]

Let the RTS state smoother provide

\[
\mu_{w,t}=E(w_t\mid y_{1:T}),
\qquad
V_{w,t}=\operatorname{Var}(w_t\mid y_{1:T}).
\]

Integrating over the posterior distribution of \(w_t\) gives

\[
E(\eta_t\mid y_{1:T})=A\mu_{w,t},
\]

\[
\operatorname{Var}(\eta_t\mid y_{1:T})
=Q-AS A^\top+A V_{w,t}A^\top.
\]

The first covariance term is stored as `unresolved_covariance`. It preserves the
innovation uncertainty that cannot be identified from \(R\eta_t\). It is zero
when the original innovation is uniquely determined by the state disturbance,
but can be positive for a non-injective selection map.

## Time indexing

`KalmanSmootherResult.state_disturbance_mean[t]` describes the transition from
state time \(t\) to \(t+1\):

\[
w_{t+1}=\alpha_{t+1}-d-T\alpha_t.
\]

Consequently, `InnovationDisturbanceResult.innovation_mean[t]` stores posterior
moments for \(\eta_{t+1}\). With \(T\) observations, the result contains
\(T-1\) innovation disturbances. The initialization disturbance before the first
stored state is not reconstructed.

## Fitted-model usage

```python
incomplete = observations.copy()
incomplete[20:24, 1] = np.nan
incomplete[80, :] = np.nan

result = fitted_model.smooth_innovation_disturbances(
    incomplete,
    rcond=1e-10,
)

print(result.innovation_mean)
print(result.innovation_covariance)
print(result.unresolved_covariance)
print(result.process_rank)
print(result.used_pseudoinverse)
```

Calling the method without `data` uses the fitted training sample. Supplying a
new observation matrix uses the fitted coefficients and covariance, while the
forward filter preserves partial-location and fully missing-row semantics.

## Low-level usage

```python
from pystarmax import (
    innovation_disturbance_smoother,
    kalman_filter,
    kalman_smoother,
)

filtered = kalman_filter(observations, state_space)
state_result = kalman_smoother(filtered)
innovation_result = innovation_disturbance_smoother(state_result)
```

## Result fields

`InnovationDisturbanceResult` stores immutable arrays and diagnostics:

- `innovation_mean`: posterior means, shape `(time - 1, locations)`;
- `innovation_covariance`: posterior marginal covariance at each transition;
- `conditioning_map`: \(A=QR^\top(RQR^\top)^+\);
- `unresolved_covariance`: \(Q-AS A^\top\);
- `process_support_projector`: projector onto the retained positive eigenspace
  of \(S\);
- `process_rank` and `used_pseudoinverse`;
- `mean_support_residual` and `covariance_support_residual`;
- the originating `smoother_result` and state-space `model`.

`unresolved_variance` returns the trace of `unresolved_covariance`.

## Support diagnostics

Exact state-disturbance moments lie in the range of \(S=RQR^\top\). Floating
point operations can introduce tiny components outside that support. The result
therefore reports, for every transition,

\[
\lVert \mu_w-\Pi_S\mu_w\rVert_2
\]

and

\[
\lVert V_w-\Pi_SV_w\Pi_S\rVert_F,
\]

where \(\Pi_S\) is the retained positive-eigenspace projector. These diagnostics
make numerical support leakage visible. The conditional map itself naturally
uses only the supported component; it does not fabricate innovation information
from unsupported state directions.

## Rank-deficient policy

The eigenspace threshold is `rcond * largest_eigenvalue`. The implementation:

1. symmetrizes \(S\);
2. rejects materially negative eigenvalues;
3. uses `numpy.linalg.solve` when \(S\) is full rank;
4. otherwise constructs the Moore--Penrose inverse from retained positive
   eigenpairs;
5. records rank and pseudoinverse use;
6. projects only numerically tiny negative posterior covariance eigenvalues to
   zero and raises for material indefiniteness.

## Validation references

The implementation is checked against independently calculable cases:

- a non-injective selection with correlated two-dimensional innovations, where
  the unresolved null-space variance has a closed form;
- duplicated state injection, where one scalar innovation is copied into two
  state coordinates and must be recovered uniquely through a rank-deficient
  process covariance;
- deliberately inconsistent state moments to verify support-residual reporting;
- a fully observed scalar AR(1), where original innovations equal state
  disturbances exactly;
- a moving-average companion state, where recovered innovation moments reproduce
  both copies of the state disturbance;
- fitted-model, one-time-point, immutability, and argument-validation cases.

## Current limits

- parameters are treated as fixed; estimator uncertainty is not propagated;
- only marginal posterior covariance for each innovation time is exposed;
  cross-time innovation-disturbance covariance is not yet returned;
- the initialization disturbance before the first stored state is unavailable;
- exact diffuse smoothing and simulation smoothing are unavailable;
- direct Kalman maximum likelihood remains stationary and non-seasonal;
- matrices are dense, so large spatial systems remain expensive.
