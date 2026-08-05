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

Passing explicit data to the fitted method starts a fresh exact-diffuse filter
under the fitted parameters. It does not continue from the training terminal
posterior.

## Public result

`ExactDiffuseLagOneCovarianceResult` exposes immutable:

```python
result.lag_one_covariance
result.observation_lag_one_covariance
result.state_disturbance_mean
result.state_disturbance_covariance
```

The orientation is:

\[
\texttt{lag\_one\_covariance}[t]
=
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y_{1:T}).
\]

This matches the existing ordinary RTS smoother and is intentionally not
symmetrized. Non-symmetric state transitions retain their supplied orientation.

The result also stores the exact smoother result, source dimensions and ranks,
source-support residual, discrepancies against independently computed marginal
moments and state-disturbance moments, and any positive-semidefinite numerical
correction. A one-time-point sample returns zero-length transition arrays.

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
diffuse coordinates are eliminated analytically:

\[
\delta=A^+(c-B\xi).
\]

After substituting this expression, the remaining equality constraints on the
proper source vector are imposed. If `V` spans the posterior proper-source null
space, the conditional state-path loading is

\[
H=(G-DA^+B)V.
\]

Adjacent covariance is therefore evaluated exactly as

\[
\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y)
=H_tH_{t+1}^{\mathsf T}.
\]

No finite large-variance initialization and no Monte Carlo approximation are
used by the public covariance routine.

## State-disturbance identity

For

\[
u_t=\alpha_{t+1}-c-T\alpha_t,
\]

and

\[
C_t=\operatorname{Cov}(\alpha_t,\alpha_{t+1}\mid Y),
\]

Step 32 reconstructs

\[
\operatorname{Var}(u_t\mid Y)
=
P_{t+1}+TP_tT^{\mathsf T}
-C_t^{\mathsf T}T^{\mathsf T}
-TC_t.
\]

It compares this result with the separate exact information-form disturbance
smoother. A material mismatch raises `LinAlgError`. This is an independent check
of adjacent-time indexing and matrix orientation.

## Proper-posterior policy

The routine requires:

```python
filter_result.final_diffuse_rank == 0
```

If any diffuse direction remains unresolved, it raises `RuntimeError`. It never
substitutes an arbitrary finite covariance.

## Independent validation

The tests cover:

1. a diffuse local-level final-anchor model with closed-form marginal and
   adjacent covariances;
2. exact reduction to ordinary RTS lag-one covariance when initial diffuse
   covariance is zero;
3. a non-symmetric transition matrix with partially observed locations;
4. empirical adjacent cross-covariance from the separately implemented
   exact-diffuse conditional simulation smoother;
5. the same conditional-path comparison for a period-two seasonal augmented
   state;
6. state-disturbance covariance reconstruction against independent exact
   information-form disturbance smoothing;
7. ordinary and seasonal fitted training-result reuse and fresh new-data
   initialization;
8. one-time-point empty transitions, unresolved diffuse rank, validation,
   immutability, and public exports.

## CI history

Initial CI #656 found:

- one invalid immutability assertion that expected NumPy to reject changing the
  write flag on an owned empty array;
- Black formatting in the new test file.

The result arrays were already initially read-only. The invalid assertion was
removed from the empty-array case and replaced by a direct write attempt on a
nonempty immutable covariance. No numerical formula, tolerance, or reference
value changed.

A temporary branch-only workflow pinned Black 26.5.1. It applied one exact
expression-formatting change, committed it, and deleted itself. The formal PR
diff contains no workflow file.

Authoritative implementation CI #660, run ID `30964497447`, passed on exact head
`37ddfb4920ff1af37d7a25140efbb71c6aeb6550`:

- 266 tests passed;
- total branch coverage: 87.40%;
- new lag-one covariance module branch coverage: 89.7%;
- Black, isort, Ruff, and mypy passed;
- diagnostic-reference regeneration was clean;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11–3.14.

A synchronized documentation CI and a final record gate remain required on the
exact merge candidate.

## Deliberate boundary

Step 32 provides a transparent moderate-sample dense exact reference. It does
not claim:

- the memory-linear exact diffuse `L2` recursion;
- arbitrary non-adjacent state covariance;
- cross-time state-disturbance or primitive-innovation covariance;
- sparse or chunked source conditioning;
- parameter-uncertainty propagation.

The dense reference is intended to become the numerical oracle for the future
recursive implementation.

## Next stage

Create a separate 0.0.33 project for the memory-linear exact diffuse `L2`
recursion. The recursive output must match the 0.0.32 dense oracle for:

- genuine diffuse closed forms;
- zero-diffuse ordinary RTS reduction;
- non-symmetric transitions;
- missing and partially observed measurements;
- ordinary and seasonal augmented states;
- reconstructed state-disturbance moments.

Arbitrary cross-time disturbance and primitive-innovation covariance should be
implemented only after recursive/dense equivalence is established.
