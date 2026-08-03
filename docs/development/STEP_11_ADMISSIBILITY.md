# Step 11 handoff: STARMA admissibility

Updated: 2026-08-04

## Objective

Add one explicit and reusable definition of autoregressive stationarity and
moving-average invertibility for the matrix-polynomial STARMA convention used by
pySTARMAx, then apply that definition consistently to diagnostics, maximum-
likelihood starting values, optimization, fitted results, and likelihood-Hessian
inference.

The implementation preserves non-symmetric spatial-weight orientation and does
not hide a non-admissible result when a user explicitly disables a constraint.

## Model convention

The stationary model is

\[
z_t = c + \sum_{i=1}^{p}A_i z_{t-i}
      + \varepsilon_t + \sum_{j=1}^{q}B_j\varepsilon_{t-j},
\]

with

\[
A_i = \sum_k\phi_{ik}W_k,
\qquad
B_j = \sum_k\theta_{jk}W_k.
\]

The AR companion top row is

```text
[A1, A2, ..., Ap]
```

and stationarity is

```text
rho(C_AR) < 1 - stability_margin.
```

Because the package uses a positive MA sign, the inverse recursion is

\[
\varepsilon_t = r_t - \sum_{j=1}^{q}B_j\varepsilon_{t-j}.
\]

The inverse-MA companion top row is therefore

```text
[-B1, -B2, ..., -Bq]
```

and invertibility is

```text
rho(C_MA_inverse) < 1 - invertibility_margin.
```

This sign is critical. Do not reuse an AR companion with a positive MA top row.

## Public API added

Top-level exports now include:

- `PolynomialKind`;
- `PolynomialAdmissibility`;
- `STARMAAdmissibility`;
- `compose_lag_operators()`;
- `autoregressive_diagnostics()`;
- `moving_average_diagnostics()`;
- `starma_admissibility()`;
- `autoregressive_spectral_radius()`;
- `moving_average_inverse_spectral_radius()`.

A fitted `KalmanSTARMA` model now also provides:

```python
diagnostic = model.admissibility()
```

## Result design

`PolynomialAdmissibility` stores immutable arrays for:

- temporal-lag operator matrices;
- block companion matrix;
- complex companion eigenvalues.

It also stores:

- polynomial kind;
- spectral radius;
- configured limit;
- signed distance `limit - radius`;
- final admissibility decision.

`STARMAAdmissibility` combines an AR diagnostic and an inverse-MA diagnostic and
provides:

- `stationary`;
- `invertible`;
- `admissible`;
- `minimum_distance`;
- a compact summary.

Zero-order polynomials use empty companion matrices and empty eigenvalue vectors,
with spectral radius zero.

## Weight coercion

The public diagnostic layer accepts:

1. `SpatialWeights`, used unchanged;
2. a single NumPy matrix, interpreted under the package convention as a
   non-identity spatial matrix with identity prepended;
3. a matrix sequence, converted through `SpatialWeights.from_matrices()`.

For research reproducibility, prefer passing a named `SpatialWeights` object.
The implementation preserves matrix orientation exactly and does not transpose,
symmetrize, or reorder matrices.

## Kalman maximum-likelihood changes

`KalmanSTARMA.__init__()` now includes:

```python
enforce_invertibility: bool = True
invertibility_margin: float = 1e-6
```

The pre-existing stationarity controls remain:

```python
enforce_stationarity: bool = True
stability_margin: float = 1e-6
```

### Automatic starting values

Automatic conditional-estimator starts are processed by
`_shrink_initial_dynamics()`.

- AR coefficients are iteratively scaled until the AR companion radius lies
  safely below the configured limit.
- MA coefficients are independently scaled until the inverse-MA companion radius
  lies safely below its configured limit.
- Zero-order blocks are skipped.
- Disabled constraints are not applied.
- User-supplied `start_params` are not silently rescaled.

### Objective feasibility

Each likelihood candidate is decoded into AR parameters, MA parameters,
covariance, state-space matrices, AR radius, and inverse-MA radius.

For enabled constraints, squared boundary excesses are accumulated:

```text
excess_AR^2 + excess_MA^2
```

A large feasibility penalty is returned when the total is positive. The
likelihood is evaluated only when all enabled criteria pass.

### Final candidate

After L-BFGS-B returns, the final candidate is decoded again.

- enabled stationarity failure raises `RuntimeError`;
- enabled invertibility failure raises `RuntimeError`;
- disabled criteria do not block return;
- diagnostics are always retained in the result.

## Fitted-result changes

`KalmanSTARMAResult` now records:

- `spectral_radius`;
- `ma_inverse_spectral_radius`;
- `stability_limit`;
- `invertibility_limit`;
- `stationarity_enforced`;
- `invertibility_enforced`.

It exposes:

- `stationary`;
- `invertible`;
- `admissible`;
- `stability_boundary_distance`;
- `invertibility_boundary_distance`.

The summary prints both dynamic criteria and whether each was enforced.

This distinction is deliberate. For example, a user can set
`enforce_invertibility=False`, receive a non-invertible fitted result, and still
see a negative invertibility distance and `invertibility_enforced=False`.

## Likelihood-inference changes

The finite-difference negative log likelihood reconstructs both AR and inverse-MA
spectral radii. It uses the fitted model's two enforcement flags and margins.

Any stencil point entering either enabled feasibility penalty is rejected by the
existing invalid-threshold mechanism. The Hessian never differentiates the
large artificial penalty surface.

`LikelihoodInferenceResult` now includes:

- `stability_boundary_distance`;
- `invertibility_boundary_distance`;
- `minimum_admissibility_distance`.

## Main files

- `src/pystarmax/admissibility.py` — operator composition, companion construction,
  eigensystems, and immutable diagnostic results;
- `src/pystarmax/_maximum_likelihood_model.py` — dual starts, objective penalty,
  final validation, and fitted diagnostic method;
- `src/pystarmax/_maximum_likelihood_result.py` — fitted radii, limits, distances,
  flags, and summaries;
- `src/pystarmax/likelihood_inference.py` — dual-boundary finite differences;
- `src/pystarmax/__init__.py` — public exports and version 0.0.11;
- `tests/test_admissibility.py` — independent scalar, polynomial-root, spatial,
  zero-order, and validation tests;
- `tests/test_maximum_likelihood_admissibility.py` — fitting and enforcement tests;
- `tests/test_likelihood_inference.py` — inference boundary tests;
- `docs/admissibility.md` — user-facing statistical guide;
- `examples/admissibility.py` — diagnostic and fitted-model example.

## Independent tests

The test suite includes:

1. scalar AR(1) radius and configured boundary behavior;
2. scalar MA(1) sign verification using a negative companion entry;
3. scalar MA(2) comparison with reciprocal polynomial-root modulus;
4. non-symmetric two-location spatial operators without transposition;
5. immutable operators, companion matrices, and complex eigenvalues;
6. zero-order AR and MA polynomials with radius zero;
7. joint failure caused independently by AR or MA;
8. matrix-input identity-prepend convention;
9. invalid parameter shape, non-finite value, margin, and weight inputs;
10. scalar MA(1) maximum-likelihood recovery;
11. forced non-invertible optimizer return rejected when enforcement is enabled;
12. the same forced candidate retained and explicitly diagnosed when enforcement
    is disabled;
13. pre-fit and constructor-control validation;
14. dual boundary distances in likelihood inference.

## Final validation

GitHub Actions CI run #269 completed successfully on the final implementation,
test, example, and documentation head:

- 103 tests passed without Python test warnings;
- total branch coverage was 87.48%;
- Black, isort, Ruff, and mypy passed;
- independent diagnostic reference generation produced a clean diff;
- strict MkDocs passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11 through 3.14;
- exactly 20 formal files remained in the PR difference;
- the standard CI workflow matched `main` and was absent from the PR;
- no temporary formatting or diagnostic artifacts remained.

The validation-record edits after run #269 are documentation only. One final
CI repeat is required before marking PR #11 ready and merging it.

## Performance note

The current low-level spectral-radius helpers construct dense companion matrices
and use dense eigenvalue decomposition. During maximum-likelihood optimization,
this occurs for both AR and MA blocks on every objective evaluation.

This is acceptable for the current small and medium reference problems, but it
is a known large-network cost. Do not change the public statistical definition
when adding acceleration. Potential later implementations include:

- direct small-order block operations;
- sparse companion matrices;
- dominant-eigenvalue solvers;
- caching weight-dependent structure;
- compiled operator assembly.

Any accelerated route must preserve exact dense-reference tests and matrix
orientation.

## Deliberate exclusions

0.0.11 does not include:

- a smooth bijective stationarity/invertibility parameterization;
- nearest-admissible projection for arbitrary user starts;
- uncertainty for spectral radii or roots;
- delta-method covariance-element inference;
- multiplicative seasonal factor admissibility;
- integrated or seasonal Kalman MLE wrappers;
- state or disturbance smoothing;
- sparse eigenvalue computation.

## Next-stage decision

The theoretical next target is smooth stability/invertibility parameterization.
A general matrix-polynomial bijection is substantially more delicate than the
current explicit spectral criterion. Do not claim a general solution by applying
scalar `tanh` independently to STARMA coefficients; coefficient-wise bounds do
not guarantee a multivariate companion spectral radius below one.

Recommended decision rule:

1. first investigate a mathematically valid matrix-polynomial parameterization
   with an invertible mapping and stable Jacobian;
2. require scalar ARMA reductions and non-commuting spatial tests;
3. require compatibility with likelihood-Hessian inference;
4. if a general auditable parameterization cannot be completed cleanly in one
   stage, implement delta-method innovation-covariance inference next while
   retaining the explicit feasibility controls.

## Handoff checklist

Before Step 12:

- read `PROJECT_STATUS.md`;
- read `docs/model.md`, `docs/admissibility.md`,
  `docs/maximum_likelihood.md`, and `docs/likelihood_inference.md`;
- inspect `compose_lag_operators()`, `_companion_matrix()`,
  `_shrink_initial_dynamics()`, and the likelihood objective together;
- preserve the positive-MA inverse-recursion sign;
- preserve disabled-constraint diagnostics;
- do not differentiate through feasibility penalties;
- do not substitute coefficient-wise clipping for matrix-polynomial admissibility;
- add independent scalar and non-commuting matrix references before changing the
  admissibility definition.
