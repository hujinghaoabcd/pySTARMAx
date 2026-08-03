# Step 13 handoff: Kalman fixed-interval state smoothing

## Repository position

- development version: `0.0.13`;
- branch: `agent/kalman-state-smoothing`;
- pull request: PR #13, `Add Kalman fixed-interval state smoothing`;
- base: version 0.0.12 on `main`;
- authoritative implementation/documentation validation: GitHub Actions CI #304,
  run ID `30852463900`;
- validation result: 119 tests passed and 87.32% branch coverage;
- Black, isort, Ruff, mypy, independent fixture regeneration, strict MkDocs,
  source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11 through 3.14;
- PR surface: 20 formal files and no temporary workflow or diagnostic files.

A final documentation-only commit records these results. Its complete CI run is
the merge gate and is written into the PR description without creating another
validation-record commit.

## Delivered API

### Low-level

```python
smoothed = kalman_smoother(filter_result, rcond=1e-10)
```

### Fitted maximum-likelihood model

```python
smoothed = fitted_model.smooth()
smoothed_new = fitted_model.smooth(new_incomplete_data)
```

### Result

`KalmanSmootherResult` stores:

- `smoothed_state`;
- `smoothed_covariance`;
- `smoothing_gain`;
- `lag_one_covariance`;
- `state_disturbance_mean`;
- `state_disturbance_covariance`;
- `prediction_rank`;
- `used_pseudoinverse`;
- the original `filter_result`.

All public arrays are defensive copies and read-only.

## Mathematical convention

For

\[
\alpha_{t+1}=d+T\alpha_t+w_{t+1},
\]

the RTS gain is

\[
J_t=P_{t|t}T^\top P_{t+1|t}^{+}.
\]

The smoothed moments are

\[
a_{t|T}=a_{t|t}+J_t(a_{t+1|T}-a_{t+1|t}),
\]

\[
P_{t|T}=P_{t|t}+J_t(P_{t+1|T}-P_{t+1|t})J_t^\top.
\]

The stored lag-one covariance is

\[
C_{t,t+1|T}=J_tP_{t+1|T}.
\]

The state-equation disturbance moments are derived from
\(w_{t+1}=\alpha_{t+1}-d-T\alpha_t\).

## Numerical policy

- full-rank predicted covariance uses `numpy.linalg.solve`;
- rank-deficient predicted covariance uses a positive-eigenspace pseudoinverse;
- `rcond` controls the retained eigenspace;
- every transition records numerical rank and pseudoinverse use;
- covariances are symmetrized;
- only tiny negative eigenvalues are projected to zero;
- materially indefinite covariance raises;
- the final smoothed state and covariance equal the final filtered values.

## Important interpretation boundary

`state_disturbance_mean` is the conditional mean of the state disturbance
\(w_t=R\eta_t\). It must not be labelled as the original location-level
innovation \(\eta_t\) unless the selection mapping is explicitly invertible and
the corresponding conditional covariance is derived consistently.

A future original-innovation disturbance smoother should solve the conditional
Gaussian mapping from state disturbances to location innovations. It should not
apply an arbitrary matrix inverse or pseudoinverse to `state_disturbance_mean`
alone.

## Validation references

Tests include:

1. scalar AR(1) one-gap Gaussian bridge with analytic conditional mean and
   variance;
2. a contiguous missing block compared with direct joint-Gaussian conditioning;
3. exact disturbance recovery for fully observed scalar AR(1) data;
4. explicit rank-deficient prediction and pseudoinverse reporting;
5. covariance reduction relative to filtering;
6. fitted-model smoothing on a new incomplete sequence;
7. a one-time-point edge case;
8. immutable arrays and argument validation;
9. the full inherited package suite, including covariance and likelihood
   inference regression tests.

The authoritative coverage job reported:

- 119 tests passed;
- total branch coverage: 87.32%;
- `src/pystarmax/smoothing.py`: 87.1% branch coverage;
- required coverage threshold: 80%.

## Inherited fixes included in this stage

Two 0.0.12 defects became visible when the restored full CI workflow ran:

1. `infer_kalman_starma()` referenced an undefined `observations` local when
   setting `n_locations`; the result now uses the fitted innovation covariance
   dimension.
2. covariance-element indices were inferred by mypy as a fixed one-element
   tuple in the scalar branch; an explicit variable-length tuple annotation now
   covers scalar, diagonal, and full covariance branches.

These fixes are covered by the inherited covariance and likelihood inference
tests and are documented in the Step 12 handoff.

## Files changed

Core:

- `src/pystarmax/smoothing.py`;
- `src/pystarmax/maximum_likelihood.py`;
- `src/pystarmax/__init__.py`;
- `src/pystarmax/likelihood_inference.py`;
- `src/pystarmax/covariance_inference.py`.

Tests:

- `tests/test_state_smoothing.py`;
- formatting alignment in `tests/test_covariance_inference.py`.

Documentation and examples:

- `docs/smoothing.md`;
- `docs/covariance_inference.md`;
- `examples/state_smoothing.py`;
- README, index, navigation, roadmap, project status, citation metadata;
- Step 12 and Step 13 handoffs.

## Next recommended stage

The most direct continuation is original location-level innovation disturbance
smoothing with a mathematically explicit conditional Gaussian derivation.
Alternative high-priority work remains integrated/seasonal MLE wrappers, sparse
state matrices, and smooth admissibility parameterization.

Do not call state disturbances original innovations, and do not add a naive
`pinv(selection) @ state_disturbance_mean` API without deriving its posterior
covariance and null-space behavior.
