# Step 13 handoff: Kalman fixed-interval state smoothing

## Repository position

- development version: `0.0.13`;
- branch: `agent/kalman-state-smoothing`;
- pull request: PR #13, `Add Kalman fixed-interval state smoothing`;
- base: version 0.0.12 on `main`;
- core validation run: GitHub Actions CI #286;
- core validation result: 119 tests passed and 87.32% branch coverage;
- quality, strict documentation, distributions, and the full operating-system
  matrix are revalidated after final documentation changes before merge.

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
