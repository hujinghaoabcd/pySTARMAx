# Step 28 handoff: seasonal exact-diffuse smoothing

## Repository position

- development version: `0.0.28`;
- branch: `agent/seasonal-exact-diffuse-smoothing`;
- pull request: PR #28, `Add seasonal exact diffuse smoothing`;
- base: version 0.0.27 on `main` at merge commit
  `66fc0442b561ce257f2653e54bbecfd635c78f09`.

## Delivered API

```python
from pystarmax import SeasonalExactDiffuseKalmanSTARIMA

model = SeasonalExactDiffuseKalmanSTARIMA(...)
result = model.fit(data, weights)

smoothed = model.smooth()
disturbances = model.smooth_innovation_disturbances()
```

Both methods accept optional new original-level data and a numerical tolerance.
Supplying new data starts a fresh exact-diffuse filter under the fitted
parameters. It does not continue from the terminal training posterior.

## Architectural decision

Step 28 does not introduce a seasonal copy of the backward smoother. The
validated exact-diffuse state and disturbance smoothers already operate on the
generic `ExactDiffuseFilterResult` and `StateSpaceModel` contracts.

The 0.0.26 seasonal augmented state satisfies those contracts. Its full
selection matrix maps each primitive innovation into the original-level lag
companion and stationary transformed state. Therefore:

- `exact_diffuse_smoother()` is the state posterior engine;
- `exact_diffuse_disturbance_smoother()` is the primitive/state-disturbance
  posterior engine;
- `seasonal_exact_diffuse_model.py` is a thin fitted facade;
- no finite large-variance approximation is introduced;
- no second seasonal state convention is created.

## Independent mathematical references

### Period-two seasonal bridge

For

\[
y_t=y_{t-2}+\eta_t,
\qquad \eta_t\sim\mathcal N(0,q),
\]

the even and odd subsequences form two independent random walks. For

```text
[0, 10, missing, missing, 4, 14]
```

the exact posterior means are

```text
[0, 10, 2, 12, 4, 14]
```

and the two missing-level variances are `q/2`.

The primitive innovation posterior verifies both timing and unresolved diffuse
uncertainty:

- first transition mean `0`, variance `q`;
- four identified bridge innovations with mean `2`, variance `q/2`.

### Ordinary reduction

With `P=D=Q=0`, the seasonal facade is compared directly with
`ExactDiffuseKalmanSTARIMA`. The tests require agreement of smoothed state means,
state covariances, primitive innovation means, and primitive innovation
covariances.

### Missing and partial observations

Multivariate tests use different observation masks by location. Observed cells
are reproduced, missing cells receive posterior moments, covariance matrices
remain positive semidefinite within tolerance, and the original mask is retained.

## Implementation validation

CI #589, run ID `30951765063`, validated implementation head
`50a80b5cca2fbca188631bed5beffc8925deaf97`:

- 238 tests passed;
- total branch coverage: 87.16%;
- the new fitted facade was fully covered;
- `seasonal_exact_diffuse_mle.py` branch coverage: 83.9%;
- Black, isort, Ruff, and mypy passed;
- diagnostic-reference regeneration was clean;
- strict MkDocs passed;
- sdist, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11–3.14.

The first CI failure was limited to an outdated version assertion and a fitted
fixture whose short missing bridge produced no finite seasonal differences for
MLE initialization. The fixed-parameter closed-form state and innovation tests
already passed. The repair changed only the public-version assertion and fitted
sample length; no implementation formula or numerical tolerance was weakened.

## Deliberate boundaries

Step 28 does not provide:

- lag-one exact-diffuse state covariance during a nontrivial diffuse phase;
- cross-time disturbance covariance;
- seasonal exact-diffuse likelihood inference;
- seasonal exact-diffuse forecast paths or intervals;
- seasonal conditional simulation smoothing;
- parameter-aware posterior paths;
- sparse or parallel execution.

The diffuse `L2` recursion remains a separate research task and must not be
approximated with an arbitrary large finite initialization.

## Remaining roadmap after Step 28

The next core stage is seasonal exact-diffuse likelihood inference and natural
innovation-covariance inference. Forecast paths and intervals should follow in a
separate PR. After Step 28 the inventory contains six major technical
workstreams, eight numbered core milestones, and approximately 13–17
independently reviewable projects after ecosystem work is split.

## Merge checklist

Before merging PR #28:

1. synchronize README, documentation home, project status, roadmap, remaining
   work, citation metadata, navigation, example, method guide, and this handoff;
2. run final CI on the exact synchronized head;
3. record final run number, run ID, test count, and coverage;
4. confirm no workflow helper or generated artifact remains;
5. confirm no review thread or submitted review remains;
6. mark ready and squash-merge;
7. create the seasonal exact-diffuse inference branch from the new `main`.
