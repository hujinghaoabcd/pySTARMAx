# Step 07 handoff: rolling-origin interval evaluation

Updated: 2026-08-03

## Goal

Add an out-of-sample evaluation layer for the conditional and bootstrap
prediction intervals introduced in versions 0.0.5 and 0.0.6. The layer compares
interval methods without changing model estimation or forecast APIs.

## Public API

### `interval_score()`

Computes the central Winkler interval score elementwise. It validates identical
array shapes, finite values, ordered endpoints, and a nominal level strictly
between zero and one.

### `IntervalMetrics`

Frozen and slotted aggregate result with:

- nominal coverage;
- empirical coverage;
- signed coverage gap;
- absolute coverage error;
- average width;
- mean interval score;
- point-forecast MAE and RMSE;
- scalar forecast count.

### `RollingOriginResult`

Frozen and slotted result with read-only arrays shaped
`(origins, horizon, locations)`:

- observed outcomes;
- forecast means;
- lower and upper endpoints;
- derived coverage indicators, widths, and scores;
- aggregate metrics overall and by forecast horizon.

### `rolling_origin_evaluate()`

Accepts a zero-argument model factory, `(time, location)` observations, spatial
weights, an initial training window, horizon, origin step, and optional fixed
window length. At every origin it:

1. selects the expanding or fixed-length training sample;
2. creates and fits a fresh model;
3. derives a deterministic origin-specific seed;
4. calls `predict_interval()` or `predict_bootstrap_interval()`;
5. stores the realized observations and interval arrays.

The function rejects incomplete final horizons and prevents `interval_kwargs`
from overriding `steps`, `level`, or `random_state`.

## Statistical decisions

- Coverage is reported with width and interval score; coverage alone can reward
  unnecessarily wide intervals.
- The interval score is proper for the specified central interval and penalizes
  misses according to their distance beyond the endpoint.
- Metrics are available both pooled and by forecast horizon because calibration
  and sharpness often deteriorate with lead time.
- Expanding windows are the default; fixed windows are explicit through
  `window_size`.
- Overlapping forecast horizons are allowed, but version 0.0.7 does not estimate
  standard errors for dependent score averages.
- The release diagnoses calibration; it does not automatically alter endpoints.

## Files introduced or updated

- `src/pystarmax/evaluation.py`
- `tests/test_evaluation.py`
- `examples/rolling_origin_evaluation.py`
- `docs/evaluation.md`
- `docs/index.md`
- `src/pystarmax/__init__.py`
- `mkdocs.yml`
- `README.md`
- `docs/roadmap.md`
- `CITATION.cff`
- `PROJECT_STATUS.md`

## Focused validation

The focused tests cover:

- exact Winkler score values inside, below, and above an interval;
- invalid shapes, levels, endpoints, and metric values;
- immutable metrics and result arrays;
- pooled and horizon-specific aggregation;
- expanding-window origin construction;
- fixed-length rolling windows;
- conditional and bootstrap method dispatch;
- deterministic origin seed derivation;
- reserved keyword rejection;
- missing interval-method rejection.

## Final validation

GitHub Actions CI run #155 completed successfully on the complete implementation:

- 69 tests passed;
- total branch coverage was 89.33%;
- Black, isort, Ruff, and mypy passed;
- exact diagnostic fixture regeneration produced a clean diff;
- strict MkDocs build passed;
- source distribution, wheel, and Twine checks passed;
- Ubuntu, Windows, and macOS passed on Python 3.11, 3.12, 3.13, and 3.14.

CI run #157 repeated the full workflow after final documentation and handoff
updates and also completed successfully. The README status was then mechanically
updated to 0.0.7 by a self-deleting workflow; the present user-authored handoff
commit triggers the final branch-head confirmation.

## Known limitations

- no standard errors or confidence intervals for empirical coverage and mean scores;
- overlapping-origin dependence is not adjusted;
- no automatic calibration, conformalization, or endpoint rescaling;
- no model comparison significance tests;
- no persistence helper for large result cubes;
- bootstrap rolling evaluation executes serially and can be expensive;
- missing observations remain unsupported by the estimator and evaluator.

## Next development stage

The next core-statistical priority is exact state-space/Kalman likelihood with a
missing-observation pathway. A smaller performance stage can alternatively add
parallel execution for bootstrap replications and rolling origins without
changing public numerical semantics.
