# Rolling-origin interval evaluation

Version 0.0.7 adds `rolling_origin_evaluate()` for out-of-sample assessment of
point forecasts and prediction intervals. The evaluator repeatedly creates a
fresh model, fits only observations available at one forecast origin, generates
an interval, and stores the subsequent observations for scoring.

## Expanding-window evaluation

```python
from pystarmax import STARMA, rolling_origin_evaluate

result = rolling_origin_evaluate(
    lambda: STARMA(ar_order=1, ma_order=1),
    series,
    weights,
    initial_window=180,
    horizon=6,
    step=6,
    interval_method="conditional",
    level=0.95,
    interval_kwargs={"n_simulations": 1000},
    random_state=42,
)
```

At origin `t`, the default expanding window fits rows `[0, t)`. Only complete
forecast horizons are retained, so every stored origin contributes the same
number of future steps and locations.

## Fixed-length rolling window

```python
result = rolling_origin_evaluate(
    lambda: STARMA(ar_order=1, ma_order=1),
    series,
    weights,
    initial_window=180,
    window_size=120,
    horizon=6,
    step=6,
    interval_method="conditional",
    interval_kwargs={"n_simulations": 1000},
    random_state=42,
)
```

With `window_size=120`, origin `t` fits only rows `[t-120, t)`. This is useful
when parameter stability over the complete historical sample is doubtful.

## Bootstrap interval evaluation

```python
result = rolling_origin_evaluate(
    lambda: STARMA(ar_order=1, ma_order=1),
    series,
    weights,
    initial_window=180,
    horizon=3,
    step=12,
    interval_method="bootstrap",
    interval_kwargs={
        "n_bootstrap": 200,
        "bootstrap_method": "residual",
        "include_future_innovations": True,
    },
    random_state=42,
)
```

Bootstrap rolling evaluation refits the model repeatedly inside every forecast
origin. It is therefore substantially more expensive than conditional-innovation
evaluation. The same model factory is used at every origin, so model orders and
spatial weights remain fixed unless the factory itself implements a selection
procedure.

## Result arrays

`RollingOriginResult` stores read-only arrays with shape
`(origins, horizon, locations)`:

- `observed`: realized future observations;
- `mean`: point forecasts returned by the interval method;
- `lower` and `upper`: interval endpoints;
- `covered`: whether each observation lies inside its interval;
- `widths`: interval widths;
- `scores`: elementwise central interval scores.

`origins` contains the integer time index of every first forecasted row.

## Aggregate metrics

```python
metrics = result.metrics()
print(metrics.empirical_coverage)
print(metrics.coverage_gap)
print(metrics.average_width)
print(metrics.mean_interval_score)
print(metrics.mae)
print(metrics.rmse)

for horizon, values in enumerate(result.metrics_by_horizon(), start=1):
    print(horizon, values.empirical_coverage, values.mean_interval_score)
```

`IntervalMetrics` reports:

- nominal and empirical coverage;
- signed coverage gap, `empirical - nominal`;
- absolute coverage error;
- average interval width;
- mean central interval score;
- point-forecast MAE and RMSE;
- number of scalar forecasts summarized.

Coverage alone is not sufficient because arbitrarily wide intervals can attain
high coverage. Width and interval score should be reported with coverage.

## Central interval score

For nominal coverage `1-alpha`, lower endpoint `L`, upper endpoint `U`, and
observation `y`, `interval_score()` implements the Winkler interval score:

```text
(U - L)
+ (2 / alpha) * (L - y)  when y < L
+ (2 / alpha) * (y - U)  when y > U
```

Lower scores are better. The score rewards narrow intervals when they cover and
penalizes misses in proportion to their distance outside the interval.

## Reproducibility

A single NumPy random generator controls the evaluation. Every origin receives a
deterministically derived integer seed, so repeating the same configuration with
the same `random_state` reproduces interval simulations without reusing an
identical random stream at all origins.

## Interpretation limits

- Rolling-origin evaluation measures performance only for the supplied series,
  origins, horizons, and spatial weights.
- Overlapping horizons create dependent forecast errors; the current aggregate
  metrics do not attach standard errors to their means.
- Empirical coverage can be noisy when the number of origins or locations is
  small.
- Version 0.0.7 evaluates and diagnoses calibration but does not automatically
  rescale or conformalize interval endpoints.
- Missing observations are not supported by the current estimators or evaluator.
