from __future__ import annotations

import pystarmax
from pystarmax import (
    SeasonalExactDiffuseKalmanSTARIMA,
    SeasonalExactDiffuseKalmanSTARIMAResult,
    infer_seasonal_exact_diffuse_kalman_starima,
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)
from pystarmax.seasonal_exact_diffuse_forecasting import (
    seasonal_exact_diffuse_differenced_forecast_interval as ModuleDifferencedInterval,
)
from pystarmax.seasonal_exact_diffuse_forecasting import (
    seasonal_exact_diffuse_forecast_interval as ModuleInterval,
)
from pystarmax.seasonal_exact_diffuse_forecasting import (
    simulate_seasonal_exact_diffuse_differenced_forecast_paths as ModuleDifferencedPaths,
)
from pystarmax.seasonal_exact_diffuse_forecasting import (
    simulate_seasonal_exact_diffuse_forecast_paths as ModulePaths,
)
from pystarmax.seasonal_exact_diffuse_inference import (
    infer_seasonal_exact_diffuse_kalman_starima as ModuleInference,
)
from pystarmax.seasonal_exact_diffuse_model import (
    SeasonalExactDiffuseKalmanSTARIMA as ModuleEstimator,
)
from pystarmax.seasonal_exact_diffuse_model import (
    SeasonalExactDiffuseKalmanSTARIMAResult as ModuleResult,
)


def test_seasonal_exact_diffuse_public_exports() -> None:
    assert pystarmax.__version__ == "0.0.30"
    assert SeasonalExactDiffuseKalmanSTARIMA is ModuleEstimator
    assert SeasonalExactDiffuseKalmanSTARIMAResult is ModuleResult
    assert infer_seasonal_exact_diffuse_kalman_starima is ModuleInference
    assert seasonal_exact_diffuse_forecast_interval is ModuleInterval
    assert (
        seasonal_exact_diffuse_differenced_forecast_interval
        is ModuleDifferencedInterval
    )
    assert simulate_seasonal_exact_diffuse_forecast_paths is ModulePaths
    assert (
        simulate_seasonal_exact_diffuse_differenced_forecast_paths
        is ModuleDifferencedPaths
    )
    assert hasattr(SeasonalExactDiffuseKalmanSTARIMA, "smooth")
    assert hasattr(
        SeasonalExactDiffuseKalmanSTARIMA,
        "smooth_innovation_disturbances",
    )
    assert hasattr(SeasonalExactDiffuseKalmanSTARIMA, "likelihood_inference")
    assert hasattr(SeasonalExactDiffuseKalmanSTARIMA, "simulate_forecast_paths")
    assert hasattr(
        SeasonalExactDiffuseKalmanSTARIMA,
        "simulate_differenced_forecast_paths",
    )
    assert hasattr(SeasonalExactDiffuseKalmanSTARIMA, "predict_interval")
    assert hasattr(
        SeasonalExactDiffuseKalmanSTARIMA,
        "predict_differenced_interval",
    )
    assert "SeasonalExactDiffuseKalmanSTARIMA" in pystarmax.__all__
    assert "SeasonalExactDiffuseKalmanSTARIMAResult" in pystarmax.__all__
    assert "infer_seasonal_exact_diffuse_kalman_starima" in pystarmax.__all__
    assert "simulate_seasonal_exact_diffuse_forecast_paths" in pystarmax.__all__
    assert (
        "simulate_seasonal_exact_diffuse_differenced_forecast_paths"
        in pystarmax.__all__
    )
    assert "seasonal_exact_diffuse_forecast_interval" in pystarmax.__all__
    assert (
        "seasonal_exact_diffuse_differenced_forecast_interval"
        in pystarmax.__all__
    )
