from __future__ import annotations

import pystarmax
from pystarmax import (
    ExactDiffuseLagOneCovarianceResult,
    SeasonalExactDiffuseKalmanSTARIMA,
    SeasonalExactDiffuseKalmanSTARIMAResult,
    SeasonalExactDiffuseSimulationSmootherResult,
    exact_diffuse_lag_one_covariance,
    infer_seasonal_exact_diffuse_kalman_starima,
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    seasonal_exact_diffuse_simulation_smoother,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)
from pystarmax.exact_diffuse_lag_one_covariance import (
    ExactDiffuseLagOneCovarianceResult as ModuleLagOneResult,
)
from pystarmax.exact_diffuse_lag_one_covariance import (
    exact_diffuse_lag_one_covariance as ModuleLagOneCovariance,
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
from pystarmax.seasonal_exact_diffuse_simulation_smoothing import (
    SeasonalExactDiffuseSimulationSmootherResult as ModuleSimulationResult,
)
from pystarmax.seasonal_exact_diffuse_simulation_smoothing import (
    seasonal_exact_diffuse_simulation_smoother as ModuleSimulationSmoother,
)


def test_seasonal_exact_diffuse_public_exports() -> None:
    assert pystarmax.__version__ == "0.0.32"
    assert SeasonalExactDiffuseKalmanSTARIMA is ModuleEstimator
    assert SeasonalExactDiffuseKalmanSTARIMAResult is ModuleResult
    assert SeasonalExactDiffuseSimulationSmootherResult is ModuleSimulationResult
    assert ExactDiffuseLagOneCovarianceResult is ModuleLagOneResult
    assert exact_diffuse_lag_one_covariance is ModuleLagOneCovariance
    assert infer_seasonal_exact_diffuse_kalman_starima is ModuleInference
    assert seasonal_exact_diffuse_simulation_smoother is ModuleSimulationSmoother
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
        "smooth_lag_one_covariance",
    )
    assert hasattr(
        SeasonalExactDiffuseKalmanSTARIMA,
        "simulate_smoothing_paths",
    )
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
    assert "SeasonalExactDiffuseSimulationSmootherResult" in pystarmax.__all__
    assert "ExactDiffuseLagOneCovarianceResult" in pystarmax.__all__
    assert "exact_diffuse_lag_one_covariance" in pystarmax.__all__
    assert "infer_seasonal_exact_diffuse_kalman_starima" in pystarmax.__all__
    assert "seasonal_exact_diffuse_simulation_smoother" in pystarmax.__all__
    assert "simulate_seasonal_exact_diffuse_forecast_paths" in pystarmax.__all__
    assert (
        "simulate_seasonal_exact_diffuse_differenced_forecast_paths"
        in pystarmax.__all__
    )
    assert "seasonal_exact_diffuse_forecast_interval" in pystarmax.__all__
    assert "seasonal_exact_diffuse_differenced_forecast_interval" in pystarmax.__all__
