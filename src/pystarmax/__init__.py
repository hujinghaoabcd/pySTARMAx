# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Public package interface for pySTARMAx."""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"
__version__ = "0.0.32"

from pystarmax.admissibility import (
    PolynomialAdmissibility,
    PolynomialKind,
    STARMAAdmissibility,
    autoregressive_diagnostics,
    autoregressive_spectral_radius,
    compose_lag_operators,
    moving_average_diagnostics,
    moving_average_inverse_spectral_radius,
    starma_admissibility,
)
from pystarmax.covariance_inference import (
    InnovationCovarianceInference,
    InnovationCovarianceTransform,
    delta_method_covariance,
    innovation_covariance_delta_inference,
    innovation_covariance_transform,
)
from pystarmax.diagnostics import (
    PortmanteauResult,
    space_time_portmanteau,
    stacf,
    stcov,
    stpacf,
    stpacf_regression,
    stpacf_yule_walker,
)
from pystarmax.differencing import (
    CombinedDifferencingState,
    DifferencingState,
    SeasonalDifferencingState,
    combined_difference,
    differencing_coefficients,
    ordinary_difference,
    restore_fitted_values,
    seasonal_difference,
)
from pystarmax.evaluation import (
    IntervalMetrics,
    RollingOriginResult,
    interval_score,
    rolling_origin_evaluate,
)
from pystarmax.exact_diffuse import (
    ExactDiffuseFilterResult,
    exact_diffuse_filter,
    exact_diffuse_loglikelihood,
)
from pystarmax.exact_diffuse_disturbance_smoothing import (
    ExactDiffuseDisturbanceResult,
    exact_diffuse_disturbance_smoother,
)
from pystarmax.exact_diffuse_forecasting import (
    exact_diffuse_forecast_interval,
    simulate_exact_diffuse_forecast_paths,
)
from pystarmax.exact_diffuse_inference import infer_exact_diffuse_kalman_starima
from pystarmax.exact_diffuse_lag_one_covariance import (
    ExactDiffuseLagOneCovarianceResult,
    exact_diffuse_lag_one_covariance,
)
from pystarmax.exact_diffuse_model import (
    ExactDiffuseKalmanSTARIMA,
    ExactDiffuseKalmanSTARIMAResult,
)
from pystarmax.exact_diffuse_simulation_smoothing import (
    ExactDiffuseSimulationSmootherResult,
    exact_diffuse_simulation_smoother,
)
from pystarmax.exact_diffuse_smoothing import (
    ExactDiffuseSmootherResult,
    exact_diffuse_smoother,
)
from pystarmax.exact_integrated import (
    ExactIntegratedStateSpace,
    build_exact_integrated_state_space,
    exact_integrated_filter,
    exact_integrated_loglikelihood,
)
from pystarmax.exact_seasonal_integrated import (
    ExactSeasonalIntegratedStateSpace,
    build_exact_seasonal_integrated_state_space,
    exact_seasonal_integrated_filter,
    exact_seasonal_integrated_loglikelihood,
)
from pystarmax.forecasting import ForecastInterval
from pystarmax.innovation_smoothing import (
    InnovationDisturbanceResult,
    innovation_disturbance_smoother,
)
from pystarmax.integrated_forecasting import KalmanSTARIMA
from pystarmax.integrated_maximum_likelihood import KalmanSTARIMAResult
from pystarmax.kalman_forecasting import (
    integrated_kalman_forecast_interval,
    inverse_forecast_paths,
    kalman_forecast_interval,
    simulate_kalman_forecast_paths,
)
from pystarmax.likelihood_inference import (
    FiniteDifferenceCurvature,
    LikelihoodInferenceResult,
    finite_difference_curvature,
    finite_difference_hessian,
    infer_kalman_starma,
)
from pystarmax.maximum_likelihood import (
    CovarianceType,
    KalmanSTARMA,
    KalmanSTARMAResult,
)
from pystarmax.models import STAR, STARIMA, STARMA, SeasonalSTARIMA
from pystarmax.results import STARMAResult
from pystarmax.seasonal import LagOperator, expand_multiplicative_operators
from pystarmax.seasonal_exact_diffuse_forecasting import (
    seasonal_exact_diffuse_differenced_forecast_interval,
    seasonal_exact_diffuse_forecast_interval,
    simulate_seasonal_exact_diffuse_differenced_forecast_paths,
    simulate_seasonal_exact_diffuse_forecast_paths,
)
from pystarmax.seasonal_exact_diffuse_inference import (
    infer_seasonal_exact_diffuse_kalman_starima,
)
from pystarmax.seasonal_exact_diffuse_model import (
    SeasonalExactDiffuseKalmanSTARIMA,
    SeasonalExactDiffuseKalmanSTARIMAResult,
)
from pystarmax.seasonal_exact_diffuse_simulation_smoothing import (
    SeasonalExactDiffuseSimulationSmootherResult,
    seasonal_exact_diffuse_simulation_smoother,
)
from pystarmax.seasonal_forecasting import SeasonalKalmanSTARIMA
from pystarmax.seasonal_likelihood_inference import infer_seasonal_kalman_starima
from pystarmax.seasonal_maximum_likelihood import (
    SeasonalKalmanAdmissibility,
    SeasonalKalmanSTARIMAResult,
)
from pystarmax.simulation import (
    simulate_seasonal_starima,
    simulate_seasonal_starma,
    simulate_starima,
    simulate_starma,
)
from pystarmax.smoothing import KalmanSmootherResult, kalman_smoother
from pystarmax.state_space import (
    Initialization,
    KalmanFilterResult,
    StateSpaceModel,
    build_starma_state_space,
    fitted_starma_state_space,
    kalman_filter,
    kalman_loglikelihood,
)
from pystarmax.weights import (
    SpatialWeights,
    distance_weights,
    lattice_weights,
    row_standardize,
)

__all__ = [
    "STAR",
    "STARMA",
    "STARIMA",
    "SeasonalSTARIMA",
    "KalmanSTARMA",
    "KalmanSTARIMA",
    "SeasonalKalmanSTARIMA",
    "SeasonalExactDiffuseKalmanSTARIMA",
    "ExactDiffuseKalmanSTARIMA",
    "STARMAResult",
    "KalmanSTARMAResult",
    "KalmanSTARIMAResult",
    "SeasonalKalmanSTARIMAResult",
    "SeasonalExactDiffuseKalmanSTARIMAResult",
    "ExactDiffuseKalmanSTARIMAResult",
    "ExactDiffuseSmootherResult",
    "ExactDiffuseDisturbanceResult",
    "ExactDiffuseLagOneCovarianceResult",
    "ExactDiffuseSimulationSmootherResult",
    "SeasonalExactDiffuseSimulationSmootherResult",
    "SeasonalKalmanAdmissibility",
    "KalmanSmootherResult",
    "InnovationDisturbanceResult",
    "LikelihoodInferenceResult",
    "FiniteDifferenceCurvature",
    "InnovationCovarianceInference",
    "InnovationCovarianceTransform",
    "ExactDiffuseFilterResult",
    "ExactIntegratedStateSpace",
    "ExactSeasonalIntegratedStateSpace",
    "PolynomialAdmissibility",
    "STARMAAdmissibility",
    "PolynomialKind",
    "ForecastInterval",
    "IntervalMetrics",
    "RollingOriginResult",
    "StateSpaceModel",
    "KalmanFilterResult",
    "Initialization",
    "CovarianceType",
    "DifferencingState",
    "SeasonalDifferencingState",
    "CombinedDifferencingState",
    "SpatialWeights",
    "PortmanteauResult",
    "simulate_starma",
    "simulate_starima",
    "simulate_seasonal_starma",
    "simulate_seasonal_starima",
    "ordinary_difference",
    "seasonal_difference",
    "combined_difference",
    "differencing_coefficients",
    "restore_fitted_values",
    "interval_score",
    "rolling_origin_evaluate",
    "simulate_kalman_forecast_paths",
    "kalman_forecast_interval",
    "integrated_kalman_forecast_interval",
    "inverse_forecast_paths",
    "simulate_exact_diffuse_forecast_paths",
    "exact_diffuse_forecast_interval",
    "simulate_seasonal_exact_diffuse_forecast_paths",
    "simulate_seasonal_exact_diffuse_differenced_forecast_paths",
    "seasonal_exact_diffuse_forecast_interval",
    "seasonal_exact_diffuse_differenced_forecast_interval",
    "exact_diffuse_filter",
    "exact_diffuse_loglikelihood",
    "exact_diffuse_smoother",
    "exact_diffuse_disturbance_smoother",
    "exact_diffuse_lag_one_covariance",
    "exact_diffuse_simulation_smoother",
    "seasonal_exact_diffuse_simulation_smoother",
    "infer_exact_diffuse_kalman_starima",
    "infer_seasonal_exact_diffuse_kalman_starima",
    "build_exact_integrated_state_space",
    "exact_integrated_filter",
    "exact_integrated_loglikelihood",
    "build_exact_seasonal_integrated_state_space",
    "exact_seasonal_integrated_filter",
    "exact_seasonal_integrated_loglikelihood",
    "finite_difference_curvature",
    "finite_difference_hessian",
    "infer_kalman_starma",
    "infer_seasonal_kalman_starima",
    "innovation_covariance_transform",
    "innovation_covariance_delta_inference",
    "delta_method_covariance",
    "compose_lag_operators",
    "autoregressive_diagnostics",
    "moving_average_diagnostics",
    "starma_admissibility",
    "autoregressive_spectral_radius",
    "moving_average_inverse_spectral_radius",
    "build_starma_state_space",
    "fitted_starma_state_space",
    "kalman_filter",
    "kalman_loglikelihood",
    "kalman_smoother",
    "innovation_disturbance_smoother",
    "LagOperator",
    "expand_multiplicative_operators",
    "stcov",
    "stacf",
    "stpacf",
    "stpacf_yule_walker",
    "stpacf_regression",
    "space_time_portmanteau",
    "row_standardize",
    "distance_weights",
    "lattice_weights",
]
