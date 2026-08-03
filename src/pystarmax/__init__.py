# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Public package interface for pySTARMAx."""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"
__version__ = "0.0.9"

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
from pystarmax.forecasting import ForecastInterval
from pystarmax.maximum_likelihood import (
    CovarianceType,
    KalmanSTARMA,
    KalmanSTARMAResult,
)
from pystarmax.models import STAR, STARIMA, STARMA, SeasonalSTARIMA
from pystarmax.results import STARMAResult
from pystarmax.seasonal import LagOperator, expand_multiplicative_operators
from pystarmax.simulation import (
    simulate_seasonal_starima,
    simulate_seasonal_starma,
    simulate_starima,
    simulate_starma,
)
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
    "STARMAResult",
    "KalmanSTARMAResult",
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
    "build_starma_state_space",
    "fitted_starma_state_space",
    "kalman_filter",
    "kalman_loglikelihood",
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
