# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Public package interface for pySTARMAx."""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"
__version__ = "0.0.4"

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
    ordinary_difference,
    seasonal_difference,
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
    "STARMAResult",
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
