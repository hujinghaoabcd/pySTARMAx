# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Public package interface for pySTARMAx."""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"
__version__ = "0.0.1"

from pystarmax.diagnostics import (
    PortmanteauResult,
    space_time_portmanteau,
    stacf,
    stpacf,
)
from pystarmax.models import STAR, STARMA
from pystarmax.results import STARMAResult
from pystarmax.simulation import simulate_starma
from pystarmax.weights import (
    SpatialWeights,
    distance_weights,
    lattice_weights,
    row_standardize,
)

__all__ = [
    "STAR",
    "STARMA",
    "STARMAResult",
    "SpatialWeights",
    "PortmanteauResult",
    "simulate_starma",
    "stacf",
    "stpacf",
    "space_time_portmanteau",
    "row_standardize",
    "distance_weights",
    "lattice_weights",
]
