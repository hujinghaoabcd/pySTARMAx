from __future__ import annotations

import numpy as np
import pytest

from pystarmax.exact_seasonal_integrated import (
    build_exact_seasonal_integrated_state_space,
)
from pystarmax.state_space import StateSpaceModel


def test_seasonal_exact_diffuse_initial_arrays_are_immutable() -> None:
    transformed = StateSpaceModel(
        transition=np.array([[0.4]]),
        design=np.array([[1.0]]),
        selection=np.array([[1.0]]),
        state_intercept=np.array([0.1]),
        innovation_covariance=np.array([[0.3]]),
        ar_order=1,
        ma_order=0,
    )
    specification = build_exact_seasonal_integrated_state_space(
        transformed,
        ordinary_integration_order=1,
        seasonal_integration_order=1,
        seasonal_period=2,
    )

    for values in (
        specification.polynomial_coefficients,
        specification.initial_state,
        specification.initial_covariance,
        specification.initial_diffuse_covariance,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = 0.0
