from __future__ import annotations

import numpy as np

from pystarmax.exact_seasonal_integrated import (
    build_exact_seasonal_integrated_state_space,
)
from pystarmax.state_space import StateSpaceModel


def test_seasonal_exact_diffuse_preserves_matrix_orientation() -> None:
    transition = np.array([[0.2, 0.3], [0.0, 0.4]])
    design = np.array([[1.0, 0.0], [0.5, 1.0]])
    selection = np.array([[1.0, 0.0], [0.0, 1.0]])
    intercept = np.array([0.1, -0.2])
    transformed = StateSpaceModel(
        transition=transition,
        design=design,
        selection=selection,
        state_intercept=intercept,
        innovation_covariance=np.diag([0.3, 0.5]),
        ar_order=1,
        ma_order=0,
    )

    specification = build_exact_seasonal_integrated_state_space(
        transformed,
        ordinary_integration_order=0,
        seasonal_integration_order=1,
        seasonal_period=2,
    )
    model = specification.model

    np.testing.assert_allclose(
        model.transition[:2, :4],
        np.block([[np.zeros((2, 2)), np.eye(2)]]),
    )
    np.testing.assert_allclose(model.transition[:2, 4:], design @ transition)
    np.testing.assert_allclose(
        model.transition[2:4, :4],
        np.block([[np.eye(2), np.zeros((2, 2))]]),
    )
    np.testing.assert_allclose(model.transition[4:, 4:], transition)
    np.testing.assert_allclose(model.selection[:2], design @ selection)
    np.testing.assert_allclose(model.selection[4:], selection)
    np.testing.assert_allclose(model.state_intercept[:2], design @ intercept)
    np.testing.assert_allclose(model.state_intercept[4:], intercept)
