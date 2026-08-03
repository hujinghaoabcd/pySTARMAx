from __future__ import annotations

import numpy as np
import pytest

from pystarmax.admissibility import (
    autoregressive_diagnostics,
    autoregressive_spectral_radius,
    compose_lag_operators,
    moving_average_diagnostics,
    moving_average_inverse_spectral_radius,
    starma_admissibility,
)
from pystarmax.weights import SpatialWeights


def identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_scalar_ar1_stationarity_and_boundary() -> None:
    weights = identity_weights(1)
    stable = autoregressive_diagnostics([[0.8]], weights, margin=0.01)
    boundary = autoregressive_diagnostics([[0.99]], weights, margin=0.01)

    assert stable.order == 1
    assert stable.n_locations == 1
    assert stable.spectral_radius == pytest.approx(0.8)
    assert stable.limit == pytest.approx(0.99)
    assert stable.distance == pytest.approx(0.19)
    assert stable.admissible
    assert not boundary.admissible
    assert boundary.distance == pytest.approx(0.0)
    assert autoregressive_spectral_radius([[0.8]], weights) == pytest.approx(0.8)
    assert "AR stationarity" in stable.summary()


def test_scalar_ma1_invertibility_uses_negative_companion() -> None:
    weights = identity_weights(1)
    diagnostic = moving_average_diagnostics([[0.75]], weights)
    negative = moving_average_diagnostics([[-0.75]], weights)
    noninvertible = moving_average_diagnostics([[1.05]], weights)

    np.testing.assert_allclose(diagnostic.companion_matrix, [[-0.75]])
    np.testing.assert_allclose(negative.companion_matrix, [[0.75]])
    assert diagnostic.spectral_radius == pytest.approx(0.75)
    assert negative.spectral_radius == pytest.approx(0.75)
    assert diagnostic.admissible
    assert negative.admissible
    assert not noninvertible.admissible
    assert moving_average_inverse_spectral_radius([[0.75]], weights) == pytest.approx(
        0.75
    )
    assert "MA invertibility" in diagnostic.summary()


def test_scalar_ma2_matches_inverse_polynomial_roots() -> None:
    weights = identity_weights(1)
    theta_1 = 0.35
    theta_2 = -0.2
    diagnostic = moving_average_diagnostics([[theta_1], [theta_2]], weights)
    polynomial_roots = np.roots([theta_2, theta_1, 1.0])
    expected_radius = float(np.max(1.0 / np.abs(polynomial_roots)))

    np.testing.assert_allclose(
        diagnostic.companion_matrix,
        [[-theta_1, -theta_2], [1.0, 0.0]],
    )
    assert diagnostic.spectral_radius == pytest.approx(expected_radius)
    assert diagnostic.admissible == (expected_radius < diagnostic.limit)


def test_spatial_operator_orientation_is_preserved() -> None:
    identity = np.eye(2, dtype=float)
    asymmetric = np.array([[0.0, 1.0], [0.25, 0.0]], dtype=float)
    weights = SpatialWeights(
        matrices=(identity, asymmetric),
        names=("W0", "W1"),
    )
    parameters = np.array([[0.4, 0.2], [0.1, -0.05]], dtype=float)

    operators = compose_lag_operators(parameters, weights)
    diagnostic = autoregressive_diagnostics(parameters, weights)

    np.testing.assert_allclose(operators[0], 0.4 * identity + 0.2 * asymmetric)
    np.testing.assert_allclose(operators[1], 0.1 * identity - 0.05 * asymmetric)
    np.testing.assert_allclose(diagnostic.operators, operators)
    np.testing.assert_allclose(diagnostic.companion_matrix[:2, :2], operators[0])
    np.testing.assert_allclose(diagnostic.companion_matrix[:2, 2:4], operators[1])
    np.testing.assert_allclose(diagnostic.companion_matrix[2:4, :2], identity)
    assert not diagnostic.operators.flags.writeable
    assert not diagnostic.companion_matrix.flags.writeable
    assert not diagnostic.eigenvalues.flags.writeable


def test_zero_orders_are_admissible_with_radius_zero() -> None:
    weights = identity_weights(3)
    joint = starma_admissibility(
        np.empty((0, 1)),
        np.empty((0, 1)),
        weights,
    )

    assert joint.autoregressive.order == 0
    assert joint.moving_average.order == 0
    assert joint.autoregressive.companion_matrix.shape == (0, 0)
    assert joint.moving_average.eigenvalues.shape == (0,)
    assert joint.stationary
    assert joint.invertible
    assert joint.admissible
    assert joint.minimum_distance == pytest.approx(1.0 - 1e-6)
    assert "Jointly admissible: True" in joint.summary()


def test_joint_admissibility_can_fail_on_either_polynomial() -> None:
    weights = identity_weights(1)
    unstable = starma_admissibility([[1.1]], [[0.2]], weights)
    noninvertible = starma_admissibility([[0.2]], [[1.1]], weights)

    assert not unstable.stationary
    assert unstable.invertible
    assert not unstable.admissible
    assert noninvertible.stationary
    assert not noninvertible.invertible
    assert not noninvertible.admissible
    assert unstable.minimum_distance < 0.0
    assert noninvertible.minimum_distance < 0.0


def test_matrix_input_uses_existing_identity_prepend_convention() -> None:
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    operators = compose_lag_operators([[0.3, 0.1]], adjacency)

    np.testing.assert_allclose(operators[0], 0.3 * np.eye(2) + 0.1 * adjacency)


def test_admissibility_validation() -> None:
    weights = identity_weights(2)
    with pytest.raises(ValueError, match="shape"):
        autoregressive_diagnostics([[0.2, 0.1]], weights)
    with pytest.raises(ValueError, match="finite"):
        moving_average_diagnostics([[np.nan]], weights)
    with pytest.raises(ValueError, match="between zero and one"):
        autoregressive_diagnostics([[0.2]], weights, margin=0.0)
    with pytest.raises(ValueError, match="must not be empty"):
        compose_lag_operators([], [])
    with pytest.raises(TypeError, match="weights must be"):
        compose_lag_operators([[0.2]], object())
