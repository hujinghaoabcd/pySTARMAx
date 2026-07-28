import numpy as np
import pytest

from pystarmax import SpatialWeights, distance_weights, lattice_weights, row_standardize


def test_row_standardize_preserves_zero_rows() -> None:
    matrix = np.array([[0.0, 2.0], [0.0, 0.0]])
    result = row_standardize(matrix)
    np.testing.assert_allclose(result, [[0.0, 1.0], [0.0, 0.0]])


def test_higher_order_lattice_weights_are_valid() -> None:
    adjacency = lattice_weights(2, 3)
    weights = SpatialWeights.from_adjacency(adjacency, max_order=2)
    assert weights.n_locations == 6
    assert len(weights) == 3
    np.testing.assert_allclose(weights[0], np.eye(6))
    assert np.all(np.diag(weights[1]) == 0)
    assert np.all(np.diag(weights[2]) == 0)


def test_distance_weights_threshold() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    weights = distance_weights(coordinates, threshold=2.0)
    np.testing.assert_allclose(weights[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(weights[2], [0.0, 0.0, 0.0])


def test_spatial_weights_are_immutable() -> None:
    weights = SpatialWeights.from_adjacency(lattice_weights(1, 2))
    with pytest.raises(ValueError):
        weights[0][0, 0] = 2.0
