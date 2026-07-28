import numpy as np

from pystarmax import SpatialWeights, lattice_weights, simulate_starma


def test_simulation_is_deterministic() -> None:
    weights = SpatialWeights.from_adjacency(lattice_weights(1, 3))
    first = simulate_starma(
        phi=np.array([[0.4, 0.1]]),
        theta=np.zeros((0, 2)),
        weights=weights,
        n_steps=20,
        burnin=10,
        random_state=7,
    )
    second = simulate_starma(
        phi=np.array([[0.4, 0.1]]),
        theta=np.zeros((0, 2)),
        weights=weights,
        n_steps=20,
        burnin=10,
        random_state=7,
    )
    np.testing.assert_allclose(first, second)
    assert first.shape == (20, 3)
