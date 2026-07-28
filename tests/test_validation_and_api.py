import numpy as np
import pytest

from pystarmax import (
    STAR,
    STARMA,
    SpatialWeights,
    distance_weights,
    lattice_weights,
    simulate_starma,
    space_time_portmanteau,
    stacf,
    stpacf,
)


def test_public_validation_errors() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        STAR().fit(np.ones(10), np.eye(1))
    with pytest.raises(ValueError, match="finite"):
        STAR().fit(np.array([[1.0], [np.nan], [2.0]]), np.eye(1))
    with pytest.raises(ValueError, match="cannot both be zero"):
        STARMA(0, 0)
    with pytest.raises(ValueError, match="max_iter"):
        STARMA(1, 1, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        STARMA(1, 1, tol=0.0)
    with pytest.raises(ValueError, match="ridge"):
        STAR(ridge=-1.0)


def test_weight_constructors_and_errors() -> None:
    identity = np.eye(3)
    weights = SpatialWeights.from_matrices([identity], names=["identity"])
    assert weights.names == ("identity",)
    assert weights.max_order == 0
    assert list(iter(weights))[0].shape == (3, 3)

    prepended = SpatialWeights.from_matrices(
        [np.ones((3, 3)) - identity], names=["neighbors"], prepend_identity=True
    )
    assert prepended.names == ("W0", "neighbors")

    with pytest.raises(ValueError, match="must not be empty"):
        SpatialWeights.from_matrices([])
    with pytest.raises(ValueError, match="square"):
        SpatialWeights.from_adjacency(np.ones((2, 3)))
    with pytest.raises(ValueError, match="positive"):
        lattice_weights(0, 2)
    with pytest.raises(ValueError, match="power"):
        distance_weights(np.array([[0.0], [1.0]]), power=0.0)
    with pytest.raises(ValueError, match="threshold"):
        distance_weights(np.array([[0.0], [1.0]]), threshold=0.0)


def test_simulation_accepts_matrix_sequence_and_covariance() -> None:
    adjacency = lattice_weights(1, 2)
    weights = SpatialWeights.from_adjacency(adjacency)
    data = simulate_starma(
        phi=np.array([[0.2, 0.1]]),
        theta=np.zeros((0, 2)),
        weights=list(weights.matrices),
        n_steps=12,
        burnin=3,
        innovation_covariance=np.array([[1.0, 0.2], [0.2, 1.0]]),
        random_state=np.random.default_rng(3),
    )
    assert data.shape == (12, 2)
    with pytest.raises(ValueError, match="positive"):
        simulate_starma(
            phi=np.array([[0.2, 0.1]]),
            theta=np.zeros((0, 2)),
            weights=weights,
            n_steps=2,
            innovation_covariance=0.0,
        )
    with pytest.raises(ValueError, match="spatial columns"):
        simulate_starma(
            phi=np.array([[0.2]]),
            theta=np.zeros((0, 2)),
            weights=weights,
            n_steps=2,
        )


def test_model_runtime_errors_and_summary() -> None:
    model = STAR()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict()
    with pytest.raises(ValueError, match="positive"):
        model.predict(steps=0)

    weights = SpatialWeights.from_adjacency(lattice_weights(1, 2))
    data = simulate_starma(
        phi=np.array([[0.3, 0.1]]),
        theta=np.zeros((0, 2)),
        weights=weights,
        n_steps=80,
        random_state=9,
    )
    result = model.fit(data, weights)
    text = result.summary()
    assert "pySTARMAx model result" in text
    assert "ar.t1.W0" in text
    assert result.n_params == 3
    with pytest.raises(ValueError, match="positive"):
        model.predict(steps=0)


def test_diagnostic_boundary_errors() -> None:
    weights = SpatialWeights.from_adjacency(lattice_weights(1, 2))
    data = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match="smaller"):
        stacf(data, weights, max_tlag=10)
    with pytest.raises(ValueError, match="positive"):
        stpacf(data, weights, max_tlag=0)
    with pytest.raises(ValueError, match="complete rows"):
        space_time_portmanteau(
            np.array([[np.nan, np.nan], [1.0, 2.0]]), weights, max_tlag=1
        )
