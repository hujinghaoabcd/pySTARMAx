import numpy as np

from pystarmax import (
    SpatialWeights,
    lattice_weights,
    space_time_portmanteau,
    stacf,
    stpacf,
)


def test_stacf_and_stpacf_shapes() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 4))
    weights = SpatialWeights.from_adjacency(lattice_weights(2, 2))
    acf = stacf(data, weights, max_tlag=4)
    pacf = stpacf(data, weights, max_tlag=3)
    assert acf.shape == (5, 2)
    assert pacf.shape == (3, 2)
    assert list(acf.columns) == ["W0", "W1"]


def test_portmanteau_result_is_finite() -> None:
    rng = np.random.default_rng(7)
    residuals = rng.normal(size=(120, 3))
    weights = SpatialWeights.from_adjacency(lattice_weights(1, 3))
    result = space_time_portmanteau(residuals, weights, max_tlag=5)
    assert np.isfinite(result.statistic)
    assert 0.0 <= result.p_value <= 1.0
