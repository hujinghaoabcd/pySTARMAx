import numpy as np

from pystarmax import STAR, STARMA, SpatialWeights, lattice_weights, simulate_starma


def test_star_recovers_stable_coefficients() -> None:
    weights = SpatialWeights.from_adjacency(lattice_weights(2, 2))
    series = simulate_starma(
        phi=np.array([[0.45, 0.20]]),
        theta=np.zeros((0, 2)),
        weights=weights,
        n_steps=1200,
        burnin=300,
        intercept=0.25,
        innovation_covariance=0.5,
        random_state=123,
    )
    model = STAR(ar_order=1)
    result = model.fit(series, weights)
    assert result.converged
    assert result.params.shape == (3,)
    np.testing.assert_allclose(result.params[1:], [0.45, 0.20], atol=0.06)
    assert model.predict(steps=4).shape == (4, 4)


def test_starma_fit_returns_complete_result() -> None:
    weights = SpatialWeights.from_adjacency(lattice_weights(1, 3))
    series = simulate_starma(
        phi=np.array([[0.35, 0.15]]),
        theta=np.array([[0.20, 0.05]]),
        weights=weights,
        n_steps=500,
        burnin=200,
        random_state=42,
    )
    result = STARMA(ar_order=1, ma_order=1, max_iter=50).fit(series, weights)
    assert result.params.shape == (5,)
    assert result.coefficients.shape == (5, 4)
    assert result.fitted_values.shape == series.shape
    assert np.isnan(result.residuals[0]).all()
    assert np.isfinite(result.bic)


def test_model_rejects_mismatched_weights() -> None:
    series = np.ones((20, 3))
    weights = SpatialWeights.from_adjacency(lattice_weights(1, 2))
    try:
        STAR().fit(series, weights)
    except ValueError as error:
        assert "locations" in str(error)
    else:
        raise AssertionError("mismatched weights should fail")
