import numpy as np
import pytest

from pystarmax import (
    STAR,
    STARIMA,
    STARMA,
    SeasonalSTARIMA,
    SpatialWeights,
    simulate_seasonal_starima,
    simulate_seasonal_starma,
    simulate_starima,
    simulate_starma,
)
from pystarmax.bootstrap import (
    draw_bootstrap_innovations,
    finite_centered_residuals,
    restore_bootstrap_series,
    validate_bootstrap_arguments,
)


def _identity_weights(n_locations: int = 2) -> SpatialWeights:
    return SpatialWeights.from_matrices(
        [np.eye(n_locations, dtype=float)],
        names=["W0"],
    )


def _assert_interval(interval: object, shape: tuple[int, int]) -> None:
    assert interval.shape == shape  # type: ignore[attr-defined]
    assert np.all(np.isfinite(interval.mean))  # type: ignore[attr-defined]
    assert np.all(interval.lower <= interval.upper)  # type: ignore[attr-defined]


def test_centered_residual_vectors_preserve_joint_rows() -> None:
    residuals = np.array(
        [
            [np.nan, np.nan],
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    centered = finite_centered_residuals(residuals)
    np.testing.assert_allclose(centered.mean(axis=0), 0.0)

    first = draw_bootstrap_innovations(
        bootstrap_method="residual",
        residuals=residuals,
        covariance=np.eye(2),
        n_simulations=3,
        steps=5,
        random_state=42,
    )
    second = draw_bootstrap_innovations(
        bootstrap_method="residual",
        residuals=residuals,
        covariance=np.eye(2),
        n_simulations=3,
        steps=5,
        random_state=42,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[..., 1], 10.0 * first[..., 0])


def test_parametric_bootstrap_and_argument_validation() -> None:
    draws = draw_bootstrap_innovations(
        bootstrap_method="parametric",
        residuals=np.ones((3, 2)),
        covariance=np.array([[1.0, 0.4], [0.4, 1.5]]),
        n_simulations=4,
        steps=3,
        random_state=7,
    )
    assert draws.shape == (4, 3, 2)
    with pytest.raises(ValueError, match="residual.*parametric"):
        draw_bootstrap_innovations(
            bootstrap_method="unknown",
            residuals=np.ones((3, 2)),
            covariance=np.eye(2),
            n_simulations=2,
            steps=2,
            random_state=0,
        )
    with pytest.raises(ValueError, match="at least two"):
        validate_bootstrap_arguments(
            n_bootstrap=1,
            max_attempts=None,
            bootstrap_method="residual",
        )
    with pytest.raises(ValueError, match="max_attempts"):
        validate_bootstrap_arguments(
            n_bootstrap=4,
            max_attempts=3,
            bootstrap_method="residual",
        )


def test_recursive_bootstrap_series_restoration() -> None:
    observed = np.arange(10.0, 16.0)[:, None]
    transformed = np.arange(1.0, 6.0)[:, None]
    restored = restore_bootstrap_series(
        transformed,
        observed,
        ordinary_order=1,
    )
    expected = np.array([10.0, 11.0, 13.0, 16.0, 20.0, 25.0])
    np.testing.assert_allclose(restored[:, 0], expected)
    np.testing.assert_allclose(np.diff(restored[:, 0]), transformed[:, 0])


def test_starma_bootstrap_is_reproducible() -> None:
    weights = _identity_weights()
    series = simulate_starma(
        phi=np.array([[0.35]]),
        theta=np.empty((0, 1)),
        weights=weights,
        n_steps=70,
        burnin=60,
        innovation_covariance=np.array([[1.0, 0.25], [0.25, 0.8]]),
        random_state=11,
    )
    model = STARMA(ar_order=1, include_intercept=False)
    model.fit(series, weights)

    first = model.predict_bootstrap_interval(
        steps=3,
        n_bootstrap=4,
        random_state=19,
    )
    second = model.predict_bootstrap_interval(
        steps=3,
        n_bootstrap=4,
        random_state=19,
    )
    _assert_interval(first, (3, 2))
    np.testing.assert_allclose(first.lower, second.lower)
    np.testing.assert_allclose(first.upper, second.upper)
    assert "residual bootstrap" in first.method


def test_starma_parameter_only_and_star_parametric_routes() -> None:
    weights = _identity_weights()
    series = simulate_starma(
        phi=np.array([[0.25]]),
        theta=np.empty((0, 1)),
        weights=weights,
        n_steps=60,
        burnin=50,
        random_state=3,
    )
    model = STARMA(ar_order=1, include_intercept=False)
    model.fit(series, weights)
    parameter_only = model.predict_bootstrap_interval(
        steps=2,
        n_bootstrap=3,
        include_future_innovations=False,
        random_state=5,
    )
    _assert_interval(parameter_only, (2, 2))
    assert "parameter uncertainty" in parameter_only.method

    star = STAR(ar_order=1, include_intercept=False)
    star.fit(series, weights)
    parametric = star.predict_bootstrap_interval(
        steps=2,
        n_bootstrap=2,
        bootstrap_method="parametric",
        random_state=6,
    )
    _assert_interval(parametric, (2, 2))
    assert "parametric bootstrap" in parametric.method


def test_starima_bootstrap_restores_original_scale() -> None:
    weights = _identity_weights()
    series = simulate_starima(
        phi=np.array([[0.25]]),
        theta=np.empty((0, 1)),
        weights=weights,
        n_steps=65,
        integration_order=1,
        burnin=50,
        random_state=13,
    )
    model = STARIMA(
        ar_order=1,
        integration_order=1,
        include_intercept=False,
    )
    model.fit(series, weights)
    interval = model.predict_bootstrap_interval(
        steps=3,
        n_bootstrap=3,
        random_state=17,
    )
    _assert_interval(interval, (3, 2))
    np.testing.assert_allclose(interval.mean, model.predict(steps=3))


def test_linear_seasonal_starima_bootstrap() -> None:
    weights = _identity_weights()
    series = simulate_seasonal_starima(
        phi=np.array([[0.2]]),
        seasonal_phi=np.empty((0, 1)),
        theta=np.empty((0, 1)),
        seasonal_theta=np.empty((0, 1)),
        weights=weights,
        seasonal_period=4,
        n_steps=70,
        seasonal_integration_order=1,
        burnin=50,
        random_state=23,
    )
    model = SeasonalSTARIMA(
        ar_order=1,
        seasonal_integration_order=1,
        seasonal_period=4,
        include_intercept=False,
    )
    model.fit(series, weights)
    interval = model.predict_bootstrap_interval(
        steps=4,
        n_bootstrap=3,
        include_future_innovations=False,
        random_state=29,
    )
    _assert_interval(interval, (4, 2))


def test_multiplicative_seasonal_bootstrap_route() -> None:
    weights = _identity_weights()
    series = simulate_seasonal_starma(
        phi=np.array([[0.15]]),
        seasonal_phi=np.array([[0.1]]),
        theta=np.empty((0, 1)),
        seasonal_theta=np.empty((0, 1)),
        weights=weights,
        seasonal_period=3,
        n_steps=45,
        burnin=40,
        random_state=31,
    )
    model = SeasonalSTARIMA(
        ar_order=1,
        seasonal_ar_order=1,
        seasonal_integration_order=0,
        seasonal_period=3,
        include_intercept=False,
        max_iter=200,
    )
    model.fit(series, weights)
    interval = model.predict_bootstrap_interval(
        steps=2,
        n_bootstrap=2,
        require_convergence=False,
        max_attempts=4,
        random_state=37,
    )
    _assert_interval(interval, (2, 2))


def test_bootstrap_requires_a_fitted_model() -> None:
    model = STARMA(ar_order=1)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_bootstrap_interval(
            n_bootstrap=2,
            random_state=0,
        )
