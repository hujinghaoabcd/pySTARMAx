from __future__ import annotations

import numpy as np
import pytest

from pystarmax import KalmanSTARMA, SpatialWeights


def identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def simulate_scalar_ar1(
    *,
    phi: float,
    standard_deviation: float,
    n_steps: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    data = np.empty((n_steps, 1), dtype=float)
    data[0, 0] = rng.normal(
        scale=standard_deviation / np.sqrt(1.0 - phi**2)
    )
    for time_index in range(1, n_steps):
        data[time_index, 0] = (
            phi * data[time_index - 1, 0]
            + rng.normal(scale=standard_deviation)
        )
    return data


def test_scalar_white_noise_matches_closed_form_mle() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=1.25, scale=0.7, size=(400, 1))
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
    )

    result = model.fit(data, identity_weights(1))

    expected_mean = float(np.mean(data))
    expected_variance = float(np.mean((data - expected_mean) ** 2))
    assert result.converged
    assert result.intercept == pytest.approx(expected_mean, abs=1e-5)
    assert result.innovation_covariance[0, 0] == pytest.approx(
        expected_variance,
        rel=1e-5,
    )
    assert result.n_params == 2
    assert result.n_observations == data.size
    assert result.aic == pytest.approx(-2.0 * result.log_likelihood + 4.0)
    assert not result.params.flags.writeable
    assert not result.innovation_covariance.flags.writeable
    assert "Kalman maximum-likelihood" in result.summary()


def test_scalar_ar1_recovers_parameters_with_missing_values() -> None:
    data = simulate_scalar_ar1(
        phi=0.55,
        standard_deviation=0.6,
        n_steps=500,
        random_state=7,
    )
    incomplete = data.copy()
    incomplete[20:460:9, 0] = np.nan
    model = KalmanSTARMA(
        ar_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=600,
    )

    result = model.fit(incomplete, identity_weights(1))

    assert result.converged
    assert result.ar_parameters[0, 0] == pytest.approx(0.55, abs=0.10)
    assert result.innovation_covariance[0, 0] == pytest.approx(0.36, abs=0.08)
    assert result.n_observations == int(
        np.count_nonzero(np.isfinite(incomplete))
    )
    assert result.spectral_radius < 1.0
    filtered = model.filter()
    assert filtered is result.filter_result
    assert np.isnan(filtered.innovations[20, 0])


def test_recursive_prediction_uses_last_filtered_state() -> None:
    data = simulate_scalar_ar1(
        phi=0.4,
        standard_deviation=0.5,
        n_steps=300,
        random_state=11,
    )
    model = KalmanSTARMA(
        ar_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
    )
    result = model.fit(data, identity_weights(1))

    predictions = model.predict(steps=3)
    phi = result.ar_parameters[0, 0]
    expected = np.array(
        [
            phi * data[-1, 0],
            phi**2 * data[-1, 0],
            phi**3 * data[-1, 0],
        ]
    )
    np.testing.assert_allclose(predictions[:, 0], expected, atol=1e-8)
    with pytest.raises(ValueError, match="positive"):
        model.predict(steps=0)


@pytest.mark.parametrize("covariance_type", ["diagonal", "full"])
def test_multivariate_covariance_parameterizations(
    covariance_type: str,
) -> None:
    rng = np.random.default_rng(123)
    covariance = np.array([[0.5, 0.22], [0.22, 1.8]], dtype=float)
    data = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=covariance,
        size=700,
    )
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type=covariance_type,  # type: ignore[arg-type]
        include_intercept=False,
        max_iter=600,
    )

    result = model.fit(data, identity_weights(2))

    assert result.converged
    np.testing.assert_allclose(
        np.diag(result.innovation_covariance),
        np.diag(covariance),
        atol=0.15,
    )
    if covariance_type == "diagonal":
        assert result.innovation_covariance[0, 1] == 0.0
        assert result.n_params == 2
    else:
        assert result.innovation_covariance[0, 1] == pytest.approx(
            0.22,
            abs=0.10,
        )
        assert result.n_params == 3
    assert np.all(np.linalg.eigvalsh(result.innovation_covariance) > 0.0)


def test_filter_new_missing_data_and_state_space_access() -> None:
    rng = np.random.default_rng(9)
    data = rng.normal(size=(120, 2))
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="diagonal",
        include_intercept=True,
    )
    model.fit(data, identity_weights(2))
    incomplete = data[:20].copy()
    incomplete[3, 0] = np.nan
    incomplete[8, :] = np.nan

    filtered = model.filter(incomplete)

    assert filtered.n_observations == incomplete.size - 3
    assert filtered.log_likelihood_contributions[8] == 0.0
    assert model.to_state_space().n_locations == 2


def test_start_values_and_input_validation() -> None:
    rng = np.random.default_rng(4)
    data = rng.normal(size=(80, 1))
    weights = identity_weights(1)
    model = KalmanSTARMA(
        ar_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
    )

    with pytest.raises(ValueError, match="exactly 1"):
        model.fit(data, weights, start_params=[0.1, 0.2])
    with pytest.raises(ValueError, match="positive"):
        model.fit(data, weights, start_covariance=-1.0)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.to_state_space()

    mostly_missing = data.copy()
    mostly_missing[:, 0] = np.nan
    mostly_missing[0, 0] = 1.0
    with pytest.raises(ValueError, match="at least two"):
        model.fit(mostly_missing, weights)


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="covariance_type"):
        KalmanSTARMA(covariance_type="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initialization"):
        KalmanSTARMA(initialization="known")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stability_margin"):
        KalmanSTARMA(stability_margin=1.0)
    with pytest.raises(ValueError, match="max_iter"):
        KalmanSTARMA(max_iter=0)
