import numpy as np
import pytest

from pystarmax import (
    STARIMA,
    STARMA,
    CombinedDifferencingState,
    DifferencingState,
    SeasonalDifferencingState,
    SeasonalSTARIMA,
    SpatialWeights,
    combined_difference,
    expand_multiplicative_operators,
    seasonal_difference,
    simulate_seasonal_starima,
    simulate_seasonal_starma,
    simulate_starma,
)


def _identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights.from_matrices(
        [np.eye(n_locations, dtype=float)], names=["W0"]
    )


def test_seasonal_difference_and_forecast_inversion() -> None:
    data = np.array([[1.0], [4.0], [5.0], [9.0], [8.0], [15.0]])
    differenced, state = seasonal_difference(data, order=1, period=2)
    np.testing.assert_allclose(differenced, data[2:] - data[:-2])
    assert state.order == 1
    assert state.period == 2
    np.testing.assert_allclose(state.histories[0], data[-2:])
    assert not state.histories[0].flags.writeable

    forecast = np.array([[2.0], [3.0], [4.0]])
    expected = np.array([[10.0], [18.0], [14.0]])
    np.testing.assert_allclose(state.inverse_forecast(forecast), expected)
    np.testing.assert_allclose(state.inverse_forecast(forecast), expected)


def test_second_order_seasonal_state_round_trip() -> None:
    time = np.arange(14.0)
    full = (0.3 * time**2 + np.sin(time)).reshape(-1, 1)
    train_length = 9
    _, train_state = seasonal_difference(full[:train_length], order=2, period=2)
    full_differenced, _ = seasonal_difference(full, order=2, period=2)
    future_differences = full_differenced[train_length - 4 :]
    restored = train_state.inverse_forecast(future_differences)
    np.testing.assert_allclose(restored, full[train_length:])


def test_combined_difference_round_trip() -> None:
    time = np.arange(20.0)
    full = (0.2 * time**2 + np.cos(time / 2.0)).reshape(-1, 1)
    train_length = 13
    _, train_state = combined_difference(
        full[:train_length],
        ordinary_order=1,
        seasonal_order=1,
        seasonal_period=3,
    )
    full_differenced, _ = combined_difference(
        full,
        ordinary_order=1,
        seasonal_order=1,
        seasonal_period=3,
    )
    future_differences = full_differenced[train_length - 4 :]
    restored = train_state.inverse_forecast(future_differences)
    np.testing.assert_allclose(restored, full[train_length:])
    assert train_state.ordinary_order == 1
    assert train_state.seasonal_order == 1
    assert train_state.seasonal_period == 3


def test_seasonal_difference_validation() -> None:
    data = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError, match="positive"):
        seasonal_difference(data, order=1, period=0)
    with pytest.raises(ValueError, match="times period"):
        seasonal_difference(data, order=3, period=2)
    with pytest.raises(ValueError, match="one matrix"):
        SeasonalDifferencingState(
            order=1,
            period=2,
            n_locations=1,
            histories=(),
        )
    with pytest.raises(ValueError, match="shape"):
        SeasonalDifferencingState(
            order=1,
            period=2,
            n_locations=1,
            histories=(np.zeros((3, 1)),),
        )
    with pytest.raises(ValueError, match="share locations"):
        CombinedDifferencingState(
            ordinary=DifferencingState(
                order=0,
                n_locations=1,
                anchors=(),
            ),
            seasonal=SeasonalDifferencingState(
                order=0,
                period=2,
                n_locations=2,
                histories=(),
            ),
        )


def test_multiplicative_operator_uses_seasonal_left_matrix_order() -> None:
    identity = np.eye(2)
    ordinary_matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
    seasonal_matrix = np.array([[0.0, 0.0], [1.0, 0.0]])
    weights = SpatialWeights.from_matrices(
        [identity, ordinary_matrix, seasonal_matrix],
        names=["W0", "WA", "WS"],
    )
    ordinary = np.array([[0.0, 1.0, 0.0]])
    seasonal = np.array([[0.0, 0.0, 1.0]])

    ar_terms = expand_multiplicative_operators(
        ordinary,
        seasonal,
        weights,
        seasonal_period=4,
        kind="ar",
    )
    ma_terms = expand_multiplicative_operators(
        ordinary,
        seasonal,
        weights,
        seasonal_period=4,
        kind="ma",
    )
    ar_cross = next(term for term in ar_terms if ".cross." in term.label)
    ma_cross = next(term for term in ma_terms if ".cross." in term.label)
    assert ar_cross.lag == 5
    np.testing.assert_allclose(ar_cross.matrix, -(seasonal_matrix @ ordinary_matrix))
    np.testing.assert_allclose(ma_cross.matrix, seasonal_matrix @ ordinary_matrix)
    assert not np.allclose(
        seasonal_matrix @ ordinary_matrix,
        ordinary_matrix @ seasonal_matrix,
    )


def test_seasonal_simulation_without_seasonal_factors_matches_starma() -> None:
    weights = _identity_weights(2)
    options = dict(
        phi=np.array([[0.3]]),
        theta=np.array([[0.1]]),
        weights=weights,
        n_steps=80,
        burnin=40,
        random_state=41,
    )
    baseline = simulate_starma(**options)
    seasonal = simulate_seasonal_starma(
        **options,
        seasonal_phi=np.zeros((0, 1)),
        seasonal_theta=np.zeros((0, 1)),
        seasonal_period=4,
    )
    np.testing.assert_allclose(seasonal, baseline)


def test_seasonal_starima_recovers_ar_factor_coefficients() -> None:
    weights = _identity_weights(2)
    data = simulate_seasonal_starma(
        phi=np.array([[0.25]]),
        seasonal_phi=np.array([[0.35]]),
        theta=np.zeros((0, 1)),
        seasonal_theta=np.zeros((0, 1)),
        weights=weights,
        seasonal_period=4,
        n_steps=700,
        burnin=300,
        random_state=1234,
    )
    model = SeasonalSTARIMA(
        ar_order=1,
        seasonal_ar_order=1,
        seasonal_period=4,
        seasonal_integration_order=0,
        include_intercept=False,
        max_iter=2000,
    )
    result = model.fit(data, weights)
    assert model.order == (1, 0, 0)
    assert model.seasonal_order == (1, 0, 0, 4)
    assert model.max_lag == 5
    assert result.converged
    np.testing.assert_allclose(result.params, [0.25, 0.35], atol=0.08)
    assert result.parameter_names == ("ar.t1.W0", "sar.t4.W0")
    assert model.predict(6).shape == (6, 2)


def test_seasonal_starima_recovers_ma_factor_coefficients() -> None:
    weights = _identity_weights(2)
    data = simulate_seasonal_starma(
        phi=np.zeros((0, 1)),
        seasonal_phi=np.zeros((0, 1)),
        theta=np.array([[0.20]]),
        seasonal_theta=np.array([[0.30]]),
        weights=weights,
        seasonal_period=4,
        n_steps=1000,
        burnin=300,
        random_state=2,
    )
    model = SeasonalSTARIMA(
        ar_order=0,
        ma_order=1,
        seasonal_ar_order=0,
        seasonal_ma_order=1,
        seasonal_period=4,
        seasonal_integration_order=0,
        include_intercept=False,
        max_iter=2000,
    )
    result = model.fit(data, weights)
    assert result.converged
    np.testing.assert_allclose(result.params, [0.20, 0.30], atol=0.08)
    assert result.parameter_names == ("ma.t1.W0", "sma.t4.W0")


def test_seasonal_starima_zero_seasonal_orders_match_starima() -> None:
    weights = _identity_weights(2)
    data = simulate_starma(
        phi=np.array([[0.30]]),
        theta=np.zeros((0, 1)),
        weights=weights,
        n_steps=180,
        burnin=80,
        random_state=7,
    )
    ordinary = STARIMA(
        ar_order=1,
        integration_order=0,
        include_intercept=False,
    )
    seasonal = SeasonalSTARIMA(
        ar_order=1,
        integration_order=0,
        seasonal_ar_order=0,
        seasonal_integration_order=0,
        seasonal_ma_order=0,
        seasonal_period=4,
        include_intercept=False,
    )
    ordinary_result = ordinary.fit(data, weights)
    seasonal_result = seasonal.fit(data, weights)
    np.testing.assert_allclose(seasonal_result.params, ordinary_result.params)
    np.testing.assert_allclose(seasonal.predict(5), ordinary.predict(5))
    assert seasonal.core_model_ is not None


def test_seasonal_differencing_only_delegates_to_starma_core() -> None:
    weights = _identity_weights(1)
    stationary = simulate_starma(
        phi=np.array([[0.2]]),
        theta=np.zeros((0, 1)),
        weights=weights,
        n_steps=220,
        burnin=80,
        random_state=12,
    )
    initial = CombinedDifferencingState(
        ordinary=DifferencingState(order=0, n_locations=1, anchors=()),
        seasonal=SeasonalDifferencingState(
            order=1,
            period=4,
            n_locations=1,
            histories=(np.zeros((4, 1)),),
        ),
    )
    data = initial.inverse_forecast(stationary)
    transformed, _ = combined_difference(
        data,
        ordinary_order=0,
        seasonal_order=1,
        seasonal_period=4,
    )
    baseline = STARMA(ar_order=1, include_intercept=False)
    baseline_result = baseline.fit(transformed, weights)
    model = SeasonalSTARIMA(
        ar_order=1,
        seasonal_ar_order=0,
        seasonal_integration_order=1,
        seasonal_period=4,
        include_intercept=False,
    )
    result = model.fit(data, weights)
    np.testing.assert_allclose(result.params, baseline_result.params)
    assert model.core_model_ is not None


def test_simulate_seasonal_starima_round_trip() -> None:
    weights = _identity_weights(2)
    options = dict(
        phi=np.array([[0.2]]),
        seasonal_phi=np.array([[0.25]]),
        theta=np.zeros((0, 1)),
        seasonal_theta=np.zeros((0, 1)),
        weights=weights,
        seasonal_period=4,
        n_steps=100,
        burnin=60,
        random_state=22,
    )
    stationary = simulate_seasonal_starma(**options)
    integrated = simulate_seasonal_starima(
        **options,
        integration_order=1,
        seasonal_integration_order=1,
    )
    transformed, _ = combined_difference(
        integrated,
        ordinary_order=1,
        seasonal_order=1,
        seasonal_period=4,
    )
    np.testing.assert_allclose(transformed, stationary[5:])


def test_seasonal_starima_runtime_and_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SeasonalSTARIMA(
            ar_order=0,
            ma_order=0,
            seasonal_ar_order=0,
            seasonal_ma_order=0,
        )
    with pytest.raises(ValueError, match="positive"):
        SeasonalSTARIMA(seasonal_period=0)

    model = SeasonalSTARIMA(
        ar_order=0,
        seasonal_ar_order=1,
        seasonal_period=4,
        seasonal_integration_order=0,
    )
    with pytest.raises(RuntimeError, match="fit"):
        model.predict()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_differenced()
    with pytest.raises(ValueError, match="too few"):
        model.fit(np.arange(5.0).reshape(5, 1), _identity_weights(1))


def test_simulate_seasonal_starima_validates_initial_state() -> None:
    weights = _identity_weights(1)
    options = dict(
        phi=np.array([[0.2]]),
        seasonal_phi=np.zeros((0, 1)),
        theta=np.zeros((0, 1)),
        seasonal_theta=np.zeros((0, 1)),
        weights=weights,
        seasonal_period=4,
        n_steps=12,
        burnin=5,
        random_state=3,
    )
    wrong_period = CombinedDifferencingState(
        ordinary=DifferencingState(order=0, n_locations=1, anchors=()),
        seasonal=SeasonalDifferencingState(
            order=1,
            period=3,
            n_locations=1,
            histories=(np.zeros((3, 1)),),
        ),
    )
    with pytest.raises(ValueError, match="period"):
        simulate_seasonal_starima(
            **options,
            integration_order=0,
            seasonal_integration_order=1,
            initial_state=wrong_period,
        )
