import numpy as np
import pytest

from pystarmax import (
    STARIMA,
    STARMA,
    DifferencingState,
    SpatialWeights,
    ordinary_difference,
    simulate_starima,
    simulate_starma,
)


def _identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights.from_matrices(
        [np.eye(n_locations, dtype=float)], names=["W0"]
    )


def test_ordinary_difference_first_order_and_state() -> None:
    data = np.array(
        [
            [1.0, 4.0],
            [3.0, 7.0],
            [8.0, 9.0],
        ]
    )
    differenced, state = ordinary_difference(data, order=1)
    np.testing.assert_allclose(differenced, [[2.0, 3.0], [5.0, 2.0]])
    assert state.order == 1
    assert state.n_locations == 2
    np.testing.assert_allclose(state.anchors[0], data[-1])
    assert not state.anchors[0].flags.writeable
    with pytest.raises(ValueError):
        state.anchors[0][0] = 99.0


def test_second_order_forecast_inversion_is_recursive_and_repeatable() -> None:
    data = np.array([[0.0], [1.0], [3.0], [6.0]])
    differenced, state = ordinary_difference(data, order=2)
    np.testing.assert_allclose(differenced, [[1.0], [1.0]])
    forecast = np.array([[1.0], [2.0]])
    expected = np.array([[10.0], [16.0]])
    np.testing.assert_allclose(state.inverse_forecast(forecast), expected)
    np.testing.assert_allclose(state.inverse_forecast(forecast), expected)


def test_zero_order_difference_and_inverse_are_identity_operations() -> None:
    data = np.arange(12.0).reshape(6, 2)
    differenced, state = ordinary_difference(data, order=0)
    np.testing.assert_allclose(differenced, data)
    assert state.anchors == ()
    forecast = np.array([[20.0, 21.0], [22.0, 23.0]])
    restored = state.inverse_forecast(forecast)
    np.testing.assert_allclose(restored, forecast)
    assert restored is not forecast


def test_differencing_validation_errors() -> None:
    data = np.arange(8.0).reshape(4, 2)
    with pytest.raises(ValueError, match="smaller"):
        ordinary_difference(data, order=4)
    with pytest.raises(ValueError, match="non-negative"):
        ordinary_difference(data, order=-1)
    with pytest.raises(ValueError, match="two-dimensional"):
        ordinary_difference(np.arange(4.0), order=1)
    with pytest.raises(ValueError, match="one vector"):
        DifferencingState(order=2, n_locations=2, anchors=(np.zeros(2),))
    with pytest.raises(ValueError, match="shape"):
        DifferencingState(order=1, n_locations=2, anchors=(np.zeros(3),))


def test_inverse_forecast_validation_errors() -> None:
    _, state = ordinary_difference(np.arange(10.0).reshape(5, 2), order=1)
    with pytest.raises(ValueError, match="two-dimensional"):
        state.inverse_forecast(np.ones(2))
    with pytest.raises(ValueError, match="at least one"):
        state.inverse_forecast(np.empty((0, 2)))
    with pytest.raises(ValueError, match="locations"):
        state.inverse_forecast(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        state.inverse_forecast(np.array([[np.nan, 1.0]]))


def test_starima_fit_and_original_scale_prediction() -> None:
    weights = _identity_weights(2)
    data = simulate_starima(
        phi=np.array([[0.35]]),
        theta=np.zeros((0, 1)),
        weights=weights,
        n_steps=240,
        integration_order=1,
        burnin=100,
        random_state=11,
    )
    model = STARIMA(ar_order=1, integration_order=1, include_intercept=False)
    result = model.fit(data, weights)
    assert model.order == (1, 1, 0)
    assert result is model.result_
    np.testing.assert_allclose(result.params, [0.35], atol=0.12)
    assert model.differenced_data_ is not None
    np.testing.assert_allclose(model.differenced_data_, np.diff(data, axis=0))

    differenced_forecast = model.predict_differenced(steps=4)
    original_forecast = model.predict(steps=4)
    assert model.differencing_state_ is not None
    expected = model.differencing_state_.inverse_forecast(differenced_forecast)
    np.testing.assert_allclose(original_forecast, expected)
    assert original_forecast.shape == (4, 2)


def test_starima_zero_order_matches_starma() -> None:
    weights = _identity_weights(2)
    data = simulate_starma(
        phi=np.array([[0.30]]),
        theta=np.zeros((0, 1)),
        weights=weights,
        n_steps=180,
        burnin=80,
        random_state=7,
    )
    starma = STARMA(ar_order=1, include_intercept=False)
    starima = STARIMA(
        ar_order=1,
        integration_order=0,
        include_intercept=False,
    )
    result_starma = starma.fit(data, weights)
    result_starima = starima.fit(data, weights)
    np.testing.assert_allclose(result_starima.params, result_starma.params)
    np.testing.assert_allclose(starima.predict(5), starma.predict(5))


def test_starima_runtime_and_short_sample_errors() -> None:
    model = STARIMA(ar_order=1, integration_order=1)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_differenced()
    with pytest.raises(ValueError, match="after differencing"):
        model.fit(np.arange(3.0).reshape(3, 1), _identity_weights(1))

    fitted = STARIMA(ar_order=1, integration_order=0)
    fitted.fit(np.arange(8.0).reshape(8, 1), _identity_weights(1))
    with pytest.raises(ValueError, match="positive"):
        fitted.predict(steps=0)


def test_simulate_starima_integrates_the_stationary_process() -> None:
    weights = _identity_weights(2)
    options = dict(
        phi=np.array([[0.25]]),
        theta=np.zeros((0, 1)),
        weights=weights,
        n_steps=80,
        burnin=40,
        random_state=123,
    )
    stationary = simulate_starma(**options)
    first_order = simulate_starima(**options, integration_order=1)
    second_order = simulate_starima(**options, integration_order=2)
    np.testing.assert_allclose(np.diff(first_order, axis=0), stationary[1:])
    np.testing.assert_allclose(np.diff(second_order, n=2, axis=0), stationary[2:])


def test_simulate_starima_accepts_and_validates_initial_state() -> None:
    weights = _identity_weights(1)
    options = dict(
        phi=np.array([[0.2]]),
        theta=np.zeros((0, 1)),
        weights=weights,
        n_steps=12,
        burnin=5,
        random_state=3,
    )
    stationary = simulate_starma(**options)
    state = DifferencingState(
        order=1,
        n_locations=1,
        anchors=(np.array([10.0]),),
    )
    integrated = simulate_starima(
        **options,
        integration_order=1,
        initial_state=state,
    )
    np.testing.assert_allclose(integrated[0], 10.0 + stationary[0])

    with pytest.raises(ValueError, match="order"):
        simulate_starima(
            **options,
            integration_order=2,
            initial_state=state,
        )
    with pytest.raises(ValueError, match="locations"):
        simulate_starima(
            **options,
            integration_order=1,
            initial_state=DifferencingState(
                order=1,
                n_locations=2,
                anchors=(np.zeros(2),),
            ),
        )
