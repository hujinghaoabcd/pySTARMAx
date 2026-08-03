from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    KalmanSTARMA,
    SpatialWeights,
    delta_method_covariance,
    innovation_covariance_transform,
)


def identity_weights(n_locations: int) -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(n_locations, dtype=float),),
        names=("W0",),
    )


def test_scalar_covariance_transform_uses_one_shared_variance() -> None:
    standard_deviation = 0.7
    transform = innovation_covariance_transform(
        [np.log(standard_deviation)],
        covariance_type="scalar",
        n_locations=3,
    )
    variance = standard_deviation**2

    assert transform.parameter_names if hasattr(transform, "parameter_names") else True
    assert transform.raw_parameter_names == ("cov.log_std.shared",)
    assert transform.element_names == ("variance.shared",)
    assert transform.element_indices == ((0, 0),)
    assert transform.n_elements == 1
    assert transform.n_raw_parameters == 1
    np.testing.assert_allclose(transform.elements, [variance])
    np.testing.assert_allclose(transform.covariance_matrix, variance * np.eye(3))
    np.testing.assert_allclose(transform.jacobian, [[2.0 * variance]])
    assert transform.element_table.index.tolist() == ["variance.shared"]
    assert not transform.elements.flags.writeable
    assert not transform.jacobian.flags.writeable


def test_diagonal_covariance_transform_has_location_variances() -> None:
    standard_deviations = np.array([0.5, 1.2, 0.8], dtype=float)
    transform = innovation_covariance_transform(
        np.log(standard_deviations),
        covariance_type="diagonal",
        n_locations=3,
    )
    variances = standard_deviations**2

    assert transform.element_names == (
        "variance.location0",
        "variance.location1",
        "variance.location2",
    )
    assert transform.element_indices == ((0, 0), (1, 1), (2, 2))
    np.testing.assert_allclose(transform.elements, variances)
    np.testing.assert_allclose(transform.covariance_matrix, np.diag(variances))
    np.testing.assert_allclose(transform.jacobian, np.diag(2.0 * variances))


def test_full_covariance_transform_matches_analytic_two_location_jacobian() -> None:
    lower_00 = 1.2
    lower_10 = -0.35
    lower_11 = 0.8
    raw = np.array(
        [np.log(lower_00), lower_10, np.log(lower_11)],
        dtype=float,
    )
    transform = innovation_covariance_transform(
        raw,
        covariance_type="full",
        n_locations=2,
    )
    expected_covariance = np.array(
        [
            [lower_00**2, lower_00 * lower_10],
            [
                lower_00 * lower_10,
                lower_10**2 + lower_11**2,
            ],
        ],
        dtype=float,
    )
    expected_jacobian = np.array(
        [
            [2.0 * lower_00**2, 0.0, 0.0],
            [lower_00 * lower_10, lower_00, 0.0],
            [0.0, 2.0 * lower_10, 2.0 * lower_11**2],
        ],
        dtype=float,
    )

    assert transform.element_names == (
        "variance.location0",
        "covariance.location1.location0",
        "variance.location1",
    )
    assert transform.element_indices == ((0, 0), (1, 0), (1, 1))
    np.testing.assert_allclose(transform.covariance_matrix, expected_covariance)
    np.testing.assert_allclose(
        transform.elements,
        expected_covariance[np.tril_indices(2)],
    )
    np.testing.assert_allclose(transform.jacobian, expected_jacobian)


def test_full_covariance_analytic_jacobian_matches_finite_difference() -> None:
    raw = np.array([0.2, -0.3, -0.1, 0.4, -0.2, 0.15], dtype=float)
    transform = innovation_covariance_transform(
        raw,
        covariance_type="full",
        n_locations=3,
    )
    step = 1e-6
    numerical = np.empty_like(transform.jacobian)
    for parameter_index in range(raw.size):
        plus = raw.copy()
        minus = raw.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        plus_elements = innovation_covariance_transform(
            plus,
            covariance_type="full",
            n_locations=3,
        ).elements
        minus_elements = innovation_covariance_transform(
            minus,
            covariance_type="full",
            n_locations=3,
        ).elements
        numerical[:, parameter_index] = (plus_elements - minus_elements) / (
            2.0 * step
        )

    np.testing.assert_allclose(transform.jacobian, numerical, rtol=2e-6, atol=2e-8)


def test_delta_method_covariance_matches_matrix_formula() -> None:
    jacobian = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=float)
    source = np.array([[0.4, 0.1], [0.1, 0.7]], dtype=float)
    propagated = delta_method_covariance(jacobian, source)

    np.testing.assert_allclose(propagated, jacobian @ source @ jacobian.T)
    np.testing.assert_allclose(propagated, propagated.T)


def test_scalar_white_noise_delta_standard_error_matches_chain_rule() -> None:
    rng = np.random.default_rng(2026)
    data = rng.normal(loc=0.6, scale=0.9, size=(900, 1))
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
    )
    fit = model.fit(data, identity_weights(1))
    likelihood_inference = model.infer(relative_step=2e-4)
    covariance_inference = likelihood_inference.innovation_covariance_inference()

    fitted_variance = float(fit.innovation_covariance[0, 0])
    expected_standard_error = (
        2.0 * fitted_variance * likelihood_inference.standard_errors[-1]
    )
    assert likelihood_inference.covariance_type == "scalar"
    assert likelihood_inference.n_locations == 1
    assert covariance_inference.parameter_names == ("variance.shared",)
    assert covariance_inference.estimates[0] == pytest.approx(fitted_variance)
    assert covariance_inference.standard_errors[0] == pytest.approx(
        expected_standard_error
    )
    np.testing.assert_allclose(
        covariance_inference.covariance,
        covariance_inference.transform.jacobian
        @ covariance_inference.source_covariance
        @ covariance_inference.transform.jacobian.T,
    )
    np.testing.assert_allclose(
        covariance_inference.dynamic_cross_covariance,
        likelihood_inference.covariance[:1, 1:]
        @ covariance_inference.transform.jacobian.T,
    )
    assert covariance_inference.dynamic_parameter_names == ("intercept",)
    assert covariance_inference.table.index.tolist() == ["variance.shared"]
    assert covariance_inference.standard_error_matrix.shape == (1, 1)
    assert "delta-method inference" in covariance_inference.summary()
    intervals = covariance_inference.confidence_intervals(level=0.95)
    assert intervals.loc["variance.shared", "lower"] < fitted_variance
    assert intervals.loc["variance.shared", "upper"] > fitted_variance
    assert not covariance_inference.covariance.flags.writeable
    assert not covariance_inference.dynamic_cross_covariance.flags.writeable


def test_covariance_inference_validation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        innovation_covariance_transform(
            [0.0],
            covariance_type="scalar",
            n_locations=0,
        )
    with pytest.raises(ValueError, match="exactly"):
        innovation_covariance_transform(
            [0.0, 0.0],
            covariance_type="scalar",
            n_locations=2,
        )
    with pytest.raises(ValueError, match="finite"):
        innovation_covariance_transform(
            [np.nan],
            covariance_type="scalar",
            n_locations=1,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        delta_method_covariance([1.0], np.eye(1))
    with pytest.raises(ValueError, match="square"):
        delta_method_covariance(np.eye(1), np.ones((1, 2)))
    with pytest.raises(ValueError, match="dimensions"):
        delta_method_covariance(np.ones((1, 2)), np.eye(1))


def test_covariance_interval_level_validation() -> None:
    rng = np.random.default_rng(7)
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
    )
    model.fit(rng.normal(size=(150, 1)), identity_weights(1))
    covariance_inference = model.infer().innovation_covariance_inference()

    with pytest.raises(ValueError, match="between zero and one"):
        covariance_inference.confidence_intervals(level=1.0)
