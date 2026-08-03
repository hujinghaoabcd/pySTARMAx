from __future__ import annotations

import numpy as np
import pytest

from pystarmax import (
    KalmanSTARMA,
    SpatialWeights,
    finite_difference_curvature,
    finite_difference_hessian,
)


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
    data[0, 0] = rng.normal(scale=standard_deviation / np.sqrt(1.0 - phi**2))
    for time_index in range(1, n_steps):
        data[time_index, 0] = phi * data[time_index - 1, 0] + rng.normal(
            scale=standard_deviation
        )
    return data


def test_finite_difference_curvature_matches_quadratic_exactly() -> None:
    hessian = np.array([[4.0, 1.25], [1.25, 3.0]], dtype=float)
    linear = np.array([-2.0, 0.75], dtype=float)
    point = np.array([0.4, -0.8], dtype=float)

    def objective(value: np.ndarray) -> float:
        return float(0.5 * value @ hessian @ value + linear @ value + 3.0)

    curvature = finite_difference_curvature(
        objective,
        point,
        relative_step=1e-3,
        absolute_step=1e-6,
    )

    np.testing.assert_allclose(
        curvature.gradient,
        hessian @ point + linear,
        atol=1e-9,
    )
    np.testing.assert_allclose(curvature.hessian, hessian, atol=1e-8)
    np.testing.assert_allclose(
        finite_difference_hessian(
            objective,
            point,
            relative_step=1e-3,
        ),
        hessian,
        atol=1e-8,
    )
    assert curvature.n_function_evaluations == 9
    assert not curvature.gradient.flags.writeable
    assert not curvature.hessian.flags.writeable


def test_finite_difference_validation_and_invalid_region() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        finite_difference_curvature(lambda value: float(value @ value), [])
    with pytest.raises(ValueError, match="relative_step"):
        finite_difference_curvature(
            lambda value: float(value @ value),
            [1.0],
            relative_step=0.0,
        )
    with pytest.raises(ValueError, match="invalid parameter region"):
        finite_difference_curvature(
            lambda value: float(1e12 + value @ value),
            [0.0],
            invalid_threshold=1e11,
        )


def test_white_noise_inference_matches_closed_form_standard_errors() -> None:
    rng = np.random.default_rng(1234)
    true_mean = 1.4
    true_standard_deviation = 0.8
    n_time = 900
    data = rng.normal(
        loc=true_mean,
        scale=true_standard_deviation,
        size=(n_time, 1),
    )
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
        max_iter=500,
    )
    fit = model.fit(data, identity_weights(1))

    inference = model.infer(relative_step=2e-4)

    fitted_standard_deviation = float(np.sqrt(fit.innovation_covariance[0, 0]))
    expected_mean_se = fitted_standard_deviation / np.sqrt(n_time)
    expected_log_std_se = 1.0 / np.sqrt(2.0 * n_time)
    assert inference.positive_definite
    assert not inference.used_pseudoinverse
    assert inference.rank == 2
    assert inference.standard_errors[0] == pytest.approx(
        expected_mean_se,
        rel=0.08,
    )
    assert inference.standard_errors[1] == pytest.approx(
        expected_log_std_se,
        rel=0.08,
    )
    assert inference.max_abs_gradient < 1e-2
    assert inference.objective_value == pytest.approx(-fit.log_likelihood)
    assert inference.n_function_evaluations == 9
    assert inference.coefficient_table.index.tolist() == ["intercept"]
    assert inference.optimizer_table.shape == (2, 4)
    assert inference.stability_boundary_distance == pytest.approx(
        fit.stability_boundary_distance
    )
    assert inference.invertibility_boundary_distance == pytest.approx(
        fit.invertibility_boundary_distance
    )
    assert inference.minimum_admissibility_distance == pytest.approx(
        min(fit.stability_boundary_distance, fit.invertibility_boundary_distance)
    )
    intervals = inference.confidence_intervals(level=0.95)
    assert intervals.loc["intercept", "lower"] < fit.intercept
    assert intervals.loc["intercept", "upper"] > fit.intercept
    assert "likelihood-curvature inference" in inference.summary()
    assert not inference.covariance.flags.writeable


def test_ar1_missing_data_inference_reports_stable_curvature() -> None:
    data = simulate_scalar_ar1(
        phi=0.45,
        standard_deviation=0.55,
        n_steps=550,
        random_state=99,
    )
    data[25:500:13, 0] = np.nan
    model = KalmanSTARMA(
        ar_order=1,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=600,
    )
    fit = model.fit(data, identity_weights(1))

    inference = model.infer(relative_step=2e-4)

    assert inference.positive_definite
    assert inference.rank == 2
    assert inference.standard_errors[0] > 0.0
    assert inference.standard_errors[1] > 0.0
    assert inference.stability_boundary_distance == pytest.approx(
        1.0 - model.stability_margin - fit.spectral_radius
    )
    assert inference.invertibility_boundary_distance == pytest.approx(
        1.0 - model.invertibility_margin
    )
    assert inference.stability_boundary_distance > 0.1
    assert inference.invertibility_boundary_distance > 0.9
    assert inference.condition_number > 1.0
    assert np.all(np.isfinite(inference.correlation))


def test_inference_requires_fit_and_valid_controls() -> None:
    model = KalmanSTARMA(ar_order=0, ma_order=0)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.infer()

    rng = np.random.default_rng(5)
    model.fit(rng.normal(size=(80, 1)), identity_weights(1))
    with pytest.raises(ValueError, match="rcond"):
        model.infer(rcond=0.0)
    inference = model.infer()
    with pytest.raises(ValueError, match="between zero and one"):
        inference.confidence_intervals(level=1.0)
