from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from pystarmax import SpatialWeights, stacf, stcov, stpacf, stpacf_yule_walker

pytestmark = pytest.mark.reference

FIXTURE_PATH = Path(__file__).parent / "reference/classic_diagnostics_v1.json"


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fraction_array(values: object) -> np.ndarray:
    nested = values
    assert isinstance(nested, list)
    return np.asarray(
        [[float(Fraction(value)) for value in row] for row in nested], dtype=float
    )


def _weights(fixture: dict[str, object]) -> SpatialWeights:
    raw_weights = fixture["weights"]
    names = fixture["weight_names"]
    assert isinstance(raw_weights, list)
    assert isinstance(names, list)
    matrices = [_fraction_array(matrix) for matrix in raw_weights]
    return SpatialWeights.from_matrices(matrices, names=names)


def test_classic_covariance_orientation_matches_exact_reference() -> None:
    fixture = _load_fixture()
    data = np.asarray(fixture["data"], dtype=float)
    weights = _weights(fixture)
    blocks = fixture["covariance_blocks_exact"]
    assert isinstance(blocks, list)

    for temporal_lag, block in enumerate(blocks):
        expected = _fraction_array(block)
        for past_lag in range(len(weights)):
            for future_lag in range(len(weights)):
                actual = stcov(
                    data,
                    weights,
                    past_spatial_lag=past_lag,
                    future_spatial_lag=future_lag,
                    temporal_lag=temporal_lag,
                )
                np.testing.assert_allclose(
                    actual, expected[past_lag, future_lag], rtol=0.0, atol=1e-14
                )


def test_classic_stacf_matches_independent_fraction_fixture() -> None:
    fixture = _load_fixture()
    data = np.asarray(fixture["data"], dtype=float)
    weights = _weights(fixture)
    expected = np.asarray(fixture["stacf"], dtype=float)
    actual = stacf(data, weights, max_tlag=int(fixture["max_tlag"]))
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-13, atol=1e-13)


def test_classic_yule_walker_stpacf_matches_exact_fixture() -> None:
    fixture = _load_fixture()
    data = np.asarray(fixture["data"], dtype=float)
    weights = _weights(fixture)
    expected = np.asarray(fixture["stpacf"], dtype=float)
    max_tlag = int(fixture["max_tlag"])

    direct = stpacf_yule_walker(data, weights, max_tlag=max_tlag, solver="solve")
    default = stpacf(data, weights, max_tlag=max_tlag, solver="solve")

    np.testing.assert_allclose(direct.to_numpy(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(default.to_numpy(), expected, rtol=1e-12, atol=1e-12)
    assert direct.attrs["method"] == "yule-walker"
    assert set(direct.attrs["solvers_used"]) == {"solve"}


def test_non_symmetric_weight_reveals_covariance_direction() -> None:
    fixture = _load_fixture()
    data = np.asarray(fixture["data"], dtype=float)
    weights = _weights(fixture)

    past_to_future = stcov(
        data,
        weights,
        past_spatial_lag=1,
        future_spatial_lag=0,
        temporal_lag=1,
    )
    reversed_spatial_roles = stcov(
        data,
        weights,
        past_spatial_lag=0,
        future_spatial_lag=1,
        temporal_lag=1,
    )

    assert not np.isclose(past_to_future, reversed_spatial_roles)


def test_yule_walker_solver_policy_is_explicit_for_singular_systems() -> None:
    data = np.ones((8, 2), dtype=float)
    weights = SpatialWeights.from_matrices([np.eye(2)], names=["W0"])

    automatic = stpacf_yule_walker(data, weights, max_tlag=2, solver="auto")
    np.testing.assert_allclose(automatic.to_numpy(), 0.0)
    assert set(automatic.attrs["solvers_used"]) == {"lstsq"}

    with np.testing.assert_raises(np.linalg.LinAlgError):
        stpacf_yule_walker(data, weights, max_tlag=2, solver="solve")


def test_regression_method_remains_available_as_legacy_diagnostic() -> None:
    fixture = _load_fixture()
    data = np.asarray(fixture["data"], dtype=float)
    weights = _weights(fixture)

    result = stpacf(data, weights, max_tlag=2, method="regression")

    assert result.shape == (2, 2)
    assert result.attrs["method"] == "regression"
    assert np.all(np.isfinite(result.to_numpy()))


def test_classic_diagnostic_validation_errors_are_clear() -> None:
    fixture = _load_fixture()
    data = np.asarray(fixture["data"], dtype=float)
    weights = _weights(fixture)

    with np.testing.assert_raises_regex(ValueError, "past_spatial_lag"):
        stcov(data, weights, past_spatial_lag=9)
    with np.testing.assert_raises_regex(ValueError, "temporal_lag"):
        stcov(data, weights, temporal_lag=data.shape[0])
    with np.testing.assert_raises_regex(ValueError, "method"):
        stpacf(data, weights, method="unknown")  # type: ignore[arg-type]
    with np.testing.assert_raises_regex(ValueError, "solver"):
        stpacf_yule_walker(data, weights, solver="unknown")  # type: ignore[arg-type]
