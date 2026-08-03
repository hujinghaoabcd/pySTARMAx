from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pystarmax._maximum_likelihood_model as likelihood_model_module
from pystarmax import KalmanSTARMA, SpatialWeights, simulate_starma


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def test_ma1_fit_reports_invertibility_and_joint_admissibility() -> None:
    weights = identity_weights()
    data = simulate_starma(
        phi=np.empty((0, 1), dtype=float),
        theta=np.array([[0.55]], dtype=float),
        weights=weights,
        n_steps=700,
        burnin=300,
        innovation_covariance=0.45,
        random_state=2026,
    )
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=1,
        covariance_type="scalar",
        include_intercept=False,
        max_iter=700,
    )

    result = model.fit(data, weights)
    diagnostic = model.admissibility()

    assert result.converged
    assert result.ma_parameters[0, 0] == pytest.approx(0.55, abs=0.15)
    assert result.spectral_radius == 0.0
    assert result.ma_inverse_spectral_radius == pytest.approx(
        abs(result.ma_parameters[0, 0])
    )
    assert result.stationary
    assert result.invertible
    assert result.admissible
    assert result.stationarity_enforced
    assert result.invertibility_enforced
    assert result.stability_boundary_distance > 0.0
    assert result.invertibility_boundary_distance > 0.0
    assert diagnostic.stationary
    assert diagnostic.invertible
    assert diagnostic.admissible
    assert diagnostic.moving_average.spectral_radius == pytest.approx(
        result.ma_inverse_spectral_radius
    )
    assert "Jointly admissible: True" in result.summary()


def test_optimizer_final_candidate_is_checked_for_invertibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(44)
    data = rng.normal(size=(120, 1))

    def fake_minimize(
        function: Any,
        x0: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> SimpleNamespace:
        candidate = np.asarray(x0, dtype=float).copy()
        candidate[0] = 1.2
        return SimpleNamespace(
            x=candidate,
            success=True,
            nit=1,
            nfev=1,
            message="forced non-invertible candidate",
        )

    monkeypatch.setattr(likelihood_model_module, "minimize", fake_minimize)
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=1,
        covariance_type="scalar",
        include_intercept=False,
        enforce_invertibility=True,
    )

    with pytest.raises(RuntimeError, match="non-invertible"):
        model.fit(data, identity_weights())


def test_invertibility_enforcement_can_be_disabled_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(45)
    data = rng.normal(size=(120, 1))

    def fake_minimize(
        function: Any,
        x0: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> SimpleNamespace:
        candidate = np.asarray(x0, dtype=float).copy()
        candidate[0] = 1.2
        return SimpleNamespace(
            x=candidate,
            success=True,
            nit=1,
            nfev=1,
            message="forced diagnostic candidate",
        )

    monkeypatch.setattr(likelihood_model_module, "minimize", fake_minimize)
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=1,
        covariance_type="scalar",
        include_intercept=False,
        enforce_invertibility=False,
    )

    result = model.fit(data, identity_weights())

    assert not result.invertibility_enforced
    assert result.stationary
    assert not result.invertible
    assert not result.admissible
    assert result.invertibility_boundary_distance < 0.0
    assert not model.admissibility().invertible


def test_admissibility_requires_fit_and_margin_validation() -> None:
    model = KalmanSTARMA(ar_order=1, ma_order=1)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.admissibility()
    with pytest.raises(ValueError, match="invertibility_margin"):
        KalmanSTARMA(invertibility_margin=0.0)
