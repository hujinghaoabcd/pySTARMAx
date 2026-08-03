from __future__ import annotations

import numpy as np
import pytest

import pystarmax.likelihood_inference as inference_module
from pystarmax import (
    FiniteDifferenceCurvature,
    KalmanSTARMA,
    SpatialWeights,
)


def identity_weights() -> SpatialWeights:
    return SpatialWeights(
        matrices=(np.eye(1, dtype=float),),
        names=("W0",),
    )


def test_singular_hessian_requires_explicit_pseudoinverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(123)
    model = KalmanSTARMA(
        ar_order=0,
        ma_order=0,
        covariance_type="scalar",
        include_intercept=True,
    )
    fit = model.fit(rng.normal(size=(120, 1)), identity_weights())
    point = fit.raw_optimizer_params
    curvature = FiniteDifferenceCurvature(
        point=point,
        steps=np.full(point.size, 1e-4, dtype=float),
        function_value=-fit.log_likelihood,
        gradient=np.zeros(point.size, dtype=float),
        hessian=np.diag([1.0, 0.0]),
        n_function_evaluations=9,
    )
    monkeypatch.setattr(
        inference_module,
        "finite_difference_curvature",
        lambda *args, **kwargs: curvature,
    )

    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        model.infer()

    result = model.infer(allow_singular=True)

    assert result.used_pseudoinverse
    assert not result.positive_definite
    assert result.rank == 1
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(np.isfinite(result.z_values))
    assert np.all(np.isfinite(result.p_values))
    assert np.all(np.isfinite(result.correlation))
