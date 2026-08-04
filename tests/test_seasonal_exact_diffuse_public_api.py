from __future__ import annotations

import pystarmax
from pystarmax import (
    SeasonalExactDiffuseKalmanSTARIMA,
    SeasonalExactDiffuseKalmanSTARIMAResult,
    infer_seasonal_exact_diffuse_kalman_starima,
)
from pystarmax.seasonal_exact_diffuse_inference import (
    infer_seasonal_exact_diffuse_kalman_starima as ModuleInference,
)
from pystarmax.seasonal_exact_diffuse_model import (
    SeasonalExactDiffuseKalmanSTARIMA as ModuleEstimator,
)
from pystarmax.seasonal_exact_diffuse_model import (
    SeasonalExactDiffuseKalmanSTARIMAResult as ModuleResult,
)


def test_seasonal_exact_diffuse_public_exports() -> None:
    assert pystarmax.__version__ == "0.0.29"
    assert SeasonalExactDiffuseKalmanSTARIMA is ModuleEstimator
    assert SeasonalExactDiffuseKalmanSTARIMAResult is ModuleResult
    assert infer_seasonal_exact_diffuse_kalman_starima is ModuleInference
    assert hasattr(SeasonalExactDiffuseKalmanSTARIMA, "smooth")
    assert hasattr(
        SeasonalExactDiffuseKalmanSTARIMA,
        "smooth_innovation_disturbances",
    )
    assert hasattr(SeasonalExactDiffuseKalmanSTARIMA, "likelihood_inference")
    assert "SeasonalExactDiffuseKalmanSTARIMA" in pystarmax.__all__
    assert "SeasonalExactDiffuseKalmanSTARIMAResult" in pystarmax.__all__
    assert "infer_seasonal_exact_diffuse_kalman_starima" in pystarmax.__all__
