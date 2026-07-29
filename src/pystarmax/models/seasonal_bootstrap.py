# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Bootstrap recursion for multiplicative seasonal STARIMA models."""

from __future__ import annotations

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.bootstrap import (
    BootstrapMethod,
    draw_bootstrap_innovations,
)
from pystarmax.models.bootstrap_starma import (
    bootstrap_sample,
    simulate_bootstrap_paths,
)
from pystarmax.models.seasonal_starima import SeasonalSTARIMA
from pystarmax.seasonal import LagOperator
from pystarmax.weights import SpatialWeights


def _seasonal_components(
    model: SeasonalSTARIMA,
) -> tuple[
    SpatialWeights,
    FloatArray,
    FloatArray,
    float,
    tuple[LagOperator, ...],
    tuple[LagOperator, ...],
]:
    if (
        model.result_ is None
        or model.weights_ is None
        or model.transformed_data_ is None
        or model._internal_residuals is None
    ):
        raise RuntimeError("fit must be called before bootstrap")
    intercept, ar_terms, ma_terms = model._operators(
        model.result_.params,
        model.weights_,
    )
    return (
        model.weights_,
        model.transformed_data_,
        model._internal_residuals,
        intercept,
        ar_terms,
        ma_terms,
    )


def seasonal_bootstrap_sample(
    model: SeasonalSTARIMA,
    *,
    bootstrap_method: BootstrapMethod,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    """Generate a same-length transformed seasonal pseudo-series."""
    if model.core_model_ is not None:
        return bootstrap_sample(
            model.core_model_,
            bootstrap_method=bootstrap_method,
            random_state=random_state,
        )

    (
        weights,
        data,
        internal_residuals,
        intercept,
        ar_terms,
        ma_terms,
    ) = _seasonal_components(model)
    if model.result_ is None:
        raise RuntimeError("fit must be called before bootstrap")

    max_lag = model.max_lag
    draws = draw_bootstrap_innovations(
        bootstrap_method=bootstrap_method,
        residuals=internal_residuals[max_lag:],
        covariance=model.result_.innovation_covariance,
        n_simulations=1,
        steps=data.shape[0] - max_lag,
        random_state=random_state,
    )[0]
    pseudo = np.empty_like(data, dtype=float)
    innovations = np.empty_like(internal_residuals, dtype=float)
    pseudo[:max_lag] = data[:max_lag]
    innovations[:max_lag] = internal_residuals[:max_lag]

    for time_index in range(max_lag, data.shape[0]):
        value = np.full(weights.n_locations, intercept, dtype=float)
        for operator in ar_terms:
            value += (
                operator.matrix
                @ pseudo[time_index - operator.lag]
            )
        for operator in ma_terms:
            value += (
                operator.matrix
                @ innovations[time_index - operator.lag]
            )
        innovation = draws[time_index - max_lag]
        pseudo[time_index] = value + innovation
        innovations[time_index] = innovation
    return np.ascontiguousarray(pseudo, dtype=float)


def simulate_seasonal_bootstrap_paths(
    model: SeasonalSTARIMA,
    *,
    steps: int,
    n_simulations: int,
    bootstrap_method: BootstrapMethod,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    """Simulate transformed future paths under a refitted seasonal model."""
    if model.core_model_ is not None:
        return simulate_bootstrap_paths(
            model.core_model_,
            steps=steps,
            n_simulations=n_simulations,
            bootstrap_method=bootstrap_method,
            random_state=random_state,
        )

    (
        weights,
        data,
        internal_residuals,
        intercept,
        ar_terms,
        ma_terms,
    ) = _seasonal_components(model)
    if model.result_ is None:
        raise RuntimeError("fit must be called before predict")

    future_innovations = draw_bootstrap_innovations(
        bootstrap_method=bootstrap_method,
        residuals=internal_residuals[model.max_lag :],
        covariance=model.result_.innovation_covariance,
        n_simulations=n_simulations,
        steps=steps,
        random_state=random_state,
    )
    max_lag = model.max_lag
    history = np.repeat(
        data[-max_lag:][None, :, :],
        n_simulations,
        axis=0,
    )
    innovation_history = np.repeat(
        internal_residuals[-max_lag:][None, :, :],
        n_simulations,
        axis=0,
    )
    paths = np.empty(
        (n_simulations, steps, weights.n_locations),
        dtype=float,
    )
    for step_index in range(steps):
        value = np.full(
            (n_simulations, weights.n_locations),
            intercept,
            dtype=float,
        )
        for operator in ar_terms:
            value += (
                history[:, -operator.lag, :]
                @ operator.matrix.T
            )
        for operator in ma_terms:
            value += (
                innovation_history[:, -operator.lag, :]
                @ operator.matrix.T
            )
        innovation = future_innovations[:, step_index, :]
        value += innovation
        paths[:, step_index, :] = value
        if max_lag > 1:
            history[:, :-1, :] = history[:, 1:, :]
            innovation_history[:, :-1, :] = (
                innovation_history[:, 1:, :]
            )
        history[:, -1, :] = value
        innovation_history[:, -1, :] = innovation
    return np.ascontiguousarray(paths, dtype=float)
