# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Bootstrap-enabled stationary STAR and STARMA estimators."""

from __future__ import annotations

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.bootstrap import (
    BootstrapMethod,
    draw_bootstrap_innovations,
    validate_bootstrap_arguments,
)
from pystarmax.forecasting import (
    ForecastInterval,
    interval_from_paths,
    random_generator,
    validate_interval_arguments,
)
from pystarmax.models.starma import STARMA as _STARMA


def bootstrap_sample(
    model: _STARMA,
    *,
    bootstrap_method: BootstrapMethod,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    """Generate a same-length pseudo-series conditional on initial lags."""
    (
        weights,
        data,
        internal_residuals,
        intercept,
        ar_params,
        ma_params,
    ) = model._fitted_components()
    if model.result_ is None:
        raise RuntimeError("fit must be called before bootstrap")

    max_lag = model.max_lag
    future_rows = data.shape[0] - max_lag
    draws = draw_bootstrap_innovations(
        bootstrap_method=bootstrap_method,
        residuals=internal_residuals[max_lag:],
        covariance=model.result_.innovation_covariance,
        n_simulations=1,
        steps=future_rows,
        random_state=random_state,
    )[0]

    pseudo = np.empty_like(data, dtype=float)
    innovations = np.empty_like(internal_residuals, dtype=float)
    pseudo[:max_lag] = data[:max_lag]
    innovations[:max_lag] = internal_residuals[:max_lag]
    for time_index in range(max_lag, data.shape[0]):
        value = np.full(weights.n_locations, intercept, dtype=float)
        for temporal_lag in range(1, model.ar_order + 1):
            for spatial_lag, matrix in enumerate(weights):
                value += ar_params[temporal_lag - 1, spatial_lag] * (
                    matrix @ pseudo[time_index - temporal_lag]
                )
        for temporal_lag in range(1, model.ma_order + 1):
            for spatial_lag, matrix in enumerate(weights):
                value += ma_params[temporal_lag - 1, spatial_lag] * (
                    matrix @ innovations[time_index - temporal_lag]
                )
        innovation = draws[time_index - max_lag]
        pseudo[time_index] = value + innovation
        innovations[time_index] = innovation
    return np.ascontiguousarray(pseudo, dtype=float)


def simulate_bootstrap_paths(
    model: _STARMA,
    *,
    steps: int,
    n_simulations: int,
    bootstrap_method: BootstrapMethod,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    """Simulate fitted future paths with residual or parametric innovations."""
    (
        weights,
        data,
        internal_residuals,
        intercept,
        ar_params,
        ma_params,
    ) = model._fitted_components()
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
        for temporal_lag in range(1, model.ar_order + 1):
            for spatial_lag, matrix in enumerate(weights):
                value += ar_params[temporal_lag - 1, spatial_lag] * (
                    history[:, -temporal_lag, :] @ matrix.T
                )
        for temporal_lag in range(1, model.ma_order + 1):
            for spatial_lag, matrix in enumerate(weights):
                value += ma_params[temporal_lag - 1, spatial_lag] * (
                    innovation_history[:, -temporal_lag, :] @ matrix.T
                )
        innovation = future_innovations[:, step_index, :]
        value += innovation
        paths[:, step_index, :] = value
        if max_lag > 1:
            history[:, :-1, :] = history[:, 1:, :]
            innovation_history[:, :-1, :] = innovation_history[:, 1:, :]
        history[:, -1, :] = value
        innovation_history[:, -1, :] = innovation
    return np.ascontiguousarray(paths, dtype=float)


class STARMA(_STARMA):
    """STARMA estimator with parameter-aware bootstrap intervals."""

    def _new_unfitted(self) -> "STARMA":
        return STARMA(
            ar_order=self.ar_order,
            ma_order=self.ma_order,
            include_intercept=self.include_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            ridge=self.ridge,
        )

    def predict_bootstrap_interval(
        self,
        steps: int = 1,
        *,
        level: float = 0.95,
        n_bootstrap: int = 200,
        bootstrap_method: str = "residual",
        include_future_innovations: bool = True,
        require_convergence: bool = True,
        max_attempts: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> ForecastInterval:
        """Return a direct-bootstrap interval with model refitting."""
        steps, level, _ = validate_interval_arguments(
            steps=steps,
            level=level,
            n_simulations=max(2, n_bootstrap),
        )
        n_bootstrap, max_attempts, method = validate_bootstrap_arguments(
            n_bootstrap=n_bootstrap,
            max_attempts=max_attempts,
            bootstrap_method=bootstrap_method,
        )
        if self.weights_ is None:
            raise RuntimeError("fit must be called before bootstrap")

        generator = random_generator(random_state)
        paths: list[FloatArray] = []
        attempts = 0
        while len(paths) < n_bootstrap and attempts < max_attempts:
            attempts += 1
            try:
                pseudo = bootstrap_sample(
                    self,
                    bootstrap_method=method,
                    random_state=generator,
                )
                fitted = self._new_unfitted()
                result = fitted.fit(pseudo, self.weights_)
                if require_convergence and not result.converged:
                    continue
                if include_future_innovations:
                    path = simulate_bootstrap_paths(
                        fitted,
                        steps=steps,
                        n_simulations=1,
                        bootstrap_method=method,
                        random_state=generator,
                    )[0]
                else:
                    path = fitted.predict(steps=steps)
                if np.all(np.isfinite(path)):
                    paths.append(np.asarray(path, dtype=float))
            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ):
                continue

        if len(paths) < n_bootstrap:
            raise RuntimeError(
                "bootstrap produced "
                f"{len(paths)} successful refits from {attempts} attempts; "
                "increase max_attempts or relax require_convergence"
            )
        scope = (
            "parameter and future-innovation uncertainty"
            if include_future_innovations
            else "parameter uncertainty"
        )
        return interval_from_paths(
            mean=self.predict(steps=steps),
            paths=np.stack(paths, axis=0),
            level=level,
            method=(f"{method} bootstrap with refitting ({scope})"),
        )


class STAR(STARMA):
    """Pure space-time autoregression with bootstrap intervals."""

    def __init__(
        self,
        ar_order: int = 1,
        *,
        include_intercept: bool = True,
        ridge: float = 0.0,
    ) -> None:
        super().__init__(
            ar_order=ar_order,
            ma_order=0,
            include_intercept=include_intercept,
            ridge=ridge,
        )
