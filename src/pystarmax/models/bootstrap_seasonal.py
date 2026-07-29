# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Bootstrap-enabled multiplicative seasonal STARIMA estimator."""

from __future__ import annotations

import numpy as np

from pystarmax._validation import FloatArray
from pystarmax.bootstrap import (
    restore_bootstrap_series,
    validate_bootstrap_arguments,
)
from pystarmax.forecasting import (
    ForecastInterval,
    interval_from_paths,
    random_generator,
    validate_interval_arguments,
)
from pystarmax.models.seasonal_bootstrap import (
    seasonal_bootstrap_sample,
    simulate_seasonal_bootstrap_paths,
)
from pystarmax.models.seasonal_starima import (
    SeasonalSTARIMA as _SeasonalSTARIMA,
)


class SeasonalSTARIMA(_SeasonalSTARIMA):
    """Multiplicative seasonal STARIMA with bootstrap intervals."""

    def _new_unfitted(self) -> "SeasonalSTARIMA":
        return SeasonalSTARIMA(
            ar_order=self.ar_order,
            integration_order=self.integration_order,
            ma_order=self.ma_order,
            seasonal_ar_order=self.seasonal_ar_order,
            seasonal_integration_order=(
                self.seasonal_integration_order
            ),
            seasonal_ma_order=self.seasonal_ma_order,
            seasonal_period=self.seasonal_period,
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
        """Return an original-scale seasonal bootstrap interval."""
        steps, level, _ = validate_interval_arguments(
            steps=steps,
            level=level,
            n_simulations=max(2, n_bootstrap),
        )
        n_bootstrap, max_attempts, method = (
            validate_bootstrap_arguments(
                n_bootstrap=n_bootstrap,
                max_attempts=max_attempts,
                bootstrap_method=bootstrap_method,
            )
        )
        if self.data_ is None or self.weights_ is None:
            raise RuntimeError("fit must be called before bootstrap")

        generator = random_generator(random_state)
        paths: list[FloatArray] = []
        attempts = 0
        while len(paths) < n_bootstrap and attempts < max_attempts:
            attempts += 1
            try:
                pseudo_transformed = seasonal_bootstrap_sample(
                    self,
                    bootstrap_method=method,
                    random_state=generator,
                )
                pseudo = restore_bootstrap_series(
                    pseudo_transformed,
                    self.data_,
                    ordinary_order=self.integration_order,
                    seasonal_order=(
                        self.seasonal_integration_order
                    ),
                    seasonal_period=self.seasonal_period,
                )
                fitted = self._new_unfitted()
                result = fitted.fit(pseudo, self.weights_)
                if require_convergence and not result.converged:
                    continue
                if include_future_innovations:
                    if fitted.differencing_state_ is None:
                        continue
                    transformed_path = (
                        simulate_seasonal_bootstrap_paths(
                            fitted,
                            steps=steps,
                            n_simulations=1,
                            bootstrap_method=method,
                            random_state=generator,
                        )[0]
                    )
                    path = fitted.differencing_state_.inverse_forecast(
                        transformed_path
                    )
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
            method=(
                f"{method} bootstrap with refitting ({scope})"
            ),
        )
