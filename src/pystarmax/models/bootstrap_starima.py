# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Bootstrap-enabled ordinary STARIMA estimator."""

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
from pystarmax.models.bootstrap_starma import (
    bootstrap_sample,
    simulate_bootstrap_paths,
)
from pystarmax.models.starima import STARIMA as _STARIMA


class STARIMA(_STARIMA):
    """Ordinary STARIMA estimator with bootstrap intervals."""

    def _new_unfitted(self) -> "STARIMA":
        return STARIMA(
            ar_order=self.ar_order,
            integration_order=self.integration_order,
            ma_order=self.ma_order,
            include_intercept=self.core_model.include_intercept,
            max_iter=self.core_model.max_iter,
            tol=self.core_model.tol,
            ridge=self.core_model.ridge,
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
        """Return an original-scale bootstrap interval with refitting."""
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
        if self.data_ is None or self.core_model.weights_ is None:
            raise RuntimeError("fit must be called before bootstrap")

        generator = random_generator(random_state)
        paths: list[FloatArray] = []
        attempts = 0
        while len(paths) < n_bootstrap and attempts < max_attempts:
            attempts += 1
            try:
                pseudo_differenced = bootstrap_sample(
                    self.core_model,
                    bootstrap_method=method,
                    random_state=generator,
                )
                pseudo = restore_bootstrap_series(
                    pseudo_differenced,
                    self.data_,
                    ordinary_order=self.integration_order,
                )
                fitted = self._new_unfitted()
                result = fitted.fit(
                    pseudo,
                    self.core_model.weights_,
                )
                if require_convergence and not result.converged:
                    continue
                if include_future_innovations:
                    if fitted.differencing_state_ is None:
                        continue
                    differenced_path = simulate_bootstrap_paths(
                        fitted.core_model,
                        steps=steps,
                        n_simulations=1,
                        bootstrap_method=method,
                        random_state=generator,
                    )[0]
                    path = fitted.differencing_state_.inverse_forecast(
                        differenced_path
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
