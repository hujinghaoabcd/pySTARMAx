# Model convention

For an `N`-location process observed at time `t`, pySTARMAx uses

\[
z_t = c\mathbf{1} +
\sum_{k=1}^{p}\sum_{\ell=0}^{L}\phi_{k\ell}W_\ell z_{t-k}
+ \varepsilon_t
+ \sum_{k=1}^{q}\sum_{\ell=0}^{L}\theta_{k\ell}W_\ell\varepsilon_{t-k}.
\]

`W_0` is the identity matrix. Observations are passed as a dense matrix with
shape `(time, location)`. The current STARMA estimator uses iterative
conditional least squares; future state-space likelihood estimation will keep
the same public model convention.

## Spatial weights

`SpatialWeights.from_adjacency()` constructs row-standardized identity and
higher-order contiguity matrices. `SpatialWeights.from_matrices()` accepts an
audited user-supplied sequence. Matrices must be finite square arrays and match
the number of locations.

## Initial observations

Fitted values and residuals are undefined for the first `max(p, q)` time steps
and are returned as `NaN` there. Conditional estimation uses zero pre-sample
innovations, a convention stated explicitly in the fitted result.
