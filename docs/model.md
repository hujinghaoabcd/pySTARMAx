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


## Classical covariance orientation

For spatial lags `l` and `k` and temporal lag `h`, diagnostics use

\[
\widehat{\gamma}_{lk}(h) =
\frac{1}{(T-h)N}
\sum_{t=1}^{T-h}
(W_l z_t)^\top(W_k z_{t+h}).
\]

The first weight therefore acts on the past observation and the second on the
future observation. This distinction is observable when row standardisation
makes a spatial-weight matrix non-symmetric.

The STACF is

\[
\widehat{\rho}_l(h) =
\frac{\widehat{\gamma}_{l0}(h)}
{\sqrt{\widehat{\gamma}_{ll}(0)\widehat{\gamma}_{00}(0)}}.
\]

The default STPACF constructs block Yule–Walker systems from these covariance
matrices and solves nested leading-principal systems, retaining the newest
coefficient at every temporal–spatial order.

## Ordinary integration

For ordinary `STARIMA(p, d, q)`, pySTARMAx defines

\[
w_t = (1-B)^d z_t,
\]

and fits the documented STARMA equation to `w_t`. The operator acts only along
the temporal axis; it does not change the spatial-weight matrices or their
ordering. The end-of-sample values of the original series and all lower-order
differences are stored so future forecasts can be integrated recursively.

A `STARMAResult` returned by `STARIMA.fit()` is therefore expressed on the
highest-difference scale. `STARIMA.predict_differenced()` retains that scale,
whereas `STARIMA.predict()` returns the recursively reconstructed original
scale. When an intercept is included and `d=1`, it is a drift term after
inversion.


## Multiplicative seasonal integration

For `(p,d,q)x(P,D,Q)_s`, the transformed process is

\[
w_t=(1-B)^d(1-B^s)^D z_t.
\]

The seasonal matrix polynomial multiplies the ordinary polynomial on the left.
Consequently, autoregressive cross terms use `-S_j A_i` and moving-average
cross terms use `+N_j M_i`. Matrix multiplication order is retained exactly;
no closure of the spatial-weight basis is assumed. Factor parameters are fitted
by nonlinear conditional least squares when `P>0` or `Q>0`.


## Original-scale fitted-value convention

For integrated models, `STARMAResult.fitted_values` remains on the highest
difference scale. `fitted_original()` applies the combined differencing
polynomial at each aligned time using observed lagged values. It therefore
represents a one-step conditional fit, not a recursively integrated fitted
trajectory.

## Forecast-interval convention

`predict_interval()` draws future innovations from the fitted contemporaneous
location covariance and recursively propagates them through AR and MA dynamics.
For STARIMA models, each complete simulated path is inverse-differenced before
quantiles are computed. The interval conditions on estimated coefficients and
does not yet include parameter uncertainty.
