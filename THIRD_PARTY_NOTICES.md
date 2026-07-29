# Third-party notices and research references

pySTARMAx is an independent Python implementation. It does not copy source code
from the R package `starma`, `pySTARMA`, or application repositories reviewed
during project design.

The following publications informed the mathematical scope and terminology:

1. Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative procedure
   for space-time modeling. *Technometrics, 22*(1), 35–47.
2. Pfeifer, P. E., & Deutsch, S. J. (1980). Identification and interpretation
   of first order space-time ARMA models. *Technometrics, 22*(3), 397–408.
   DOI: 10.1080/00401706.1980.10486172.
3. Pfeifer, P. E., & Deutsch, S. J. (1980). Independence and sphericity tests
   for the residuals of space-time ARMA models. *Communications in Statistics -
   Simulation and Computation, 9*(5), 533–549.
4. Pfeifer, P. E., & Deutsch, S. J. (1981). Seasonal space-time ARIMA
   modeling. *Geographical Analysis, 13*(2), 117–133.
   DOI: 10.1111/j.1538-4632.1981.tb00720.x.
5. Deutsch, S. J., & Pfeifer, P. E. (1981). Space-time ARMA modeling with
   contemporaneously correlated innovations. *Technometrics, 23*(4), 401–409.
   DOI: 10.1080/00401706.1981.10487686.
6. Pfeifer, P. E., & Deutsch, S. J. (1981). Variance of the sample space-time
   autocorrelation function. *Journal of the Royal Statistical Society:
   Series B, 43*(1), 28–33.
7. Cipra, T., & Motyková, I. (1987). Study on Kalman filter in time series
   analysis. *Commentationes Mathematicae Universitatis Carolinae, 28*(3),
   549–563.
8. Di Giacinto, V. (2006). A generalized space-time ARMA model with an
   application to regional unemployment analysis in Italy. *International
   Regional Science Review, 29*(2), 159–198.
   DOI: 10.1177/0160017605279457.
9. Thombs, L. A., & Schucany, W. R. (1990). Bootstrap prediction intervals
   for autoregression. *Journal of the American Statistical Association,
   85*(410), 486–492. DOI: 10.1080/01621459.1990.10476225.
10. Pascual, L., Romo, J., & Ruiz, E. (2001). Effects of parameter estimation
    on prediction densities: a bootstrap approach. *International Journal of
    Forecasting, 17*(1), 83–103.
    DOI: 10.1016/S0169-2070(00)00069-8.
11. Pascual, L., Romo, J., & Ruiz, E. (2004). Bootstrap predictive inference
    for ARIMA processes. *Journal of Time Series Analysis, 25*(4), 449–465.
    DOI: 10.1111/j.1467-9892.2004.01713.x.

Implementation comparisons and API reconnaissance also considered:

- Felix Cheysson's R package `starma`;
- `scrat-online/pySTARMA`;
- public STARIMA application repositories for rainfall, crime, and traffic.

Those projects remain governed by their own licences. No third-party code is
bundled in pySTARMAx.

## Diagnostic implementation independence

The Yule–Walker implementation in pySTARMAx was written independently from the
published covariance definition and verified against an exact-rational fixture.
The R `starma` package was inspected only to confirm the documented ordering of
nested temporal and spatial lag systems. No GPL-licensed source is copied,
translated, linked, or bundled. The fixture generator imports no pySTARMAx or
third-party numerical code.

## Seasonal implementation independence

The multiplicative seasonal implementation was written independently from the
published matrix-polynomial definition. It uses SciPy nonlinear least squares
to retain factor-product constraints. No source from a third-party STARIMA
implementation is copied, translated, linked, or bundled.

## Bootstrap implementation independence

The direct-bootstrap workflow was written independently from the published
algorithmic descriptions. No implementation source from another forecasting
package is copied, translated, linked, or bundled. The cited papers informed the
statistical scope and terminology only.
