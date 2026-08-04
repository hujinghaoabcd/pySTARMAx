# Step 30 handoff: seasonal exact-diffuse forecasting

## Development position

- development version: `0.0.30`;
- branch: `agent/seasonal-exact-diffuse-forecasting`;
- base: version 0.0.29 on `main` at merge commit
  `1f92083fcd51fa788095d753ed293137f5886204`.

## Scope

Step 30 adds fixed-parameter forecast paths and central simulation intervals for
seasonal original-level exact-diffuse models. It covers both:

- original observations `y_t`;
- the combined transformed process
  `(1-B)^d (1-B^s)^D y_t`.

The complete augmented state is simulated from the terminal exact-diffuse
posterior. Original-level values are therefore restored separately for every
simulated path before empirical quantiles are computed.

## Safety contract

Forecast simulation requires `final_diffuse_rank == 0`. An unresolved terminal
diffuse direction does not define a proper finite Gaussian posterior and is
rejected explicitly.

## Deliberate boundaries

This stage does not add parameter-uncertainty propagation, profile likelihood,
bootstrap refitting, seasonal conditional simulation smoothing, sparse forecast
execution, or analytic forecast covariance recursions.

The final API, independent references, CI record, documentation inventory, and
next-stage handoff will be completed after implementation validation.
