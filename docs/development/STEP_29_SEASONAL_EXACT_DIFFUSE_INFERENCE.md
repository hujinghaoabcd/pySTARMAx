# Step 29 handoff: seasonal exact-diffuse likelihood inference

## Repository position

- development version: `0.0.29`;
- branch: `agent/seasonal-exact-diffuse-inference`;
- pull request: PR #29, `Add seasonal exact diffuse likelihood inference`;
- base: version 0.0.28 on `main` at merge commit
  `83656043b916395b95c7a1135625cb223e8a43ce`.

## Delivered API

```python
from pystarmax import SeasonalExactDiffuseKalmanSTARIMA

model = SeasonalExactDiffuseKalmanSTARIMA(...)
result = model.fit(data, weights)
inference = model.likelihood_inference()
natural = inference.innovation_covariance_inference()
```

Equivalent functional entry point:

```python
from pystarmax import infer_seasonal_exact_diffuse_kalman_starima

inference = infer_seasonal_exact_diffuse_kalman_starima(model)
```

The fitted method and functional entry point return the shared immutable
`LikelihoodInferenceResult` contract.

## Likelihood and coordinate contract

The inference target is the original-level exact-diffuse likelihood for

\[
x_t=(1-B)^d(1-B^s)^D y_t.
\]

It is not the conditional likelihood of a pre-differenced sample.

Curvature is evaluated on the fitted raw optimizer vector. Dynamic coordinates
are the free model factors retained by the MLE result:

- intercept when enabled;
- ordinary AR coefficients;
- seasonal AR coefficients;
- ordinary MA coefficients;
- seasonal MA coefficients.

Covariance coordinates are:

- one log standard deviation for `scalar`;
- one log standard deviation per location for `diagonal`;
- unconstrained lower-Cholesky coordinates with log diagonal for `full`.

Expanded multiplicative cross-lag operators are deterministic functions of the
factor coordinates. They are not independent optimizer coordinates and do not
increase the Hessian dimension, AIC, or BIC.

## Candidate reconstruction

At every central-difference stencil point the implementation:

1. calls the seasonal factor decoder and ordered multiplicative expansion;
2. decodes the candidate innovation covariance;
3. constructs the stationary transformed STARMA state;
4. constructs the original-level seasonal exact-diffuse augmentation;
5. checks expanded AR and positive-sign inverse-MA admissibility;
6. filters the original observations using their original missing-data mask;
7. returns the negative original-level exact-diffuse log likelihood.

No transformed-data shortcut is used. No arbitrary finite large covariance is
substituted for diffuse directions.

## Observed information

For raw optimizer vector `psi`, the observed information is

\[
\mathcal I(\hat\psi)
=
\nabla^2[-\ell_{\mathrm{exact}}(\psi)]\big|_{\hat\psi}.
\]

The common finite-difference engine computes central first derivatives,
diagonal second derivatives, and mixed partials. Candidate-specific steps are

\[
h_i=\max(h_{\mathrm{abs}},
          h_{\mathrm{rel}}\max(1,|\hat\psi_i|)).
\]

Default settings are:

```python
relative_step=1e-4
absolute_step=1e-6
rcond=1e-10
```

The result retains the score, Hessian, steps, objective value, and function-
evaluation count so numerical sensitivity can be audited.

## Admissibility behavior

If stationarity or invertibility enforcement is enabled, each candidate is
checked against the fitted limits using the fully expanded recursions.
Boundary violations receive the estimator-compatible smooth quadratic penalty.
Invalid covariance, state construction, or filter candidates receive a
deterministic large value, and the finite-difference engine rejects values above
its invalid threshold.

The fitted result records:

```python
inference.stability_boundary_distance
inference.invertibility_boundary_distance
```

These diagnostics indicate when the local quadratic approximation is close to
an admissibility boundary.

## Hessian rank policy

The default contract requires a positive-definite, full-rank Hessian. Otherwise
`numpy.linalg.LinAlgError` is raised with rank and minimum-eigenvalue details.

Generalized-inverse inference requires explicit consent:

```python
inference = model.likelihood_inference(allow_singular=True)
```

Only positive eigenvalue directions above the `rcond` threshold are inverted.
Zero and negative directions receive zero generalized-inverse weight. The result
reports:

- `rank`;
- `positive_definite`;
- `used_pseudoinverse`;
- `eigenvalues`;
- `condition_number`.

This path is diagnostic. It does not turn unidentified or negatively curved
directions into conventional Wald inference.

## Natural innovation-covariance inference

The shared delta-method route converts raw covariance coordinates into the
natural lower-triangular elements of `Q`:

\[
\operatorname{vech}(Q).
\]

```python
natural = inference.innovation_covariance_inference()
```

The immutable result contains:

- natural variance and covariance estimates;
- their Jacobian and delta-method covariance;
- standard errors, Wald statistics, p-values, and correlations;
- covariance between dynamic factor parameters and natural covariance elements.

## Independent mathematical references

### Seasonal random walk

For the scalar period-two model

\[
y_t=y_{t-2}+\mu+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,q),
\]

with `n` identified transitions, the test suite verifies:

\[
\mathcal I_{\mu\mu}=n/q,
\]

and, for raw log-standard-deviation coordinate `lambda`,

\[
\mathcal I_{\lambda\lambda}=2n.
\]

The corresponding natural variance standard error is

\[
\operatorname{se}(\hat q)=\hat q\sqrt{2/n}.
\]

The numerical Hessian and delta-method result are compared directly with these
closed forms.

### Ordinary reduction

When seasonal AR, integration, and MA orders are all zero, the seasonal fitted
model reduces to the ordinary exact-integrated state route. Tests require
agreement with `ExactDiffuseKalmanSTARIMA` for:

- raw parameter estimates;
- observed-information Hessian;
- optimizer-scale parameter covariance.

### Missing data

A seasonal random walk with internal missing blocks and isolated missing values
is fitted and differentiated on the original mask. The test requires finite
curvature summaries and immutable public arrays. No missing value is filled
before candidate filtering.

### Full covariance and factor counting

A two-location seasonal model with full-Cholesky covariance verifies natural
variance/covariance naming and transformation. It also verifies that the dynamic
parameter count reflects free factors only; covariance coordinates remain a
separate block and expanded cross lags are never counted.

### Singular curvature

A controlled singular Hessian fixture verifies:

- default failure;
- explicit generalized-inverse opt-in;
- retained rank;
- finite standard errors, Wald statistics, p-values, and correlation output.

## Formatting diagnostic incident

The initial numerical CI passed every test and coverage job but failed Black on
the two newly added Python files. The ordinary CI log named the files but did
not expose the formatter patch.

A temporary read-only workflow pinned Black 26.5.1 and ran `black --diff`. It
identified only two expression-folding changes. Those exact changes were
applied without altering formulas or tolerances, and the temporary workflow was
then deleted.

The formal PR must contain no workflow change. Confirm this with the final
`main...branch` comparison before merge.

## Deliberate boundaries

Step 29 does not provide:

- analytic score or Hessian recursions;
- robust or sandwich covariance;
- profile-likelihood confidence intervals;
- parameter-aware forecast or smoother uncertainty;
- seasonal exact-diffuse forecast paths or intervals;
- seasonal conditional simulation smoothing;
- diffuse lag-one state covariance;
- cross-time disturbance covariance;
- sparse or parallel curvature evaluation.

A `k`-parameter central-difference Hessian requires

\[
1+2k+4\binom{k}{2}
\]

objective evaluations. Every evaluation rebuilds and filters the complete
model, so this release is a transparent moderate-size reference rather than a
large-parameter performance claim.

## Remaining roadmap after Step 29

The next core stage is seasonal exact-diffuse forecasting uncertainty. It should
provide transformed-scale and original-level forecast paths and intervals from
the terminal exact-diffuse posterior, reject unresolved terminal diffuse rank,
and restore levels pathwise before computing original-scale quantiles.

After Step 29 the inventory contains six major technical workstreams, seven
numbered core milestones, and approximately 12–16 independently reviewable
projects after ecosystem work is split.

## Validation record

Initial implementation CI #603, run ID `30957031201`, passed the complete
numerical test and coverage matrix. Its only failure was the Black formatting
issue described above.

The authoritative synchronized CI run number, run ID, exact final head, test
count, total branch coverage, and new-module coverage must be added here after
all public API and documentation files are synchronized.

## Merge checklist

Before merging PR #29:

1. synchronize version metadata, README, documentation home, roadmap, remaining
   work, project status, navigation, example, method guide, and this handoff;
2. run complete CI on the exact synchronized head;
3. record the final run and coverage values in this handoff, project status, and
   PR body;
4. confirm the formal diff contains no temporary workflow or generated artifact;
5. confirm no submitted review or unresolved review thread remains;
6. mark PR #29 ready and squash-merge version 0.0.29;
7. create the seasonal exact-diffuse forecasting branch from the new `main`.
