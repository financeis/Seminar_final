# Forecast model contracts

`forecast_window(state, returns, *, partition_id, decision_month, settings=None,
fixed_lambdas=None, allow_inner=False)` consumes an existing `WindowState`.
The return DataFrame has exactly the state's target months in order, and the
10 `TICKERS` columns in canonical order. The state supplies predictor base-month
keys, already aligned condition labels, the query, and next-state probabilities.
The adapter never shifts labels or fits a second PCA or partition.

Outer windows contain 48 targets ending at decision month minus two. Inner
windows must carry `window_kind="inner_fold"` and be explicitly enabled;
`full_sample` is always rejected. The adapter regenerates the timing ledger
and validates all keys and dates before fitting. Extra future return rows are
rejected instead of silently sliced.

## Outputs and table semantics

| Model | Final `score_kind` | Conditional rows per 10-ETF window |
|---|---|---|
| Ridge | `expected_return` | 50: R1–R5 conditional predictions |
| Naive | `conditional_sharpe` | 10: selected R0–R5 regime scores |
| BL | `utility_weight` | 10: selected regime **view q**, `expected_return` |
| MVO | `utility_weight` | 0: unconditional moment estimate |

`ModelForecasts.scores` returns model-to-Series copies in configured order.
`forecast_records(run_id)` and `conditional_records(run_id)` return R12 rows.
Conditional `selected_lambda` is null when no Ridge penalty applies, with the
reason in that row. Undefined Naive Sharpe is explicitly tagged `undefined`;
its zero score is a decision policy, not a measured Sharpe of zero.

BL's Eq. 19 result is separately available as `arrays['bl_posterior_mean']`.
`arrays['bl_q']`, `bl_P`, `bl_omega`, `prior_mean`, `sample_covariance`, and
`regularized_covariance` preserve its inputs. Final utility directions solve
`delta * covariance * score = posterior_mean` (BL) or `= prior_mean` (MVO).
Portfolio code owns the subsequent sign/rank selection and normalization.
The paper calls Eq. 19 an allocation; this extra conversion is a declared
interpretation, rather than a claim that the equation directly gives weights.
MVO forecasts use no regime. MVO `mx` sizing can still use the shared crisis
signal downstream.

## Ridge validation and fallback

Pure `ridge_candidates(X, Y, query, lambdas)` returns candidate-by-target MSE,
prediction, validity mask, method, and failure reason. `fit_ridge` chooses a
lambda per target, with relative tie tolerance 1e-12, zero absolute tolerance,
and the largest tied lambda. An intercept is fitted without penalty; analytic
LOO includes its hat term `1/n`. A nearly unit hat diagonal triggers actual
leave-one-out refitting. Invalid candidates are excluded, and all-invalid
selection raises an error. Arrays for invalid candidate slots use finite zero
padding under `valid=False`; `RidgeCandidates.records` emits null plus reason.

`fit_ridge(..., fixed_lambda=scalar_or_target_vector)` supports externally
selected penalties without repeating LOO. The window adapter accepts only a
`fixed_lambdas` Series with exact ticker order. `validation='blocked'` requires
this input: the forward-validation caller owns past-only selection and its
candidate-loss artifact. The same ETF lambda is used for every regime.

Fewer than the configured six observations triggers whole-window Ridge by
default; raw/effective counts and `pooled` are recorded. The alternative is
`regime_mean`, with `pooled_mean` if the regime has no observations. Equation
14 sums R1–R5 only without renormalizing probabilities. If R0 has probability
one, every Ridge ETF score is exactly zero.

## Saved state

`result.save(new_directory)` writes `models.json` and `models.npz`;
`ModelForecasts.load(directory)` validates file/array hashes, shapes, dtypes,
and the model content identity. Existing files cannot be replaced. The model
ID covers settings, training keys, source partition identity, raw returns,
predictors, all forecast rows, and numerical arrays. Public properties return
copies; private tampering fails the digest check.

For each Ridge regime, archives include chosen coefficients/intercept/lambda,
all candidate MSEs and query predictions, grid, validity masks, and validation
methods/reasons. Candidate axes are lambda then canonical ticker. Naive mean
and sample standard deviation are stored; unavailable summary values use zero
padding identified by sample counts and `mean_defined`/`std_defined` flags.

Pure numerical functions accept arbitrary asset/sample counts and permit
zero shrinkage and zero diagonal regularization for positive-definite hand
calculations. They do not relax the high-level outer-window contract.
