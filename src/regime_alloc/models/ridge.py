"""Intercept-unpenalized Ridge with analytic LOO and explicit unstable refits.

All targets share one SVD per design matrix. Candidate MSEs use the exact
intercept hat term 1/n. Invalid candidate slots contain finite zeros *only*
as archive padding: their validity mask is false and records emit null/reason.
"""
from dataclasses import dataclass
import numpy as np
from ..contracts import ErrorCode, ResearchError
from ._validation import finite, probabilities


@dataclass(frozen=True)
class RidgeCandidates:
    lambdas: np.ndarray
    loo_mse: np.ndarray
    predictions: np.ndarray
    valid: np.ndarray
    methods: tuple[str, ...]
    reasons: tuple[str, ...]
    coefficients: np.ndarray
    intercepts: np.ndarray

    def records(self, tickers):
        if len(tickers) != self.loo_mse.shape[1]:
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'candidate ticker count differs')
        return [{'ticker': ticker, 'lambda': float(lam),
                 'mse': float(self.loo_mse[i, j]) if self.valid[i] else None,
                 'prediction': float(self.predictions[i, j]) if self.valid[i] else None,
                 'status': 'ok' if self.valid[i] else 'undefined',
                 'method': self.methods[i], 'reason': self.reasons[i]}
                for i, lam in enumerate(self.lambdas) for j, ticker in enumerate(tickers)]


@dataclass(frozen=True)
class RidgeFit:
    prediction: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    selected_lambda: np.ndarray
    candidates: RidgeCandidates | None
    selection_method: str


def _input(x, y, query):
    x = finite(x, 'Ridge X', 2)
    y = finite(y, 'Ridge y')
    if y.ndim == 1: y = y[:, None]
    query = finite(query, 'Ridge query', 1)
    if y.ndim != 2 or y.shape[0] != len(x) or not len(x) or not x.shape[1] or not y.shape[1] or query.shape != (x.shape[1],):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'Ridge design/target/query shape mismatch')
    return x, y, query


def _svd(x, y):
    mx, my = x.mean(axis=0), y.mean(axis=0)
    constant_x, constant_y = np.all(x == x[0], axis=0), np.all(y == y[0], axis=0)
    mx[constant_x], my[constant_y] = x[0, constant_x], y[0, constant_y]
    u, s, vt = np.linalg.svd(x - mx, full_matrices=False)
    return mx, my, u, s, vt, u.T @ (y - my)


def _fit(decomposition, query, lam):
    mx, my, u, s, vt, projected = decomposition
    # lam can be a target-specific vector for external forward validation.
    lam = np.asarray(lam).reshape(1, -1)
    coefficients = vt.T @ ((s[:, None] / (s[:, None] ** 2 + lam)) * projected)
    intercept = my - mx @ coefficients
    prediction = query @ coefficients + intercept
    return coefficients, intercept, prediction


def ridge_candidates(x, y, query, lambdas, *, hat_tolerance=1e-8):
    """Return candidate-by-target MSE and predictions; never select by test data."""
    x, y, query = _input(x, y, query)
    grid = finite(lambdas, 'lambda grid', 1)
    if not len(grid) or (grid <= 0).any() or len(set(grid)) != len(grid):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'lambda grid requires distinct positive values')
    if len(x) < 2:
        raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'LOO requires at least two observations')
    if not np.isfinite(hat_tolerance) or not 0 < hat_tolerance < 1:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid hat diagonal tolerance')
    n, p, a = len(x), x.shape[1], y.shape[1]
    mse, predictions = np.zeros((len(grid), a)), np.zeros((len(grid), a))
    coefficients, intercepts = np.zeros((len(grid), p, a)), np.zeros((len(grid), a))
    valid, methods, reasons = np.zeros(len(grid), dtype=bool), [], []
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            decomposition = _svd(x, y)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'Ridge SVD failed') from exc
    _, _, u, s, _, _ = decomposition
    for i, lam in enumerate(grid):
        method = 'analytic_loo'
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                beta, intercept, prediction = _fit(decomposition, query, lam)
                diagonal = 1 / n + (u ** 2) @ (s ** 2 / (s ** 2 + lam))
                if np.any(1 - diagonal <= hat_tolerance):
                    method = 'explicit_refit'
                    errors = np.empty_like(y)
                    for row in range(n):
                        keep = np.arange(n) != row
                        _, _, held = _fit(_svd(x[keep], y[keep]), x[row], lam)
                        errors[row] = y[row] - held
                else:
                    errors = (y - (x @ beta + intercept)) / (1 - diagonal[:, None])
                candidate_mse = np.mean(errors ** 2, axis=0)
                for value in (beta, intercept, prediction, candidate_mse): finite(value, 'Ridge candidate result')
            mse[i], predictions[i], coefficients[i], intercepts[i] = candidate_mse, prediction, beta, intercept
            valid[i] = True
            reasons.append('')
        except (np.linalg.LinAlgError, FloatingPointError, ResearchError) as exc:
            reasons.append(f'invalid_ridge_candidate: {type(exc).__name__}')
        methods.append(method)
    return RidgeCandidates(grid.copy(), mse, predictions, valid, tuple(methods), tuple(reasons), coefficients, intercepts)


def fit_ridge(x, y, query, *, lambdas=None, fixed_lambda=None):
    """Fit arbitrary sample/asset counts. λ is scalar or one value per target.

    LOO ties use relative tolerance 1e-12 (zero absolute tolerance) and select
    the largest λ. External λ bypasses LOO; its upstream validation is recorded.
    """
    x, y, query = _input(x, y, query)
    if fixed_lambda is not None:
        if lambdas is not None:
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'specify candidate lambdas or fixed_lambda, not both')
        selected = finite(fixed_lambda, 'fixed lambda')
        if selected.ndim == 0: selected = np.full(y.shape[1], float(selected))
        if selected.shape != (y.shape[1],) or (selected <= 0).any():
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'fixed lambda must be positive per target')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                beta, intercept, prediction = _fit(_svd(x, y), query, selected)
                for value in (beta, intercept, prediction): finite(value, 'fixed Ridge result')
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'fixed Ridge fit failed') from exc
        return RidgeFit(prediction, beta, intercept, selected.copy(), None, 'fixed_external')
    candidates = ridge_candidates(x, y, query, np.logspace(-4, 4, 41) if lambdas is None else lambdas)
    if not candidates.valid.any():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'all Ridge LOO candidates failed', details={'reasons': list(candidates.reasons)})
    chosen = []
    for j in range(y.shape[1]):
        best = np.min(candidates.loo_mse[candidates.valid, j])
        eligible = np.flatnonzero(candidates.valid & np.isclose(candidates.loo_mse[:, j], best, rtol=1e-12, atol=0))
        chosen.append(eligible[np.argmax(candidates.lambdas[eligible])])
    selected = np.array(chosen)
    cols = np.arange(y.shape[1])
    return RidgeFit(candidates.predictions[selected, cols],
                    np.column_stack([candidates.coefficients[i, :, j] for j, i in enumerate(selected)]),
                    candidates.intercepts[selected, cols], candidates.lambdas[selected], candidates, 'analytic_loo_with_explicit_guard')


def aggregate_ridge(conditional_predictions, next_probabilities):
    """Equation 14: exactly R1..R5, with no normal-probability renormalization."""
    conditional = finite(conditional_predictions, 'conditional Ridge predictions', 2)
    p = probabilities(next_probabilities)
    if conditional.shape[0] != 5:
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'Ridge aggregation requires five ordered normal regimes')
    return finite(p[1:] @ conditional, 'aggregated Ridge forecast')
