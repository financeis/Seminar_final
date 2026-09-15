"""Adapt the shared timing ledger to one fit-once rolling feature window."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from ..config import ResearchSettings
from ..contracts import ErrorCode, ResearchError
from ..data import MacroVintage
from ..data.calendar import decision_ledger
from .preprocessing import FeatureResult, FittedPreprocessor, fit_preprocessor
from .transforms import transform_tcodes


@dataclass(frozen=True)
class FeatureWindow:
    preprocessor: FittedPreprocessor
    condition: FeatureResult
    predictor: FeatureResult
    current: FeatureResult
    training_rows: pd.DataFrame
    ledger: dict


def build_feature_window(vintage: MacroVintage, ledger: dict,
                         settings: ResearchSettings | None = None) -> FeatureWindow:
    """Use one selected public release for U_(s+1), U_s and U_m.

    condition is the 48-row PCA/clustering fit. predictor is one month earlier
    per target row. current is a one-row query. All three share the identical
    scaler/PCA. Training rows retain target/predictor/condition date fields for
    explicit downstream joins, rather than assigning the same index to them.
    """
    settings = settings or ResearchSettings(profile=ledger['profile'], lag_months=ledger['lag_months'])
    if settings.train_months != 48 or len(ledger['training_rows']) != 48:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'rolling features require exactly 48 fit rows')
    if settings.profile != ledger['profile'] or settings.lag_months != ledger['lag_months']:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'feature settings disagree with timing ledger')
    expected = decision_ledger(ledger['decision_month'], [vintage.vintage_month],
                               profile=settings.profile, lag_months=settings.lag_months, train_months=48)
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, f'feature timing ledger mismatch: {key}')
    if vintage.assumed_available_at != ledger['assumed_available_at']:
        raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'macro vintage availability disagrees with ledger')
    # Discard future raw rows before transformation; earlier rows are context.
    raw = vintage.values.loc[vintage.values.index <= ledger['macro_cutoff_base_month']]
    transformed = transform_tcodes(raw, vintage.tcodes)
    rows = pd.DataFrame(ledger['training_rows'])
    fit_months = rows.condition_base_month.tolist()
    provenance = {key: ledger[key] for key in ('decision_month', 'decision_at', 'profile', 'lag_months',
                  'vintage_month', 'assumed_available_at', 'macro_cutoff_base_month')}
    provenance.update(dataset_id=vintage.dataset_id, tcodes={c: int(vintage.tcodes[c]) for c in transformed})
    preprocessor = fit_preprocessor(transformed, fit_months, vintage.groups,
                                    missing_rate=settings.missing_rate, ffill_limit=settings.ffill_limit,
                                    pca_variance=settings.pca_variance, scope='rolling', provenance=provenance)
    return FeatureWindow(preprocessor, preprocessor.fit_result,
                          preprocessor.transform(transformed, rows.predictor_base_month.tolist(), expected_scope='rolling'),
                          preprocessor.transform(transformed, [ledger['current_base_month']], expected_scope='rolling'),
                          rows, dict(ledger))
