"""One content-addressed partition shared by all forecasts in a decision window."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import ResearchSettings
from ..contracts import (ErrorCode, ResearchError, canonical_id, checked_join,
                         save_npz, load_npz, write_json)
from ..data.io import read_json
from ..data.calendar import decision_ledger
from .clustering import fit_partition, partition_from_labels
from .probabilities import regime_probabilities, next_probabilities, finite_array
from .transitions import transition_matrices, ordered_months


def _digest(metadata, arrays):
    specs = {name: {'shape':list(a.shape),'dtype':a.dtype.str,
                   'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
             for name,a in arrays.items()}
    return canonical_id({'metadata':metadata,'arrays':specs})


def _freeze(metadata, arrays):
    owned = {name:np.array(a,copy=True) for name,a in arrays.items()}
    for a in owned.values():
        if a.dtype.kind not in 'biuf' or not np.isfinite(a).all():
            raise ResearchError(ErrorCode.NONFINITE_VALUE, 'window state must contain finite numeric arrays')
        a.setflags(write=False)
    meta = deepcopy(metadata)
    return WindowState(meta,owned,_digest(meta,owned))


def _partition_arrays(condition, current, months, partition):
    transition = transition_matrices(partition.labels,months,len(partition.centers))
    current_p = regime_probabilities(current,partition.stage1_centers,partition.centers[1:])
    if current_p.shape != (1,len(partition.centers)):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'window requires a single current query')
    return {'labels':partition.labels,'centers':partition.centers,'stage1_centers':partition.stage1_centers,
            'current_probability':current_p[0], 'next_probability':next_probabilities(current_p[0],transition.matrix),
            'transition_matrix':transition.matrix,'literal_transition_matrix':transition.literal_matrix,
            'transition_counts':transition.counts,'occurrences':transition.occurrences,
            'outgoing':transition.outgoing,'no_outgoing':transition.no_outgoing}, transition


@dataclass(frozen=True)
class WindowState:
    """All public views are copies. Every computation also checks the digest.

    The ID includes the actual input/query arrays, row dates, ledger, variable
    and component order, fitted preprocessing, algorithm settings, centers,
    memberships, regime order, transitions and probabilities. Interpretation
    names are deliberately not part of this predictive partition.
    """
    _metadata: dict
    _arrays: dict[str,np.ndarray]
    partition_id: str

    def _verify(self):
        try: valid = self.partition_id == _digest(self._metadata,self._arrays)
        except (TypeError,ValueError,AttributeError,ResearchError) as exc:
            raise ResearchError(ErrorCode.HASH_MISMATCH,'window state was changed under an existing ID') from exc
        if not valid:
            raise ResearchError(ErrorCode.HASH_MISMATCH,'window state was changed under an existing ID')

    @property
    def metadata(self):
        self._verify(); return deepcopy(self._metadata)

    def _array(self,name):
        self._verify(); value=self._arrays[name].copy(); value.setflags(write=False); return value

    @property
    def regime_ids(self): self._verify(); return tuple(self._metadata['regime_ids'])
    @property
    def decision_month(self): self._verify(); return self._metadata['ledger']['decision_month']
    @property
    def scope(self): self._verify(); return self._metadata['scope']
    @property
    def feature_hash(self): self._verify(); return self._metadata['feature_hash']
    @property
    def transform_hash(self): self._verify(); return self._metadata['transform_hash']
    @property
    def ledger(self): self._verify(); return deepcopy(self._metadata['ledger'])
    @property
    def training_rows(self):
        self._verify()
        rows=pd.DataFrame(deepcopy(self._metadata['training_rows']))
        rows['regime_id']=[self._metadata['regime_ids'][i] for i in self._arrays['labels']]
        return rows
    @property
    def labels(self):
        self._verify()
        return pd.Series(self._arrays['labels'].copy(),index=pd.Index([r['target_month'] for r in self._metadata['training_rows']],name='holding_month'),name='regime_label')
    @property
    def condition_labels(self):
        self._verify()
        return pd.Series(self._arrays['labels'].copy(),index=pd.Index(self._metadata['condition_months'],name='base_month'),name='regime_label')
    def _scores(self,name):
        return pd.DataFrame(self._array(name+'_scores'),index=pd.Index(self._metadata[name+'_months'],name='base_month'),columns=self._metadata['pc_columns'])
    @property
    def condition_scores(self): return self._scores('condition')
    @property
    def predictor_scores(self): return self._scores('predictor')
    @property
    def current_scores(self): return self._scores('current')
    @property
    def centers(self): return self._array('centers')
    @property
    def stage1_centers(self): return self._array('stage1_centers')
    @property
    def current_probability(self): return self._array('current_probability')
    @property
    def next_probability(self): return self._array('next_probability')
    @property
    def transition_matrix(self): return self._array('transition_matrix')
    @property
    def literal_transition_matrix(self): return self._array('literal_transition_matrix')
    @property
    def transition_counts(self): return self._array('transition_counts')

    def require_identity(self,partition_id,*,decision_month=None):
        self._verify()
        if partition_id != self.partition_id or decision_month is not None and decision_month != self.decision_month:
            raise ResearchError(ErrorCode.PARTITION_MISMATCH,'forecast/probability identity or decision month mismatch')

    def require_scope(self,scope):
        self._verify()
        if scope != self._metadata['scope']:
            raise ResearchError(ErrorCode.PARTITION_MISMATCH,'state belongs to a different analysis scope')

    def centers_in_feature_units(self):
        self._verify()
        a=self._arrays
        restored=(a['centers']@a['preprocessor_components']+a['preprocessor_pca_mean'])*a['preprocessor_scale']+a['preprocessor_mean']
        finite_array(restored,name='inverse transformed centers')
        return pd.DataFrame(restored,index=self.regime_ids,columns=self._metadata['feature_columns'])

    @property
    def feature_scale(self):
        return pd.Series(self._array('preprocessor_scale'),index=self._metadata['feature_columns'])

    def probability_records(self,run_id):
        self._verify()
        return [{'run_id':run_id,'decision_month':self.decision_month,'partition_id':self.partition_id,
                 'regime_id':r,'current_probability':float(self._arrays['current_probability'][i]),
                 'next_probability':float(self._arrays['next_probability'][i])} for i,r in enumerate(self.regime_ids)]

    def with_labels(self,labels,*,provenance):
        """Fresh state for supplied memberships; never mutates the real state.

        T08 owns shuffling and experiment policy. Its provenance is required
        so a distinct control definition cannot alias the original partition.
        """
        self._verify()
        if not isinstance(provenance,dict) or not provenance:
            raise ResearchError(ErrorCode.INVALID_CONFIG,'supplied memberships require provenance')
        partition=partition_from_labels(self._arrays['condition_scores'],labels,k_normal=len(self.regime_ids)-1)
        arrays=dict(self._arrays)
        update,transition=_partition_arrays(arrays['condition_scores'],arrays['current_scores'],self._metadata['condition_months'],partition)
        arrays.update(update)
        metadata=deepcopy(self._metadata)
        metadata.update(partition_diagnostics=partition.diagnostics,membership_provenance=deepcopy(provenance),
                        parent_partition_id=self.partition_id,skipped_transition_gaps=transition.skipped_gaps)
        return _freeze(metadata,arrays)

    def save(self,directory):
        self._verify(); directory=Path(directory)
        if any((directory/name).exists() for name in ('window_state.json','state.npz')):
            raise ResearchError(ErrorCode.RUN_CONFLICT,f'window state output already exists: {directory}')
        archive=save_npz(directory/'state.npz',self._arrays)
        manifest={'partition_id':self.partition_id,'metadata':self._metadata,'archive':archive}
        write_json(directory/'window_state.json',manifest)
        return manifest

    @classmethod
    def load(cls,directory):
        directory=Path(directory)
        record=read_json(directory/'window_state.json')
        arrays=load_npz(directory/'state.npz',record['archive'])
        state=cls(record['metadata'],arrays,record['partition_id'])
        state._verify()
        if state._metadata.get('version') != 1:
            raise ResearchError(ErrorCode.VERIFICATION_FAILED,'unknown window-state version')
        for a in arrays.values(): a.setflags(write=False)
        return state


def _validate_feature_result(result,preprocessor,months):
    if (result.feature_hash != preprocessor.feature_hash or result.transform_hash != preprocessor.transform_hash or
        list(result.scores.index) != months or list(result.imputed.index) != months or
        list(result.imputed.columns) != list(preprocessor.columns) or
        list(result.scores.columns) != [f'PC{i+1}' for i in range(preprocessor.n_components)]):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH,'feature result has mismatched dates, columns or preprocessing ID')
    masks=(result.observed_mask,result.ffill_mask,result.median_mask)
    if any(not mask.index.equals(result.imputed.index) or not mask.columns.equals(result.imputed.columns) for mask in masks):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH,'feature mask row/column mismatch')
    query=_digest({'months':months,'transform_hash':preprocessor.transform_hash},
        {'imputed':result.imputed.to_numpy(),'observed':masks[0].to_numpy(),'ffill':masks[1].to_numpy(),'median':masks[2].to_numpy()})
    if query != result.query_hash:
        raise ResearchError(ErrorCode.HASH_MISMATCH,'feature query contents changed under an existing ID')
    z=(result.imputed.to_numpy()-preprocessor.mean)/preprocessor.scale
    expected=(z-preprocessor.pca_mean)@preprocessor.components.T
    finite_array(result.scores.to_numpy(),ndim=2,name='feature scores')
    if not np.allclose(result.scores.to_numpy(),expected,atol=1e-12,rtol=1e-12):
        raise ResearchError(ErrorCode.HASH_MISMATCH,'PCA scores do not match fitted preprocessing')


def _build_window_state(features,settings=None,*,scope='rolling',window_kind='outer'):
    settings=settings or ResearchSettings(profile=features.ledger['profile'],lag_months=features.ledger['lag_months'])
    if scope not in ('rolling','full_sample') or settings.k_normal != 5:
        raise ResearchError(ErrorCode.INVALID_CONFIG,'research state requires a known scope and five normal regimes')
    prep=features.preprocessor
    prep.require_scope(scope)
    if settings.profile!=features.ledger['profile'] or settings.lag_months!=features.ledger['lag_months']:
        raise ResearchError(ErrorCode.PARTITION_MISMATCH,'regime settings disagree with feature timing policy')
    rows=features.training_rows.copy(deep=True)
    required=('target_month','predictor_base_month','condition_base_month')
    if any(c not in rows for c in required):
        raise ResearchError(ErrorCode.MISSING_DATA,'feature training rows lack date roles')
    for column in required: ordered_months(rows[column].tolist())
    if features.ledger.get('training_rows') != rows.to_dict(orient='records'):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH,'training rows disagree with timing ledger')
    if scope=='rolling':
        if window_kind=='outer' and (len(rows)!=48 or settings.train_months!=48):
            raise ResearchError(ErrorCode.INVALID_CONFIG,'outer rolling regime state requires 48 rows')
        ledger=features.ledger
        expected=decision_ledger(ledger['decision_month'],[ledger['vintage_month']],profile=ledger['profile'],lag_months=ledger['lag_months'],train_months=len(rows))
        if any(ledger.get(k)!=v for k,v in expected.items()):
            raise ResearchError(ErrorCode.VERIFICATION_FAILED,'regime window timing ledger mismatch')
    condition_months=rows.condition_base_month.tolist()
    if list(prep.fit_months)!=condition_months:
        raise ResearchError(ErrorCode.PARTITION_MISMATCH,'condition rows differ from fitted rows')
    arrays={}
    metadata={'version':1,'scope':scope,'window_kind':window_kind,'regime_ids':[f'R{i}' for i in range(6)],
              'feature_hash':prep.feature_hash,'transform_hash':prep.transform_hash,
              'feature_columns':list(prep.columns),'pc_columns':list(features.condition.scores.columns),
              'training_rows':rows.to_dict(orient='records'),'ledger':deepcopy(features.ledger),
              'preprocessor_metadata':deepcopy(prep.metadata),
              'transition_policy':'observed_adjacent_outgoing_counts; no_outgoing_self_loop',
              'literal_transition_policy':'paper_equation_5_total_occurrences; diagnostic_only',
              'paper_ambiguity':'AMB01: final occurrence has no outgoing transition',
              'membership_provenance':{'method':'algorithm_1'}}
    for role,months in [('condition',condition_months),('predictor',rows.predictor_base_month.tolist()),('current',[features.ledger['current_base_month']])]:
        result=getattr(features,role)
        _validate_feature_result(result,prep,months)
        metadata[role+'_months']=months
        metadata[role+'_query_hash']=result.query_hash
        arrays[role+'_scores']=result.scores.to_numpy(copy=True)
    # Copy all fitted arrays, preserving PCA, scaler, imputation and fit masks.
    arrays.update({'preprocessor_'+k:np.array(a,copy=True) for k,a in prep._arrays.items()})
    partition=fit_partition(arrays['condition_scores'],k_normal=5,seed=settings.seed,n_init=settings.n_init,max_iter=settings.max_iter,tol=settings.tol)
    update,transition=_partition_arrays(arrays['condition_scores'],arrays['current_scores'],condition_months,partition)
    arrays.update(update)
    metadata.update(partition_diagnostics=partition.diagnostics,skipped_transition_gaps=transition.skipped_gaps)
    return _freeze(metadata,arrays)


def build_window_state(features,settings=None,*,scope='rolling'):
    """Consume one FeatureWindow. Outer rolling requires exactly 48 rows.

    Full-sample states require their own full_sample preprocessing. T06 can
    instead use the pure fit_partition/probability APIs without a trade ledger.
    """
    return _build_window_state(features,settings,scope=scope,window_kind='outer')


def build_inner_window_state(features,settings=None):
    """Past-only shorter fold; preprocessing/state scope both remain rolling.

    The separately hashed window_kind='inner_fold' prevents an inner state
    being mistaken for an outer 48-row fit. Timing uses the same public ledger.
    """
    return _build_window_state(features,settings,scope='rolling',window_kind='inner_fold')


def checked_state_join(left,right,keys):
    if 'partition_id' not in left or 'partition_id' not in right:
        raise ResearchError(ErrorCode.MISSING_DATA,'both joined tables require partition_id')
    return checked_join(left,right,keys,partition_column='partition_id')
