"""Interpretation-only matching in common transformed economic coordinates."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
import pandas as pd
from ..contracts import ErrorCode, ResearchError, canonical_id


@dataclass(frozen=True)
class MatchResult:
    mapping: dict[str,str]
    interpretation_ids: dict[str,str]
    common_columns: tuple[str,...]
    cost_matrix: np.ndarray
    status: str
    reason: str
    match_id: str


def _frame(frame):
    if (not isinstance(frame,pd.DataFrame) or not frame.index.is_unique or
        not frame.columns.is_unique or len(frame)<2 or list(frame.index)!=[f'R{i}' for i in range(len(frame))]):
        raise ResearchError(ErrorCode.INVALID_CONFIG,'centers need unique ordered R0...Rk rows and columns')
    try: return frame.astype(float)
    except (ValueError,TypeError) as exc:
        raise ResearchError(ErrorCode.INVALID_CONFIG,'center coordinates must be numeric') from exc


def _safe_frame(frame):
    # Nonfinite coordinates are explicitly unusable, and still identity-bound.
    return {'rows':list(frame.index),'columns':list(frame.columns),
            'values':[[float(v) if np.isfinite(v) else str(v) for v in row] for row in frame.to_numpy()]}


def match_centers(previous_centers,current_centers,current_scale,*,previous_ids=None,context=None):
    """Hungarian assignment after division by the *current* feature scales.

    Inputs are already inverse-PCA/inverse-scaled, in t-code transformed units.
    R0 is fixed. A missing common valid coordinate breaks the interpretation
    chain and generates new IDs. No predictive labels/probabilities change.
    """
    # SciPy's Windows import probes the operating system through subprocesses;
    # keep it inside the requested calculation, outside package import.
    from scipy.optimize import linear_sum_assignment
    previous,current=_frame(previous_centers),_frame(current_centers)
    if not isinstance(current_scale,(pd.Series,Mapping)):
        raise ResearchError(ErrorCode.INVALID_CONFIG,'current scale must name its features')
    scale=pd.Series(current_scale,dtype=float)
    if not scale.index.is_unique:
        raise ResearchError(ErrorCode.DUPLICATE_KEY,'duplicate feature scale keys')
    common=tuple(c for c in current.columns if c in previous and c in scale and
                 np.isfinite(scale[c]) and scale[c]>0 and np.isfinite(previous[c]).all() and np.isfinite(current[c]).all())
    previous_ids = dict(previous_ids) if previous_ids is not None else {
        r:canonical_id({'centers':_safe_frame(previous),'regime':r}) for r in previous.index}
    if set(previous_ids)!=set(previous.index) or len(set(previous_ids.values()))!=len(previous_ids):
        raise ResearchError(ErrorCode.INVALID_CONFIG,'previous interpretation IDs must uniquely cover all regimes')
    identity={'previous':_safe_frame(previous),'current':_safe_frame(current),
              'scale':{str(k):float(v) if np.isfinite(v) else str(v) for k,v in scale.items()},
              'previous_ids':previous_ids,'context':context or {},'common_columns':common}
    match_id=canonical_id(identity)
    if not common or len(previous)!=len(current):
        reason='no_common_valid_features' if not common else 'regime_count_changed'
        ids={r:canonical_id({'new_chain':match_id,'regime':r}) for r in current.index}
        return MatchResult({'R0':'R0'},ids,common,np.empty((0,0)), 'disconnected',reason,match_id)
    old=previous.loc[previous.index[1:],list(common)].to_numpy()/scale[list(common)].to_numpy()
    new=current.loc[current.index[1:],list(common)].to_numpy()/scale[list(common)].to_numpy()
    with np.errstate(over='ignore',invalid='ignore'):
        cost=np.linalg.norm(new[:,None,:]-old[None,:,:],axis=2)
    if not np.isfinite(cost).all():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE,'matching distance overflow')
    row,col=linear_sum_assignment(cost)
    mapping={'R0':'R0',**{str(current.index[i+1]):str(previous.index[j+1]) for i,j in zip(row,col)}}
    ids={r:previous_ids[mapping[r]] for r in current.index}
    return MatchResult(mapping,ids,common,cost,'matched','common_feature_current_scale_hungarian',match_id)


def match_window_states(previous,current,*,previous_ids=None):
    return match_centers(previous.centers_in_feature_units(),current.centers_in_feature_units(),
                         current.feature_scale,previous_ids=previous_ids,
                         context={'previous_partition_id':previous.partition_id,'current_partition_id':current.partition_id})
