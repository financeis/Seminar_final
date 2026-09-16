"""Interpretation names must be compared in original transformed units."""
import numpy as np
import pandas as pd
from regime_alloc.regimes import match_centers


def test_current_scale_hungarian_and_crisis_fixed():
    old=pd.DataFrame([[0,0],[0,10],[10,0]],index=['R0','R1','R2'],columns=['a','b'])
    new=old.iloc[[0,2,1]].copy(); new.index=old.index
    result=match_centers(old,new,pd.Series({'a':2.,'b':5.}))
    assert result.mapping=={'R0':'R0','R1':'R2','R2':'R1'}
    assert result.status=='matched'
    np.testing.assert_allclose(result.cost_matrix,[[np.sqrt(29),0],[0,np.sqrt(29)]])


def test_pca_sign_rotation_and_component_count_reconstruction():
    raw=np.array([[0.,0,0],[1,2,0],[3,-1,0]])
    rotation=np.array([[0.,-1],[1,0]])
    old_scores=raw[:,:2]@rotation
    old_restored=old_scores@rotation.T
    # A second PCA keeps a third zero component and reverses the first axis.
    components=np.diag([-1.,1.,1.])
    new_scores=raw[[0,2,1]]@components.T
    new_restored=new_scores@components
    old=pd.DataFrame(np.column_stack([old_restored,np.zeros(3)]),index=['R0','R1','R2'],columns=list('abc'))
    new=pd.DataFrame(new_restored,index=old.index,columns=old.columns)
    result=match_centers(old,new,pd.Series(1.,index=old.columns))
    assert result.mapping=={'R0':'R0','R1':'R2','R2':'R1'}


def test_common_columns_only_and_disconnected_ids():
    old=pd.DataFrame([[0],[1],[2]],index=['R0','R1','R2'],columns=['a'])
    new=pd.DataFrame([[0],[2],[1]],index=old.index,columns=['b'])
    result=match_centers(old,new,pd.Series({'b':1.}))
    assert result.status=='disconnected'
    assert result.common_columns==()
    assert len(set(result.interpretation_ids.values()))==3
    assert result.interpretation_ids != match_centers(old,new+1,pd.Series({'b':1.})).interpretation_ids
