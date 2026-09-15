"""Deterministic two-stage partition with genuine spherical center updates."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..contracts import ErrorCode, ResearchError
from .probabilities import finite_array, unit_rows, cosine_distances
from .transitions import integer_labels

ZERO_POLICY = 'both_zero_distance_0_one_zero_distance_1'


def _settings(k, n_init, max_iter, tol, seed):
    if (any(type(x) is not int or x < 1 for x in (k,n_init,max_iter)) or
        type(seed) is not int or seed < 0 or not np.isfinite(tol) or tol <= 0):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid clustering configuration')


def _lex_order(centers):
    return np.array(sorted(range(len(centers)), key=lambda i: tuple(centers[i])), dtype=int)


def _canonical(labels, centers):
    order = _lex_order(centers)
    inverse = np.empty(len(order), dtype=int); inverse[order] = np.arange(len(order))
    return inverse[labels], centers[order]


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    iterations: int
    diagnostics: dict


@dataclass(frozen=True)
class PartitionResult:
    labels: np.ndarray
    centers: np.ndarray
    stage1_centers: np.ndarray
    diagnostics: dict


def _spherical_centers(x, labels, k):
    centers = []
    for group in range(k):
        members = x[labels == group]
        if not len(members):
            raise ResearchError(ErrorCode.DEGENERATE_PARTITION, 'empty spherical cluster')
        center = members.mean(axis=0)
        # Opposing directions can cancel. A deterministic medoid direction is
        # a valid minimizer of their zero-resultant spherical objective.
        if np.linalg.norm(center) <= 1e-14 and np.any(members):
            center = members[_lex_order(members)[0]]
        centers.append(center)
    return unit_rows(centers)


def spherical_kmeans(observations, k, *, seed=0, n_init=20, max_iter=300, tol=1e-6, initial_centers=None):
    """Minimize summed cosine distance. k=1 is valid for elbow diagnostics.

    Unique directions use 12 decimal digits, including the zero direction.
    Empty groups receive the worst-explained distinct direction, row order
    breaking ties. Final cluster numbers follow lexicographic center order.
    """
    _settings(k,n_init,max_iter,tol,seed)
    x = unit_rows(observations)
    if len(x) < k or len(np.unique(np.round(x,12),axis=0)) < k:
        raise ResearchError(ErrorCode.DEGENERATE_PARTITION, 'too few distinct directions for spherical partition')
    initial = None if initial_centers is None else unit_rows(initial_centers)
    if initial is not None and initial.shape != (k,x.shape[1]):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'initial centers shape mismatch')
    rng = np.random.default_rng(seed)
    best = None
    for restart in range(n_init):
        if initial is not None and restart == 0:
            centers = initial.copy()
        else:
            selected = [int(rng.integers(len(x)))]
            while len(selected) < k:
                nearest = cosine_distances(x,x[selected]).min(axis=1)
                nearest[selected] = 0
                if nearest.sum() <= 1e-14: break
                selected.append(int(rng.choice(len(x),p=nearest/nearest.sum())))
            if len(selected) != k: continue
            centers = x[selected].copy()
        empty_reseeds = 0
        previous = None
        converged = False
        failed = False
        for iteration in range(1,max_iter+1):
            distances = cosine_distances(x,centers)
            labels = distances.argmin(axis=1)
            counts = np.bincount(labels,minlength=k)
            for empty in np.flatnonzero(counts == 0):
                occupied = centers[counts > 0]
                distinct = cosine_distances(x,occupied).min(axis=1) > 1e-12
                choices = [i for i in range(len(x)) if counts[labels[i]] > 1 and distinct[i]]
                if not choices:
                    failed = True; break
                point = min(choices,key=lambda i:(-distances[i,labels[i]],i))
                counts[labels[point]] -= 1
                labels[point] = empty; counts[empty] = 1
                centers[empty] = x[point]
                empty_reseeds += 1
            if failed: break
            updated = _spherical_centers(x,labels,k)
            movement = np.max(np.linalg.norm(updated-centers,axis=1))
            # A stable assignment is the fixed point. Tolerance alone must not
            # leave reported labels inconsistent with the updated centers.
            same_assignment = np.array_equal(cosine_distances(x,updated).argmin(axis=1),labels)
            centers = updated
            if same_assignment and (previous is not None and np.array_equal(previous,labels) or movement <= tol):
                converged = True; break
            previous = labels.copy()
        if failed or len(np.unique(labels)) != k or len(np.unique(np.round(centers,12),axis=0)) != k:
            continue
        labels, centers = _canonical(labels,centers)
        objective = float(cosine_distances(x,centers)[np.arange(len(x)),labels].sum())
        key = (objective, tuple(centers.ravel()), tuple(labels))
        if best is None or key < best[0]:
            best = (key,ClusterResult(labels.copy(),centers.copy(),objective,iteration,
                {'seed':seed,'n_init':n_init,'max_iter':max_iter,'tol':float(tol),'converged':converged,
                 'empty_reseeds':empty_reseeds,'zero_vector_count':int((~np.any(x,axis=1)).sum()),
                 'zero_vector_policy':ZERO_POLICY,'direction_decimals':12,'restart':restart}))
    if best is None:
        raise ResearchError(ErrorCode.DEGENERATE_PARTITION, 'no valid spherical partition after restarts')
    return best[1]


def fit_partition(observations, *, k_normal=5, seed=0, n_init=20, max_iter=300, tol=1e-6):
    """Algorithm 1: smaller L2 group R0, spherical normal groups R1...Rk."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    _settings(k_normal,n_init,max_iter,tol,seed)
    x = finite_array(observations,ndim=2,name='partition observations')
    if len(x) < k_normal+1 or not x.shape[1] or len(np.unique(x,axis=0)) < 2:
        raise ResearchError(ErrorCode.DEGENERATE_PARTITION, 'too few distinct points for two-stage partition')
    with threadpool_limits(1):
        first = KMeans(n_clusters=2,n_init=n_init,max_iter=max_iter,tol=tol,random_state=seed,algorithm='lloyd').fit(x)
    counts = np.bincount(first.labels_,minlength=2)
    if np.any(counts == 0):
        raise ResearchError(ErrorCode.DEGENERATE_PARTITION, 'L2 partition contains an empty group')
    crisis = min(range(2),key=lambda i:(int(counts[i]),tuple(first.cluster_centers_[i])))
    normal_mask = first.labels_ != crisis
    normal = spherical_kmeans(x[normal_mask],k_normal,seed=seed,n_init=n_init,max_iter=max_iter,tol=tol)
    labels = np.zeros(len(x),dtype=np.int64); labels[normal_mask] = normal.labels+1
    stage1 = np.vstack([x[~normal_mask].mean(axis=0), x[normal_mask].mean(axis=0)])
    centers = np.vstack([stage1[0],normal.centers])
    return PartitionResult(labels,centers,stage1,
        {'method':'l2_then_spherical','k_normal':k_normal,'seed':seed,'n_init':n_init,'max_iter':max_iter,'tol':float(tol),
         'stage1_counts': [int((~normal_mask).sum()),int(normal_mask.sum())],
         'stage1_tie_policy':'lexicographically_smallest_center','stage1_inertia':float(first.inertia_),
         'stage1_iterations':int(first.n_iter_),'spherical':normal.diagnostics,'spherical_inertia':normal.inertia,
         'zero_vector_count':int((~np.any(x,axis=1)).sum()),'zero_vector_policy':ZERO_POLICY})


def partition_from_labels(observations, labels, *, k_normal=5):
    """Recompute centers for caller-supplied memberships (e.g. random controls).

    No shuffling or reordering is performed here. Labels must occupy every
    regime, and callers own the seed/policy that produced those labels.
    """
    x = finite_array(observations,ndim=2,name='partition observations')
    labels = integer_labels(labels,k_normal+1,len(x))
    if len(np.unique(labels)) != k_normal+1:
        raise ResearchError(ErrorCode.DEGENERATE_PARTITION, 'supplied labels must occupy every regime')
    stage1 = np.vstack([x[labels==0].mean(axis=0),x[labels!=0].mean(axis=0)])
    normal = _spherical_centers(unit_rows(x[labels!=0]),labels[labels!=0]-1,k_normal)
    return PartitionResult(labels.copy(),np.vstack([stage1[0],normal]),stage1,
        {'method':'supplied_memberships','k_normal':k_normal,'zero_vector_count':int((~np.any(x,axis=1)).sum()),
         'zero_vector_policy':ZERO_POLICY})
