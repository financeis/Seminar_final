"""Paper equations 1–4, 6 and 7; distances are never silently squared."""
from __future__ import annotations
import numpy as np
from ..contracts import ErrorCode, ResearchError


def finite_array(value, *, ndim=None, name='array'):
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{name} must be numeric') from exc
    if ndim is not None and a.ndim != ndim:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{name} must have {ndim} dimensions')
    if not np.isfinite(a).all():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'{name} contains NaN or infinity')
    return a


def unit_rows(values):
    """Normalize without overflowing large norms or underflowing tiny vectors."""
    a = finite_array(values, ndim=2)
    if not a.shape[1]:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'vectors need a coordinate')
    maximum = np.max(np.abs(a), axis=1, keepdims=True)
    scaled = np.divide(a, maximum, out=np.zeros_like(a), where=maximum != 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norm, out=np.zeros_like(a), where=norm != 0)


def cosine_distances(observations, centers):
    """Zero/zero distance is 0; zero/nonzero is 1. Output is n by k."""
    x, c = unit_rows(observations), unit_rows(centers)
    if x.shape[1] != c.shape[1] or not len(c):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'cosine coordinate dimensions disagree')
    distance = 1 - x @ c.T
    zero_x, zero_c = ~np.any(x, axis=1), ~np.any(c, axis=1)
    distance[np.ix_(zero_x, zero_c)] = 0
    if np.any(distance < -1e-12) or np.any(distance > 2 + 1e-12):
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'cosine distance outside rounding tolerance')
    return np.clip(distance, 0., 2.)


def distance_probabilities(distances):
    """Equation 1 on the final axis; all-zero distances give a uniform row."""
    d = finite_array(distances, name='distances')
    if d.ndim not in (1, 2) or d.shape[-1] < 2 or np.any(d < 0):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'distances need K >= 2 nonnegative entries')
    maximum = d.max(axis=-1, keepdims=True)
    scaled = np.divide(d, maximum, out=np.zeros_like(d), where=maximum != 0)
    total = scaled.sum(axis=-1, keepdims=True)
    fraction = np.divide(scaled, total, out=np.full_like(d, 1/d.shape[-1]), where=total != 0)
    return (1 - fraction) / (d.shape[-1] - 1)


def normalize_probabilities(values):
    p = finite_array(values, name='probabilities')
    if p.ndim not in (1, 2) or p.shape[-1] < 1 or np.any(p < 0):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'probabilities must be nonnegative vectors')
    maximum = p.max(axis=-1, keepdims=True)
    if np.any(maximum == 0):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'zero probability sum')
    scaled = p / maximum
    return scaled / scaled.sum(axis=-1, keepdims=True)


def combine_probabilities(crisis_probability, normal_probabilities):
    """q0=-max(Pnormal)*log2(1-p0), followed by sum normalization.

    The p0=1 limit is exact, so infinity is never an intermediate artifact.
    """
    normal = normalize_probabilities(normal_probabilities)
    p0 = finite_array(crisis_probability, name='crisis probability')
    if np.any((p0 < 0) | (p0 > 1)) or p0.shape != normal.shape[:-1]:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'incompatible crisis probabilities')
    is_one = p0 == 1
    safe = np.where(is_one, 0., p0)
    q0 = -normal.max(axis=-1) * np.log1p(-safe) / np.log(2.)
    result = normalize_probabilities(np.concatenate([np.expand_dims(q0, -1), normal], axis=-1))
    return np.where(np.expand_dims(is_one, -1), np.eye(1, normal.shape[-1]+1)[0], result)


def next_probabilities(current, transition):
    p = normalize_probabilities(current)
    e = finite_array(transition, ndim=2, name='transition matrix')
    if e.shape != (p.shape[-1], p.shape[-1]) or np.any(e < 0) or not np.allclose(e.sum(axis=1), 1., atol=1e-12, rtol=0):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'transition matrix must be row-stochastic')
    return normalize_probabilities(p @ e)


def regime_probabilities(observations, stage1_centers, normal_centers):
    """Soft assignment using ordered [crisis, remaining] L2 centers then cosine."""
    values = finite_array(observations, name='observations')
    single = values.ndim == 1
    x = values[None, :] if single else values
    c = finite_array(stage1_centers, ndim=2, name='stage1 centers')
    if x.ndim != 2 or c.shape != (2, x.shape[1]):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'stage1 centers require two compatible rows')
    with np.errstate(over='ignore', invalid='ignore'):
        distances = np.linalg.norm(x[:, None, :] - c[None, :, :], axis=-1)
    p0 = distance_probabilities(distances)[:, 0]
    normal = distance_probabilities(cosine_distances(x, normal_centers))
    result = combine_probabilities(p0, normal)
    return result[0] if single else result
