"""Independent hand calculations for the paper's soft-regime equations."""
import numpy as np
import pytest

from regime_alloc.contracts import ResearchError
from regime_alloc.regimes import (distance_probabilities, combine_probabilities,
    cosine_distances, transition_matrices, next_probabilities)


def test_distance_formula_uses_distances_not_squares():
    np.testing.assert_allclose(distance_probabilities([1., 2., 3.]), [5/12, 1/3, 1/4])
    np.testing.assert_allclose(distance_probabilities([0., 1., 1., 1., 1.]), [.25, .1875, .1875, .1875, .1875])
    np.testing.assert_array_equal(distance_probabilities([0., 0.]), [.5, .5])


@pytest.mark.parametrize('values', [[1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.], []])
def test_invalid_distances_are_rejected(values):
    with pytest.raises(ResearchError): distance_probabilities(values)


def test_finite_huge_distances_do_not_overflow():
    np.testing.assert_allclose(distance_probabilities([1e308, 1e308]), [.5, .5])


def test_log2_and_exact_endpoints():
    normals = np.array([.6, .4])
    np.testing.assert_allclose(combine_probabilities(.5, normals), [.375, .375, .25])
    np.testing.assert_allclose(combine_probabilities(.75, normals), [6/11, 3/11, 2/11])
    np.testing.assert_array_equal(combine_probabilities(0., normals), [0., .6, .4])
    np.testing.assert_array_equal(combine_probabilities(1., normals), [1., 0., 0.])


@pytest.mark.parametrize('p0,normal', [(-.1,[.5,.5]),(1.1,[.5,.5]),(np.nan,[.5,.5]),(.5,[0,0]),(.5,[-.1,1.1]),(.5,[np.inf,1])])
def test_invalid_probability_inputs(p0, normal):
    with pytest.raises(ResearchError): combine_probabilities(p0, normal)


def test_cosine_zero_and_opposite_vectors():
    got = cosine_distances([[0,0],[1,0],[-1,0]], [[0,0],[1,0]])
    np.testing.assert_allclose(got, [[0,1],[1,0],[1,2]])
    np.testing.assert_allclose(cosine_distances([[1e300,1e300]], [[1e-300,1e-300]]), [[0]], atol=1e-15)


def test_transition_denominators_and_gaps_hand_calculated():
    result = transition_matrices([0, 1, 1, 2], ['2000-01','2000-02','2000-03','2000-04'], 3)
    np.testing.assert_allclose(result.matrix, [[0,1,0],[0,.5,.5],[0,0,1]])
    np.testing.assert_allclose(result.literal_matrix, [[0,1,0],[0,.5,.5],[0,0,0]])
    np.testing.assert_array_equal(result.counts, [[0,1,0],[0,1,1],[0,0,0]])
    np.testing.assert_array_equal(result.no_outgoing, [False,False,True])
    gaps = transition_matrices([0,1,2], ['2000-01','2000-03','2000-04'], 3)
    np.testing.assert_array_equal(gaps.matrix, [[1,0,0],[0,0,1],[0,0,1]])
    assert gaps.skipped_gaps == 1


def test_row_vector_transition_and_joint_permutation_invariance():
    p = np.array([.2,.5,.3])
    e = np.array([[.5,.5,0],[0,.4,.6],[1,0,0]])
    expected = np.array([.4,.3,.3])
    np.testing.assert_allclose(next_probabilities(p,e), expected)
    forecasts = np.array([[.1,.2],[.3,-.4],[.5,.6]])
    order = np.array([2,0,1])
    permuted = next_probabilities(p[order], e[np.ix_(order,order)])
    np.testing.assert_allclose(permuted @ forecasts[order], expected @ forecasts)


def test_transition_duplicate_unsorted_bad_labels_rejected():
    for labels, months in [([0,1],['2000-01','2000-01']),([0,1],['2000-02','2000-01']),([0,.5],['2000-01','2000-02'])]:
        with pytest.raises(ResearchError): transition_matrices(labels,months,2)
    with pytest.raises(ResearchError): next_probabilities([.5,.5], [[.5,0],[0,1]])
