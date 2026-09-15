"""Full-sample diagnostics: independent arithmetic and isolation checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_alloc.contracts import ResearchError
from regime_alloc.reporting.regime_analysis import (
    AnalysisSettings, FullSampleAnalysis, conditional_departures, row_minmax,
    fit_full_sample, fit_gmm_comparison, summarize_indicators,
)


def sample():
    rng = np.random.default_rng(391)
    months = pd.period_range('1959-10', periods=122, freq='M').astype(str)
    values = rng.normal(size=(122, 12))
    values[:15] += 6
    raw = pd.DataFrame(values, index=months, columns=[f'X{i}' for i in range(12)])
    raw['RATE'] = np.linspace(1, 5, len(raw))
    codes = pd.Series(1, index=raw.columns)
    groups = pd.Series(1, index=raw.columns)
    groups['RATE'] = 6
    nber = pd.DataFrame({'month': months, 'usrec': np.zeros(len(months), dtype=int)})
    settings = AnalysisSettings(end_month=months[-1], n_init=2, gmm_n_init=2,
                                gmm_max_iter=500, indicators=('X0', 'RATE'))
    return raw, codes, groups, nber, settings


@pytest.fixture(scope='module')
def analysis():
    raw, codes, groups, nber, settings = sample()
    return fit_full_sample(raw, codes, groups, nber, settings=settings)


def test_conditional_probability_independent_fractions_and_no_departure():
    e = np.array([[.5, .3, .2], [0, 1, 0], [.1, .3, .6]])
    actual, no_departure = conditional_departures(e)
    np.testing.assert_allclose(actual, [[0, 3/5, 2/5], [0, 0, 0], [1/4, 3/4, 0]])
    np.testing.assert_array_equal(no_departure, [False, True, False])
    np.testing.assert_allclose(actual.sum(axis=1), [1, 0, 1])


@pytest.mark.parametrize('e', [[[1, .1], [0, 1]], [[-.1, 1.1], [0, 1]], [[np.nan]]])
def test_invalid_transition_rejected(e):
    with pytest.raises(ResearchError):
        conditional_departures(e)


def test_minmax_constants_and_missing_cells_are_distinct():
    normalized, constant = row_minmax([[2, 4, 10], [7, 7, 7], [np.nan, 0, 2]])
    np.testing.assert_allclose(normalized[:2], [[0, .25, 1], [0, 0, 0]])
    assert np.isnan(normalized[2, 0])
    np.testing.assert_allclose(normalized[2, 1:], [0, 1])
    np.testing.assert_array_equal(constant, [False, True, False])


def test_indicator_mean_uses_observed_rows_and_explicit_missing_reasons():
    frame = pd.DataFrame({'A': [2., 4., 8., np.nan], 'C': [7., 7., 7., 7.]})
    report = summarize_indicators(frame, [0, 0, 1, 1], indicators=('A', 'C', 'ABSENT'), n_regimes=2)
    assert report['means'][0] == [3., 8.]
    assert report['counts'][0] == [2, 1]
    assert report['constant'][1]
    assert report['records'][-1]['status'] == 'unavailable'
    assert report['records'][-1]['reason'] == 'series_absent'


def test_full_sample_scope_and_shared_pca_input(analysis):
    assert analysis.metadata['scope'] == 'full_sample'
    assert analysis.preprocessor.metadata['scope'] == 'full_sample'
    assert 'RATE' not in analysis.preprocessor.columns
    assert analysis.metadata['indicators']['raw']['records'][1]['series'] == 'RATE'
    with pytest.raises(ResearchError, match='scope'):
        analysis.require_scope('rolling')
    with pytest.raises(ResearchError, match='scope'):
        analysis.preprocessor.require_scope('rolling')
    np.testing.assert_array_equal(analysis.arrays['scores'], analysis.preprocessor.fit_result.scores)
    assert analysis.metadata['gmm']['input_hash'] == analysis.metadata['partition']['input_hash']
    assert analysis.arrays['probabilities'].shape == (120, 6)
    np.testing.assert_allclose(analysis.arrays['probabilities'].sum(axis=1), 1)
    np.testing.assert_allclose(analysis.arrays['gmm_probabilities'].sum(axis=1), 1)
    assert np.all((analysis.arrays['gmm_probabilities'] >= 0) & (analysis.arrays['gmm_probabilities'] <= 1))
    assert len(analysis.metadata['elbow']) == 10
    assert [row['k_normal'] for row in analysis.metadata['elbow']] == list(range(1, 11))


def test_nber_changes_only_ex_post_output(analysis):
    raw, codes, groups, nber, settings = sample()
    nber['usrec'] = 1
    changed = fit_full_sample(raw, codes, groups, nber, settings=settings)
    assert changed.model_id == analysis.model_id
    assert changed.analysis_id != analysis.analysis_id
    for name in ('scores', 'labels', 'gmm_labels', 'gmm_probabilities', 'transition_matrix'):
        np.testing.assert_array_equal(changed.arrays[name], analysis.arrays[name])


def test_gmm_parameter_replay(analysis):
    # Independent Gaussian log-density calculation from saved full covariances.
    a = analysis.arrays
    x = a['scores'][:3]
    log_density = []
    for weight, mean, cov in zip(a['gmm_weights'], a['gmm_means'], a['gmm_covariances']):
        z = x-mean
        log_density.append(np.log(weight)-.5*(x.shape[1]*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+
                                            np.sum(z*np.linalg.solve(cov, z.T).T, axis=1)))
    log_density = np.array(log_density).T
    density = np.exp(log_density-log_density.max(axis=1, keepdims=True))
    np.testing.assert_allclose(density/density.sum(axis=1, keepdims=True), a['gmm_probabilities'][:3], atol=1e-12)


def test_invalid_gmm_and_nonconvergence_are_errors(monkeypatch):
    from sklearn.mixture import GaussianMixture
    x = np.random.default_rng(7).normal(size=(40, 3))
    with pytest.raises(ResearchError):
        fit_gmm_comparison(x, np.zeros(40), settings=AnalysisSettings(gmm_reg_covar=0))
    original = GaussianMixture.fit
    def never_converged(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.converged_ = False
        return result
    monkeypatch.setattr(GaussianMixture, 'fit', never_converged)
    with pytest.raises(ResearchError, match='converg'):
        fit_gmm_comparison(x, np.arange(40)%6, settings=AnalysisSettings(gmm_n_init=1))


def test_save_load_exact_and_no_overwrite_or_tamper(analysis, tmp_path):
    directory = tmp_path/'full_sample'
    analysis.save(directory)
    restored = FullSampleAnalysis.load(directory)
    assert restored.analysis_id == analysis.analysis_id
    for key, value in analysis.arrays.items():
        np.testing.assert_array_equal(restored.arrays[key], value)
    with pytest.raises(ResearchError, match='exists'):
        analysis.save(directory)
    path = directory/'analysis.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    record['metadata']['scope'] = 'rolling'
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ResearchError):
        FullSampleAnalysis.load(directory)


def test_public_copies_and_private_tampering_guard(analysis):
    metadata = analysis.metadata
    metadata['scope'] = 'rolling'
    assert analysis.metadata['scope'] == 'full_sample'
    arrays = analysis.arrays
    arrays['labels'] = np.zeros(len(arrays['labels']))
    assert not np.array_equal(arrays['labels'], analysis.arrays['labels'])
    forged = FullSampleAnalysis(analysis.metadata, analysis.arrays, analysis.preprocessor, analysis.analysis_id)
    forged._arrays['labels'][0] = (forged._arrays['labels'][0]+1)%6
    with pytest.raises(ResearchError, match='changed'):
        forged.save(Path('this_must_not_be_created'))


def test_r0_months_extremes_and_interpretation_are_explicit(analysis):
    outliers = analysis.metadata['outlier_diagnostics']
    months = np.array(analysis.metadata['months'])[analysis.arrays['labels'] == 0].tolist()
    assert [row['month'] for row in outliers] == months
    z = analysis.preprocessor.fit_result.standardized
    for record in outliers:
        top = record['standardized_extremes'][0]
        assert abs(top['standardized_value']) == np.abs(z.loc[record['month']]).max()
        assert top['series'] != 'RATE'
    assert 'not NBER' in analysis.metadata['table1'][0]['group_definition']


def test_incomplete_nber_and_missing_raw_month_rejected():
    raw, codes, groups, nber, settings = sample()
    with pytest.raises(ResearchError):
        fit_full_sample(raw, codes, groups, nber.iloc[5:], settings=settings)
    with pytest.raises(ResearchError):
        fit_full_sample(raw.drop(index=raw.index[8]), codes, groups, nber, settings=settings)


def test_plot_inventory_render_and_no_overwrite(analysis, tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl-cache'))
    from regime_alloc.reporting.regime_plots import plot_regime_analysis
    result = plot_regime_analysis(analysis, tmp_path/'plots', language='en')
    assert {item['paper_id'] for item in result['figures']} >= {f'Figure {i}' for i in range(1, 7)}
    for item in result['figures']:
        assert (tmp_path/'plots'/item['png']).read_bytes().startswith(b'\x89PNG')
        svg = (tmp_path/'plots'/item['svg']).read_text(encoding='utf-8')
        assert '<svg' in svg
        assert '<dc:date>' not in svg
    repeated = plot_regime_analysis(analysis, tmp_path/'repeated', language='en')
    for original, replay in zip(result['figures'], repeated['figures']):
        assert original['png_sha256'] == replay['png_sha256']
        assert original['svg_sha256'] == replay['svg_sha256']
    with pytest.raises(ResearchError, match='exists'):
        plot_regime_analysis(analysis, tmp_path/'plots', language='en')
