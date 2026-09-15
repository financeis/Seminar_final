"""T6: independent metrics and bounded, offline original-script integration.

PASS applies only to the stated property. A working CSV handoff cannot establish
correct regime identities or empirical replication of the paper's performance.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import warnings

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_tests.common import TICKERS, jsonable, load_functions, record, sha256, source_ref, write_results
from audit_tests.metrics_support import environment, install_network_guard, original_hashes, run_original_script, working_directory, write_json
from audit_tests.test_data_timing import assigned, exec_nodes, prepare, ref_range, step1_first_decision, step2_first_decision, tree_of

S1, S2, S3, S4, S5 = [f'Section5_step{i}.py' for i in range(1, 6)]
VIS = 'etc_visualize_backtest.py'
RET = 'section5_step5_longonly_backtest_returns.csv'
MET = 'section5_step5_longonly_backtest_metrics.csv'
ETF = 'etf_bom_returns_aligned_demo.csv'
WEIGHTS = 'section5_step4_longonly_weights.csv'
TOL = {'atol': 1e-10, 'rtol': 1e-10, 'equal_nan': True}


def close(a, b):
    return bool(np.allclose(a, b, **TOL))


def sample_sd(values):
    if len(values) < 2:
        return math.nan
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((v - mean)**2 for v in values) / (len(values)-1))


def drawdown_oracle(values, *, initial_wealth=True):
    wealth, peak, history, dd = 1.0, (1.0 if initial_wealth else None), [], []
    for value in values:
        wealth *= 1.0 + (0.0 if math.isnan(value) else value)
        peak = wealth if peak is None else max(peak, wealth)
        history.append(wealth)
        dd.append(wealth / peak - 1.0)
    negative = [x for x in dd if x < 0]
    return {'wealth': history, 'drawdowns': dd, 'AvgDD': math.fsum(negative)/len(negative) if negative else 0.0,
            'MaxDD': min(dd) if dd else math.nan}


def metrics_oracle(values, *, initial_wealth=True):
    values = [float(v) for v in values if not math.isnan(float(v))]
    mean = math.fsum(values)/len(values) if values else math.nan
    sd = sample_sd(values)
    negative_sd = sample_sd([v for v in values if v < 0])
    downside_rms = math.sqrt(math.fsum(min(v, 0)**2 for v in values)/len(values)) if values else math.nan
    ratio = lambda divisor: math.sqrt(12)*mean/divisor if math.isfinite(divisor) and divisor > 0 else math.nan
    dd = drawdown_oracle(values, initial_wealth=initial_wealth)
    return {'Sharpe': ratio(sd), 'Sortino': ratio(negative_sd), 'Sortino_downside_RMS_reference': ratio(downside_rms),
            'downside_negative_sample_SD': negative_sd, 'downside_RMS_all_months': downside_rms,
            'AvgDD': dd['AvgDD'], 'MaxDD': dd['MaxDD'],
            '% Positive Ret.': sum(v > 0 for v in values)/len(values) if values else math.nan}


def scalar_portfolio(weights, returns):
    return math.fsum(float(w)*float(r) for w, r in zip(weights, returns))


def independent_lo(values, l):
    chosen = sorted([i for i, value in enumerate(values) if value > 0], key=lambda i: values[i], reverse=True)[:l]
    denom = math.fsum(values[i] for i in chosen)
    return [values[i]/denom if i in chosen else 0.0 for i in range(len(values))]


def fixture_weights(dates, vectors):
    rows = []
    for t, tp1, vector in zip(dates[:-1], dates[1:], vectors):
        row = {'sasdate_t': t, 'sasdate_tp1': tp1}
        for l in [2, 3, 4]:
            row.update({f'w_lo_{l}_{ticker}': float(value) for ticker, value in zip(TICKERS, vector)})
        rows.append(row)
    return pd.DataFrame(rows)


def preserve_files(source, target):
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(Path(source).glob('*.csv')):
        out = target / path.name
        shutil.copy2(path, out)
        manifest.append({'path': str(out), 'sha256': sha256(out)})
    return manifest


def run(repo: Path, output: Path) -> list[dict]:
    repo, output = Path(repo).resolve(), Path(output).resolve()
    dest = output / 'metrics_integration'
    dest.mkdir(parents=True, exist_ok=True)
    results, executions, details = [], [], {}
    functions = load_functions(repo, S5, ['compute_drawdowns', 'perf_metrics'])
    refs = [source_ref(repo, S5, name) for name in ['compute_drawdowns', 'perf_metrics']]

    def add(number, title, passed, expected, observed, *, source_refs=(), inputs=None, oracle='', classification='defined_property', status=None, notes=()):
        item = record(f'MET-{number:03}', title, passed, expected, observed, source_refs=source_refs,
                      inputs=inputs, oracle=oracle, tolerance=TOL, notes=notes, status=status)
        item['classification'] = classification
        results.append(item)
        # Keep finished checks reviewable even if a later unexpected harness failure occurs.
        write_results(output, 'metrics_integration', results)

    controls = {'initial_loss': drawdown_oracle([-.1, .05]), 'metrics': metrics_oracle([.1, -.1, 0]),
                'nonzero_sharpe': metrics_oracle([.1, 0]), 'two_losses': metrics_oracle([-.1, -.3]),
                'portfolio': scalar_portfolio([.6, .4], [.1, -.05]), 'lo': independent_lo([2., 1., -3.], 2)}
    control_ok = (close(controls['initial_loss']['wealth'], [.9, .945]) and
                  close(controls['initial_loss']['drawdowns'], [-.1, -.055]) and
                  close(controls['initial_loss']['AvgDD'], -.0775) and
                  close(controls['initial_loss']['MaxDD'], -.1) and
                  close(controls['metrics']['Sharpe'], 0) and
                  close(controls['metrics']['% Positive Ret.'], 1/3) and
                  close(controls['nonzero_sharpe']['Sharpe'], math.sqrt(6)) and
                  close(controls['two_losses']['downside_negative_sample_SD'], math.sqrt(.02)) and
                  close(controls['two_losses']['Sortino'], -math.sqrt(24)) and
                  close(controls['two_losses']['Sortino_downside_RMS_reference'], -math.sqrt(9.6)) and
                  close(controls['portfolio'], .04) and close(controls['lo'], [2/3, 1/3, 0]))
    add(1, 'Independent scalar metrics, wealth, portfolio and sizing hand controls', control_ok,
        {'wealth': [.9, .945], 'drawdowns': [-.1, -.055], 'AvgDD': -.0775, 'MaxDD': -.1, 'portfolio': .04, 'lo': [2/3, 1/3, 0],
         'nonzero_Sharpe': math.sqrt(6), 'negative_sample_SD': math.sqrt(.02), 'negative_Sortino': -math.sqrt(24), 'RMS_reference_Sortino': -math.sqrt(9.6)},
        controls, inputs='Literal hand-computable arrays; no random seed', oracle='Scalar recurrence and sums; no pandas cumprod/std or original functions')
    assert control_ok, 'Independent oracle failed its hand controls'

    fixtures = {'initial_loss': [-.1, .05], 'no_loss': [0, .01, .02, .03],
                'one_loss': [.04, -.02, .01, .03], 'constant': [.02]*4,
                'ordinary': [.03, -.04, .01, -.02, .03, 0], 'nan': [.02, math.nan, -.01, .03]}
    observed, expected, wealth_cases = {}, {}, {}
    for name, values in fixtures.items():
        actual = functions['perf_metrics'](pd.Series(values, dtype=float))
        desired = metrics_oracle(values, initial_wealth=False)
        observed[name], expected[name] = actual, desired
        avg, maximum, wealth = functions['compute_drawdowns'](pd.Series(values, dtype=float))
        wealth_cases[name] = {'AvgDD': avg, 'MaxDD': maximum, 'wealth': wealth.tolist(), 'reference': drawdown_oracle(values, initial_wealth=False)}
    behavior_ok = all(close(list(actual.values()), [expected[name][key] for key in actual]) for name, actual in observed.items())
    add(2, 'Original metrics agree with independently calculated implemented conventions across six cases', behavior_ok,
        expected, observed, source_refs=refs, inputs=fixtures,
        oracle='sqrt(12)*mean/sample SD (ddof=1), rf=0; negative-return sample SD for original Sortino; drawdown peak starts after first return; positive fraction',
        classification='implementation_convention', notes=['Agreement establishes implemented arithmetic only, not that the omitted initial wealth is correct.'])
    wealth_ok = all(close(v['wealth'], v['reference']['wealth']) and close([v['AvgDD'], v['MaxDD']], [v['reference']['AvgDD'], v['reference']['MaxDD']]) for v in wealth_cases.values())
    add(3, 'Original drawdown routine fills NaN with zero; metric wrapper drops NaN before evaluation', wealth_ok,
        {name: value['reference'] for name, value in wealth_cases.items()}, wealth_cases, source_refs=refs,
        inputs=fixtures, oracle='Scalar multiplicative wealth with NaN treated as zero only for direct drawdown call', classification='implementation_convention')
    correct_first = drawdown_oracle(fixtures['initial_loss'])
    add(4, 'A first-month loss is included when peak starts at initial wealth 1',
        close([observed['initial_loss']['AvgDD'], observed['initial_loss']['MaxDD']], [correct_first['AvgDD'], correct_first['MaxDD']]),
        correct_first, observed['initial_loss'], source_refs=refs, inputs=fixtures['initial_loss'],
        oracle='Initial investor wealth=1, wealth=.9,.945; running peak stays 1; AvgDD=-.0775, MaxDD=-.1', classification='initial_wealth_omission')
    sortino_diff = {name: {'source': observed[name]['Sortino'], 'declared_RMS_reference': expected[name]['Sortino_downside_RMS_reference'],
                         'negative_sample_SD': expected[name]['downside_negative_sample_SD'], 'all_month_RMS': expected[name]['downside_RMS_all_months']} for name in fixtures}
    add(5, 'Sortino denominator is negative-subset sample SD; reference uses downside RMS over all months',
        all(close(observed[name]['Sortino'], expected[name]['Sortino']) for name in fixtures) and
        not close(observed['ordinary']['Sortino'], expected['ordinary']['Sortino_downside_RMS_reference']),
        'Observe the documented code convention and its difference from the explicitly declared reference', sortino_diff,
        source_refs=[refs[1]], inputs=fixtures,
        oracle='Reference downside=sqrt(sum(min(r,0)^2)/n), target=0; original downside is sample SD of negatives only',
        classification='paper_ambiguity', notes=['Paper Section 5.4 does not define this denominator. Difference is not a confirmed paper-formula violation.',
                                               'No losses or one negative return make original Sortino NaN; constant positive returns make Sharpe and Sortino NaN.'])
    unit_observation = {'first_loss_MaxDD': observed['initial_loss']['MaxDD'], 'one_loss_positive_fraction': observed['one_loss']['% Positive Ret.'],
                        'ordinary_AvgDD': observed['ordinary']['AvgDD'], 'ordinary_MaxDD': observed['ordinary']['MaxDD']}
    add(6, 'Stored drawdowns and positive-return proportion use fractions, despite percent column title',
        close(unit_observation['one_loss_positive_fraction'], .75) and close(unit_observation['ordinary_MaxDD'], -.049792),
        {'positive_fraction': .75, 'ordinary_MaxDD': -.049792}, unit_observation,
        source_refs=refs, inputs=fixtures, oracle='Monthly returns and drawdowns are decimal fractions; .75 is 75%; AvgDD averages only negative drawdown periods', classification='units')

    # The actual Step5 loop is executed as an unchanged complete script.
    dates = pd.date_range('2001-01-01', periods=5, freq='MS')
    vectors = [[.6, .4]+[0.]*8, [1.]+[0.]*9, [1.]+[0.]*9, [1.]+[0.]*9]
    w = fixture_weights(dates, vectors)
    return_rows = [[.1, -.05]+[0.]*8, [-.2]+[0.]*9, [math.nan, .1]+[0.]*8]
    R = pd.DataFrame(return_rows, index=dates[[1, 2, 4]], columns=TICKERS)
    with tempfile.TemporaryDirectory(prefix='paper-audit-step5-') as tmp:
        tmp = Path(tmp)
        w.to_csv(tmp / WEIGHTS, index=False)
        R.to_csv(tmp / ETF)
        execution = run_original_script(repo, S5, tmp)
        executions.append(execution)
        if execution['exit_code'] != 0:
            raise RuntimeError('Unexpected original Step5 normal-fixture failure: ' + execution['stderr'])
        actual_returns = pd.read_csv(tmp / RET, index_col=0, parse_dates=True)
        actual_metrics = pd.read_csv(tmp / MET)
        details['loop_files'] = preserve_files(tmp, dest / 'loop_fixture')
        expected_regular = [scalar_portfolio(vectors[0], return_rows[0]), scalar_portfolio(vectors[1], return_rows[1])]
        metric_map = {'ridge_lo_2': 'ret_lo_2', 'ridge_lo_3': 'ret_lo_3', 'ridge_lo_4': 'ret_lo_4', 'spy': 'ret_spy', 'ew': 'ret_ew'}
        expected_csv_metrics = {model: metrics_oracle(actual_returns[col].tolist(), initial_wealth=False) for model, col in metric_map.items()}
        csv_metrics_ok = len(actual_metrics) == len(metric_map)
        for _, item in actual_metrics.iterrows():
            csv_metrics_ok = csv_metrics_ok and item['Model'] in expected_csv_metrics and close(
                [item[col] for col in ['Sharpe', 'Sortino', 'AvgDD', 'MaxDD', '% Positive Ret.']],
                [expected_csv_metrics[item['Model']][col] for col in ['Sharpe', 'Sortino', 'AvgDD', 'MaxDD', '% Positive Ret.']])
        add(7, 'Actual Step5 loop multiplies weights by sasdate_tp1 returns; SPY and EW hand controls',
            all(close(actual_returns[f'ret_lo_{l}'].iloc[:2], expected_regular) for l in [2, 3, 4]) and
            close(actual_returns['ret_spy'].iloc[:2], [.1, -.2]) and close(actual_returns['ret_ew'].iloc[:2], [.005, -.02]) and csv_metrics_ok,
            {'portfolio': [.04, -.2], 'SPY': [.1, -.2], 'EW': [.005, -.02], 'metrics_CSV': expected_csv_metrics},
            {'returns': actual_returns.reset_index().to_dict('records'), 'metrics_CSV': actual_metrics.to_dict('records')},
            source_refs=[ref_range(repo, S5, 33, 49, 'actual_backtest_loop_and_CSV_write'), ref_range(repo, S5, 69, 75, 'actual_metrics_CSV_write'), *refs],
            inputs={'weights': w.to_dict('records'), 'returns': R.reset_index().to_dict('records'), 'seed': None},
            oracle='Scalar dot product sum_j w_j*R_j,tp1; ten-asset arithmetic mean and exact date lookup; all five saved metric rows against independent implemented-convention oracle')
        add(8, 'Missing holding-month return date is surfaced rather than silently dropped',
            len(actual_returns) == len(w), {'expected_input_decisions': len(w), 'missing_date': dates[3]},
            {'output_rows': len(actual_returns), 'output_dates': actual_returns.index.tolist(), 'process_exit': execution['exit_code']},
            source_refs=[ref_range(repo, S5, 33, 37, 'missing_tp1_continue')], inputs={'all_weight_dates': dates[1:].tolist(), 'return_dates': R.index.tolist()},
            oracle='Each requested monthly decision needs a result or explicit rejected-row diagnostic; compare input count and output count', classification='silent_row_loss')
        missing_actual = float(actual_returns['ret_lo_2'].iloc[-1])
        add(9, 'An actively held missing return is not reported as zero portfolio return',
            math.isnan(missing_actual), {'portfolio_return': 'undefined; active SPY return missing'},
            {'portfolio_return': missing_actual, 'SPY_return': float(actual_returns['ret_spy'].iloc[-1]), 'exit_code': execution['exit_code']},
            source_refs=[ref_range(repo, S5, 36, 42, 'nansum_portfolio')],
            inputs={'weights': vectors[-1], 'returns': return_rows[-1]}, oracle='A positive weight times missing return makes the known portfolio sum incomplete', classification='missing_return_suppression')
        add(10, 'EW mean skips a missing ETF and changes denominator from ten to nine',
            close(actual_returns['ret_ew'].iloc[-1], .1/9), {'skipna_EW': .1/9, 'zero_fill_EW_reference': .1/10, 'strict_complete_basket': 'undefined'},
            float(actual_returns['ret_ew'].iloc[-1]), source_refs=[ref_range(repo, S5, 43, 44, 'benchmark_return_calculation')],
            inputs=return_rows[-1], oracle='Sum available .1 divided by nine available tickers; compare declared alternatives', classification='implementation_convention',
            notes=['Paper does not specify missing-asset policy. This is observed implicit benchmark reweighting, not a uniquely prescribed paper formula.'])

        # File-name and schema incompatibility are real complete-script failures.
        direct = run_original_script(repo, VIS, tmp)
        executions.append(direct)
        assert direct['exit_code'] == 0 or 'FileNotFoundError' in direct['stderr'], direct['stderr']
        add(11, 'Visualization accepts the filenames produced by Step5 directly', direct['exit_code'] == 0,
            'Visualization can consume unchanged Step5 output names', {'exit_code': direct['exit_code'], 'stderr': direct['stderr'], 'actual_names': [RET, MET]},
            source_refs=[ref_range(repo, VIS, 10, 15, 'visualization_input_files'), ref_range(repo, S5, 19, 20, 'backtest_output_files')],
            inputs='Only actual Step5 outputs in isolated directory', oracle='Execute unchanged visualization script; FileNotFoundError is an evidenced source contract failure', classification='file_contract')
        shutil.copy2(tmp / RET, tmp / 'backtest_returns_full_period.csv')
        shutil.copy2(tmp / MET, tmp / 'backtest_metrics_full_period.csv')
        renamed = run_original_script(repo, VIS, tmp)
        executions.append(renamed)
        assert renamed['exit_code'] == 0 or "KeyError: 'Strategy'" in renamed['stderr'], renamed['stderr']
        add(12, 'After filename-only fixture aliases, visualization accepts Step5 metric schema', renamed['exit_code'] == 0,
            {'visualization_requires': ['Strategy', 'Ann.Return', 'Ann.Vol'], 'Step5_columns': list(actual_metrics.columns)},
            {'exit_code': renamed['exit_code'], 'stderr': renamed['stderr']}, source_refs=[ref_range(repo, VIS, 108, 114, 'metric_schema_reads'), ref_range(repo, S5, 65, 75, 'metric_schema_write')],
            inputs='Exact Step5 CSV bytes copied under visualization-requested filenames only', oracle='Actual complete script; expected diagnostic exception KeyError Strategy', classification='schema_contract')
        adapted = actual_metrics.rename(columns={'Model': 'Strategy'})
        adapted['Strategy'] = adapted['Strategy'].replace({'ridge_lo_2': 'Long-only (l=2)', 'ridge_lo_3': 'Long-only (l=3)', 'ridge_lo_4': 'Long-only (l=4)', 'spy': 'SPY Buy&Hold', 'ew': 'Equal Weight'})
        adapted.to_csv(tmp / 'backtest_metrics_full_period.csv', index=False)
        ann_missing = run_original_script(repo, VIS, tmp)
        executions.append(ann_missing)
        adapted['Ann.Return'] = 0.0  # Diagnostic fixture stub solely to expose the next missing field.
        adapted.to_csv(tmp / 'backtest_metrics_full_period.csv', index=False)
        vol_missing = run_original_script(repo, VIS, tmp)
        executions.append(vol_missing)
        assert ann_missing['exit_code'] == 0 or "KeyError: 'Ann.Return'" in ann_missing['stderr'], ann_missing['stderr']
        assert vol_missing['exit_code'] == 0 or "KeyError: 'Ann.Vol'" in vol_missing['stderr'], vol_missing['stderr']
        add(13, 'Missing annual return and volatility columns remain after diagnostic Strategy mapping',
            ann_missing['exit_code'] == 0 and vol_missing['exit_code'] == 0, 'Required Ann.Return and Ann.Vol columns available from Step5',
            {'Ann.Return_case': ann_missing, 'Ann.Vol_case': vol_missing},
            source_refs=[ref_range(repo, VIS, 108, 114, 'annual_fields'), ref_range(repo, S5, 65, 75, 'metrics_fields')],
            inputs='Diagnostic fixture only: map Model column/name values to Strategy; then add Ann.Return=0 solely to reach Ann.Vol access. Original scripts unchanged.',
            oracle='Sequential actual KeyError diagnostics; fixture aliases are not a proposed correction or actual annual performance', classification='schema_contract')

    # Static proof is explicitly separate from executed schema failures.
    source5 = (repo / S5).read_text(encoding='utf-8-sig')
    comments = [{'line': i, 'text': line} for i, line in enumerate(source5.splitlines(), 1) if '10%' in line or 'log-return' in line]
    plot_nodes = [n for n in ast.walk(tree_of(repo, S5)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'plot', 'savefig', 'log', 'log1p'}]
    add(14, 'Step5 implements the commented 10 percent volatility target and cumulative log-return figures', bool(plot_nodes),
        'Executable return scaling and log-wealth plot calculation for paper Figures 10–13',
        {'comment_only': comments, 'plot_or_log_calls': [n.lineno for n in plot_nodes], 'visualization_cumulative_formula': '(1 + returns[col]).cumprod()'},
        source_refs=[ref_range(repo, S5, 77, len(source5.splitlines()), 'optional_comment_at_EOF'), ref_range(repo, VIS, 41, 44, 'unscaled_arithmetic_cumulative_plot')],
        inputs='Read-only source AST/text; paper methodology VOL_SCALING Figures 10–13', oracle='Static source inspection: no computation after optional comment; visualization directly compounds unscaled returns', classification='unimplemented_paper_output')

    # Existing Step2 + missing Step1 reproduces the actual original function bug.
    with tempfile.TemporaryDirectory(prefix='paper-audit-missing-step1-') as tmp:
        with working_directory(tmp):
            pd.DataFrame([{'sasdate_t': dates[0], 'sasdate_tp1': dates[1]}]).to_csv('section5_step2_per_regime_ridge_forecasts_DEMO.csv', index=False)
            scope = load_functions(repo, S3, extra={'STEP1_CANDIDATES': [Path('section5_step1_regime_probs.csv')],
                                                    'STEP2_CANDIDATES': [Path('section5_step2_per_regime_ridge_forecasts_DEMO.csv')]})
            df2, path2, macro_ctx = scope['load_step2_or_autogen']()
            try:
                scope['load_step1_or_autogen'](macro_ctx, list(df2['sasdate_t']))
                outcome = {'completed': True}
            except Exception as exc:
                outcome = {'completed': False, 'exception': type(exc).__name__, 'message': str(exc)}
            add(15, 'Existing Step2 with missing Step1 can auto-generate probabilities', outcome['completed'],
                'Load/generate required macro context before unpacking it', dict(outcome, macro_ctx=macro_ctx, step2_path=path2),
                source_refs=[source_ref(repo, S3, 'load_step2_or_autogen'), source_ref(repo, S3, 'load_step1_or_autogen')],
                inputs='Temp folder: only valid two-date Step2 CSV, no Step1 file, no downloader invoked',
                oracle='Execute both original loaders in actual branch order; expected diagnostic TypeError cannot unpack NoneType', classification='autogen_branch_error')

    # Execute original FRED preparation + actual missing-ETF save branch, stopping
    # before any automatic ten-window RidgeCV generation. Original AST is intact.
    contamination = []
    for filename in [S3, S4]:
        tree = tree_of(repo, filename)
        if filename == S3:
            body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_step2_or_autogen').body
        else:
            body = next(n for n in tree.body if isinstance(n, ast.If) and any(assigned(child, 'fred') for child in n.body)).body
        first = next(i for i, n in enumerate(body) if assigned(n, 'fred'))
        last = next(i for i, n in enumerate(body) if isinstance(n, ast.If) and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'synthesize_etf_bom_returns' for part in n.orelse for call in ast.walk(part)))
        nodes = body[first:last+1]
        with tempfile.TemporaryDirectory(prefix='paper-audit-synthetic-prefix-') as tmp:
            shutil.copy2(repo / 'FRED-MD_2024m12.csv', Path(tmp) / 'FRED-MD_2024m12.csv')
            scope = load_functions(repo, filename, names=['synthesize_etf_bom_returns'])
            with working_directory(tmp), warnings.catch_warnings():
                warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
                exec_nodes(repo, filename, nodes, scope)
                written = pd.read_csv(ETF, index_col=0, parse_dates=True)
                expected_synthetic = scope['synthesize_etf_bom_returns'](pd.DatetimeIndex(scope['macro_dates']))
                files = sorted(p.name for p in Path(tmp).iterdir())
                output_copy = dest / ('synthetic_prefix_' + filename.removesuffix('.py') + '.csv')
                shutil.copy2(Path(tmp) / ETF, output_copy)
                contamination.append({'source': filename, 'rows': len(written), 'columns': written.columns.tolist(), 'files_written': files,
                                      'matches_original_synthetic_generator': close(written.values, expected_synthetic.values),
                                      'source_marker_present': any('source' in col.lower() or 'synthetic' in col.lower() for col in written.columns),
                                      'path': str(output_copy), 'sha256': sha256(output_copy), 'seed': 123,
                                      'source_ref': ref_range(repo, filename, nodes[0].lineno, nodes[-1].end_lineno, 'original_FRED_prep_and_ETF_save_prefix_AST')})
    add(16, 'Missing ETF auto-generation leaves explicit synthetic provenance in downstream CSV artifacts',
        all(item['source_marker_present'] for item in contamination), 'Distinct synthetic identity or lineage metadata at the saved input boundary', contamination,
        source_refs=[item['source_ref'] for item in contamination], inputs={'FRED_sha256': sha256(repo / 'FRED-MD_2024m12.csv'), 'no_ETF_file': True, 'seed': 123},
        oracle='Execute original missing-file branches through CSV save; inspect filenames, schema and deterministic generated values', classification='synthetic_lineage_loss',
        notes=['Only prefix AST was executed. Automatic ten-window RidgeCV loops were not executed.', 'Step0 _make_synth_prices delegates to Yahoo and was never called.'])

    # One original first rolling window, with actual FRED and explicitly synthetic ETF.
    raw = pd.read_csv(repo / 'FRED-MD_2024m12.csv', skiprows=[1])
    raw.sasdate = pd.to_datetime(raw.sasdate)
    trow = pd.read_csv(repo / 'FRED-MD_2024m12.csv', nrows=1)
    codes = {col: int(trow[col].iloc[0]) for col in trow if col != 'sasdate'}
    prepared1 = prepare(repo, S1, raw, codes)
    prepared2 = prepare(repo, S2, raw, codes)
    panel = prepared1['panel'].iloc[:49].copy()
    panel2 = prepared2['panel'].iloc[:49].copy()
    rng = np.random.default_rng(20260916)
    returns = pd.DataFrame(rng.normal(.008, .02, size=(49, 10)), columns=TICKERS, index=pd.DatetimeIndex(panel.sasdate))
    returns.index.name = 'Date'
    with threadpool_limits(limits=1):
        first1 = step1_first_decision(repo, panel)
        first2 = step2_first_decision(repo, panel2, returns)
    with tempfile.TemporaryDirectory(prefix='paper-audit-connected-') as tmp:
        tmp = Path(tmp)
        pd.DataFrame([first1['record']]).to_csv(tmp / 'section5_step1_regime_probs.csv', index=False)
        pd.DataFrame([first2['forecasts']]).to_csv(tmp / 'section5_step2_per_regime_ridge_forecasts_DEMO.csv', index=False)
        returns.to_csv(tmp / ETF)
        panel.to_csv(tmp / 'audit_actual_FRED_first49.csv', index=False)
        chain = [run_original_script(repo, filename, tmp) for filename in [S3, S4, S5]]
        executions.extend(chain)
        if any(item['exit_code'] != 0 for item in chain):
            raise RuntimeError('Unexpected connected source execution failure: ' + json.dumps(chain))
        agg = pd.read_csv(tmp / 'section5_step3_ridge_aggregated_forecasts.csv')
        weights = pd.read_csv(tmp / WEIGHTS)
        backtest = pd.read_csv(tmp / RET, index_col=0, parse_dates=True)
        metrics = pd.read_csv(tmp / MET)
        expected_agg = [math.fsum(first1['record'][f'ptp1_R{i}']*first2['forecasts'][f'yhat_R{i}_{ticker}'] for i in range(1, 6)) for ticker in TICKERS]
        expected_weights = {l: independent_lo(expected_agg, l) for l in [2, 3, 4]}
        expected_return = {l: scalar_portfolio(expected_weights[l], returns.iloc[-1].tolist()) for l in [2, 3, 4]}
        date_pair = [pd.Timestamp(first1['record'][key]) for key in ['sasdate_t', 'sasdate_tp1']]
        date_ok = all(pd.Timestamp(df.iloc[0]['sasdate_t']) == date_pair[0] and pd.Timestamp(df.iloc[0]['sasdate_tp1']) == date_pair[1] for df in [agg, weights]) and backtest.index[0] == date_pair[1]
        handoff_ok = (len(agg) == len(weights) == len(backtest) == 1 and len(metrics) == 5 and date_ok and
                      close(agg[TICKERS].iloc[0], expected_agg) and
                      all(close(weights[[f'w_lo_{l}_{ticker}' for ticker in TICKERS]].iloc[0], expected_weights[l]) and close(backtest[f'ret_lo_{l}'].iloc[0], expected_return[l]) for l in [2, 3, 4]))
        files = preserve_files(tmp, dest / 'connected_fixture')
        details['connected'] = {'first_step1': first1, 'first_step2': first2, 'expected_aggregation': expected_agg,
                                'expected_weights': expected_weights, 'expected_return': expected_return,
                                'actual_backtest': backtest.reset_index().to_dict('records'), 'actual_metrics': metrics.to_dict('records'), 'files': files,
                                'macro_panels_equal': panel.equals(panel2), 'seed': 20260916, 'FRED_sha256': sha256(repo / 'FRED-MD_2024m12.csv')}
        add(17, 'One real-FRED window with declared synthetic ETF data traverses original Step1→2→3→4→5 files/dates/numbers', handoff_ok,
            {'dates': date_pair, 'rows': 1, 'metrics_rows': 5, 'aggregation': expected_agg, 'weights': expected_weights, 'return': expected_return},
            details['connected'], source_refs=[prepared1['ref'], prepared2['ref'], first1['ref'], first2['ref'],
                ref_range(repo, S3, 323, 343, 'actual_load_merge_aggregate_save'), ref_range(repo, S4, 272, 290, 'actual_weights_loop_save'), ref_range(repo, S5, 24, 75, 'actual_backtest_and_metrics')],
            inputs={'macro': 'Actual repository FRED; first 49 retained rows, first 48 for fitting', 'ETF': 'All ten ticker values explicitly synthetic', 'seed': 20260916, 'generator': 'numpy.default_rng(seed).normal(.008,.02,(49,10))'},
            oracle='Scalar Eq14 sum; independent top-positive allocation and sum_j w*R_tp1; date equality and row counts', classification='bounded_file_handoff',
            notes=['PASS proves only this handoff. REG-018/019/022 already show incompatible regime identities/partitions; no overall paper correctness claim.',
                   'One decision cannot estimate full-period Sharpe/Sortino. The files contain synthetic ETF returns, not observed investment performance.'])

    absent = [name for name in [ETF, 'etf_bom_returns.csv', 'wrds_etf_returns.csv'] if not (repo / name).exists()]
    add(18, 'Actual WRDS/ETF empirical performance replication', False, 'Original actual ETF/WRDS prices, vintage and execution calendar required',
        {'absent_candidate_files': absent, 'repository_ETF_CSVs': [p.name for p in repo.glob('*etf*.csv')], 'performed': False},
        source_refs=[ref_range(repo, S2, 13, 19, 'required_ETF_input')], inputs='Supplied repository inventory and evidence/data_timing/notes.md',
        oracle='Do not substitute a generated series or fresh Yahoo download for the original empirical input', status='SKIP', classification='missing_empirical_inputs')

    write_json(dest / 'fixtures.json', {'metrics': fixtures, 'loop_weights': w.to_dict('records'), 'loop_returns': R.reset_index().to_dict('records'), 'synthetic_prefix': contamination, 'connected': details['connected']})
    write_json(dest / 'script_executions.json', executions)
    counts = {status: Counter(row['status'] for row in results)[status] for status in ['PASS', 'FAIL', 'ERROR', 'SKIP']}
    notes = f'''# 성과지표·파이프라인 감사 (T6)

## 실행과 한계

결과: {counts}. 원 연구 코드·원 CSV·NPZ·PDF·README·requirements는 수정하지 않았다. 원 함수/AST와 전체 Step3/4/5 및 시각화 스크립트를 사용했고, 모든 쓰기는 직렬 임시 폴더에서 수행했다. 네트워크를 차단했고 Step0 다운로드 함수는 호출하지 않았다. 원 코드의 기대 예외는 FAIL이며 감사 도구 예외 ERROR와 구분한다.

## 성과지표

초기 wealth=1을 포함한 손계산 [-0.1,0.05]는 wealth [0.9,0.945], drawdown [-0.1,-0.055], AvgDD=-0.0775, MaxDD=-0.1이다. 원 Step5는 첫 수익 후 wealth부터 고점을 잡아 **AvgDD=0, MaxDD=0**을 반환했다(MET-004).

Sharpe는 rf=0, sqrt(12)×월평균/표본표준편차(ddof=1)이다. Sortino는 음수 월만의 표본표준편차를 분모로 쓴다. 손실 0/1회이면 Sortino가 NaN이고 상수 양수 수익은 Sharpe도 NaN이다. 선언한 비교 기준 downside RMS=sqrt(sum(min(r,0)^2)/전체 월수)와 수치가 다르다. **원문은 Sortino 분모를 충분히 정하지 않아 이 차이를 원문 수식 위반으로 판정하지 않는다.** AvgDD는 음수 drawdown 기간만 평균한다. DD와 `% Positive Ret.` 저장값은 비율이며 0.75는 75%다.

## 원 백테스트·파일 계약

- 실제 Step5에서 [0.6,0.4]×[0.1,-0.05]=0.04, 다음 달 -0.2를 확인했다.
- 보유월 날짜가 없으면 입력 결정 4개 중 3개만 출력한다. 경고 없이 continue한다(MET-008).
- 비중 1인 SPY의 수익이 NaN이어도 np.nansum으로 포트폴리오 수익 0을 만든다(MET-009).
- EW는 NaN 종목을 제외해 10종목 중 1종목 누락 시 0.1/9=0.011111…를 반환한다. 0 대체 기준 0.01과 다르며, 논문 미정 결측 정책으로 기록했다.
- 시각화는 Step5와 다른 파일명을 읽어 FileNotFoundError를 낸다. 임시 복사본 이름만 맞추면 `Model` 대신 `Strategy`를 요구해 KeyError를 낸다. 진단용 열·모델명만 매핑한 뒤에도 `Ann.Return`, 이어서 `Ann.Vol`의 누락 오류가 실제 발생했다(MET-011~013). 임시 Ann.Return=0은 다음 열 접근을 노출하기 위한 fixture이며 수익률 추정이 아니다.
- Step5의 10% vol-target/log-return 그림은 마지막 주석뿐이다. 시각화도 원 수익을 바로 cumprod한다. 실행되는 스케일링·로그 그림 코드는 없다(MET-014).

## 입력 조합·합성 계보

Step2만 존재하고 Step1이 없으면 원 Step2 loader가 macro_ctx=None을 반환한다. 원 Step1 loader가 이를 unpack하면서 TypeError를 낸다(MET-015). 별도로 Step3/4의 실제 FRED 전처리와 missing-ETF 생성·저장 prefix AST를 실행했다. seed=123 합성 389행이 공통 이름 `{ETF}`으로 저장되고 CSV에 합성 출처 표시나 sidecar가 없다(MET-016). 그 뒤 10창 RidgeCV 자동생성은 수행하지 않았다.

## 제한된 연결 실행

실제 FRED 첫 49 retained rows 중 48행(1992-03~1996-02)을 원 Step1/2 함수·첫 반복 AST에 넣고, seed=20260916의 명시적 합성 ETF 49×10을 사용했다. 생성된 선행 CSV를 임시 폴더에 모두 준 뒤 원 Step3→4→5 전체를 실행했다. 결정 1996-02, 보유 수익 라벨 1996-03의 1행, 지표 5행을 얻었다. 독립 식14·lo 비중·wR·날짜 연결이 일치했다(MET-017). 정확한 수치와 CSV는 connected_fixture 및 fixtures.json에 있다.

**연결 PASS는 파일·날짜·수치 전달만 뜻한다.** REG-018/019/022의 실제 Step1/2 레짐 분할·식별자 불일치 FAIL을 해소하지 않는다. 한 달의 합성 수익으로 논문 성과나 통계적 유의성을 재현했다고 주장하지 않는다. WRDS/실제 ETF 원 입력이 없으므로 실증 재현은 SKIP이다.

## 증거와 재실행

results.json은 원소스 SHA-256·심볼·줄범위, 입력·seed, expected/observed/oracle/tolerance를 기록한다. script_executions.json은 전체 스크립트의 실제 명령·stdout·stderr·종료 코드를 보존한다. execution.json은 진단 실행과 원본 18개 파일 전후 해시를 기록한다. 비유한 수는 JSON 문자열로 보존한다.

`python -X utf8 -m audit_tests.test_metrics_integration` 또는 `python -X utf8 audit_tests/run_audit.py --output reports/paper_audit/evidence`로 재실행한다. 정상 진단 완료는 FAIL이 있어도 종료 0이며 실행자 오류는 2다.
'''
    (dest / 'notes.md').write_text(notes, encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.repo / 'reports/paper_audit/evidence'
    dest = output / 'metrics_integration'
    dest.mkdir(parents=True, exist_ok=True)
    before = original_hashes(args.repo)
    stdout, stderr = io.StringIO(), io.StringIO()
    started, code, counts = datetime.now(timezone.utc).isoformat(), 0, {}
    install_network_guard()
    with redirect_stdout(stdout), redirect_stderr(stderr), threadpool_limits(limits=1):
        try:
            results = run(args.repo, output)
            counts = {status: Counter(row['status'] for row in results)[status] for status in ['PASS', 'FAIL', 'ERROR', 'SKIP']}
            print('Metrics/integration diagnostics completed:', counts)
        except Exception:
            traceback.print_exc()
            code = 2
    after = original_hashes(args.repo)
    if not before['matches_baseline'] or not after['matches_baseline']:
        code = 2
        stderr.write('Original source/data hash preservation failed.\n')
    write_json(dest / 'execution.json', {'command': getattr(sys, 'orig_argv', [sys.executable, *sys.argv]), 'cwd': str(Path.cwd()),
        'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': code,
        'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(), 'counts': counts, 'environment': environment(),
        'source_hashes_before': before, 'source_hashes_after': after, 'source_unchanged': before == after,
        'audit_source_sha256': {path.name: sha256(path) for path in [Path(__file__), Path(__file__).with_name('metrics_support.py')]},
        'evidence_sha256': {str(path.relative_to(dest)): sha256(path) for path in dest.rglob('*') if path.is_file() and path.name != 'execution.json'}})
    print(stdout.getvalue(), end='')
    print(stderr.getvalue(), end='', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
