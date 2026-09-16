"""Declared offline experiments, using the common state/forecast/accounting engine.

All child configs and the complete inventory are published before execution.
Any failed or missing child fails the suite; no seed is silently replaced.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ..config import ResearchConfig, validate_config
from ..contracts import SCHEMA_VERSION, ErrorCode, ResearchError, canonical_id
from .artifacts import RunWriter
from .engine import ExecutionContext, run_research, selected_months, _code_snapshot
from .metrics import compute_metrics
from .statistics import METRICS, PAIR_NAMES, compare_profile

REPRESENTATIVES = ('ridge_lo_2', 'naive_lo_2', 'bl_lo_2', 'mvo_lo_2', 'spy', 'ew')


@dataclass(frozen=True)
class Variant:
    name: str
    config: ResearchConfig
    changes: dict
    curve: str = 'unscaled'
    reason: str = 'one_at_a_time_declared_sensitivity'


def sensitivity_variants(config):
    recipes = {
        'lag_1': {'research.lag_months': 1}, 'lag_3': {'research.lag_months': 3},
        'missing_rate_005': {'research.missing_rate': .05}, 'ffill_limit_0': {'research.ffill_limit': 0},
        'pca_090': {'research.pca_variance': .90}, 'pca_099': {'research.pca_variance': .99},
        'fallback_regime_mean': {'research.fallback': 'regime_mean'},
        'validation_forward': {'research.validation': 'blocked'},
        'cov_shrink_0': {'research.cov_shrink': 0.}, 'cov_shrink_025': {'research.cov_shrink': .25},
        'omega_scale_05': {'research.omega_scale': .5}, 'omega_scale_2': {'research.omega_scale': 2.},
        'cost_10bp': {'portfolio.transaction_cost_bps': 10., 'portfolio.borrow_rate': .02, 'portfolio.financing_rate': .02},
        'cost_25bp': {'portfolio.transaction_cost_bps': 25., 'portfolio.borrow_rate': .02, 'portfolio.financing_rate': .02},
        'gross_cap_1': {'portfolio.gross_cap': 1.}, 'vol_lookback_12': {'portfolio.vol_lookback': 12}}
    variants = []
    for name in config.suite.sensitivity_variants:
        if name not in recipes:
            raise ResearchError(ErrorCode.INVALID_CONFIG, f'unknown sensitivity variant {name}')
        changes = recipes[name]
        sections = {section: replace(getattr(config, section), **{key.split('.')[1]: value for key, value in changes.items()
                    if key.startswith(section+'.')}) for section in ('research', 'portfolio')}
        child = validate_config(replace(config, **sections))
        reason = ('lag changes both public-vintage availability and macro base cutoff b+1' if name.startswith('lag_') else
                  'monthly prequential expanding inner train; shared ETF lambda replaces regime/ETF LOO' if name == 'validation_forward' else
                  'transaction plus annual borrow and financing cost scenario' if name.startswith('cost_') else
                  'compare scaled curves; baseline tables remain unscaled' if name in ('gross_cap_1', 'vol_lookback_12') else
                  'one_at_a_time_declared_sensitivity')
        variants.append(Variant(name, child, changes, 'scaled' if name in ('gross_cap_1', 'vol_lookback_12') else 'unscaled', reason))
    return tuple(variants)


def permute_memberships(labels, seed):
    labels = np.asarray(labels)
    if labels.shape != (48,) or labels.dtype.kind not in 'iu' or not set(labels) <= set(range(6)):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'random control requires all 48 integer regime memberships')
    if type(seed) is not int or seed < 0:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'membership seed must be a nonnegative integer')
    # Exact RNG permutation, with no rejection/redraw/cherry-picking.
    return np.random.default_rng(seed).permutation(labels)


def random_state(base, seed):
    labels = base.labels.to_numpy()
    changed = permute_memberships(labels, seed)
    return base.with_labels(changed, provenance={'method': 'permute_all48_memberships_preserve_counts',
        'seed': seed, 'rng': 'numpy.default_rng.PCG64', 'monthly_seed_policy': 'reset_seed_each_decision',
        'original_counts': np.bincount(labels, minlength=6).tolist(),
        'recompute': ['centers', 'current_probabilities', 'transitions', 'conditional_predictions']})


def _panel(path, strategies, months, *, curve='unscaled'):
    frame = pd.read_csv(Path(path)/('scaled' if curve == 'scaled' else '')/'returns.csv')
    if frame.duplicated(['holding_month', 'strategy_id']).any():
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'duplicate child return keys')
    panel = frame.pivot(index='holding_month', columns='strategy_id', values='net_return')
    if not set(strategies) <= set(panel.columns) or list(panel.index) != list(months):
        raise ResearchError(ErrorCode.MISSING_DATA, 'child return panel does not match planned months/strategies')
    result = panel.loc[list(months), list(strategies)]
    if not np.isfinite(result.to_numpy()).all():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, 'child return panel contains nonfinite values')
    return result


def _smoke_comparisons(profile, seeds):
    reason = 'smoke_independent_months_not_a_continuous_performance_series'
    return {'schema_version': SCHEMA_VERSION, 'profile': profile, 'comparisons': [
        {'schema_version': SCHEMA_VERSION, 'profile': profile, 'pair_id': pair, 'metric': metric,
         'treatment': names[0], 'control': names[1], 'direction': 'higher_is_better_including_less_negative_maxdd',
         'n_pairs': 0, 'excluded_pairs': [], 'paired_t': {'p': None, 'reason': reason},
         'nemenyi': {'p': None, 'reason': reason}, 'holm_p': {'value': None, 'reason': reason},
         'bootstrap_difference_ci': {'value': None, 'reason': reason}, 'assumptions': [reason],
         'seed_manifest': {'control_seeds': seeds}} for pair, names in PAIR_NAMES.items() for metric in METRICS]}


def run_suite(config: ResearchConfig, output, *, smoke=False, context=None):
    """Execute both profiles, main50, every declared seed, and configured variants.

    Smoke uses actual first / 2020-04 / last months and exactly two seeds. It
    exercises all configured sensitivity hooks but does not claim performance.
    Source snapshots remain fixed across the entire suite including analysis.
    """
    validate_config(config)
    if config.suite.representative_strategy != 'lo' or config.suite.representative_size != 2:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'approved comparisons require representative lo size2')
    if config.research.models != ('naive', 'ridge', 'bl', 'mvo') or config.research.strategies != ('lo', 'lns', 'los', 'mx') or config.research.selection_sizes != (2, 3, 4):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'suite main baseline requires all50 strategies')
    seeds = list(range(2 if smoke else config.suite.control_repetitions))
    revision, hashes = _code_snapshot()
    writer = RunWriter(output, {'schema_version': SCHEMA_VERSION, 'run_id': uuid.uuid4().hex,
        'kind': 'reproduction_suite', 'smoke': smoke, 'code_revision': revision, 'code_hashes': hashes,
        'config_hash': canonical_id(config), 'profiles': ['fixed_snapshot', 'vintage_lagged']})
    completed, active_id = [], ''
    try:
        writer.json('effective_config.json', {'config': asdict(config), 'smoke': smoke, 'effective_control_seeds': seeds,
            'prior_run_cache': False, 'random_membership_policy': 'all48_permutation_counts_preserved_seed_reset_each_month'})
        # Publish requested inventory even if data preflight subsequently fails.
        planned, configs = [], {}
        for profile in ('fixed_snapshot', 'vintage_lagged'):
            base = replace(config, research=replace(config.research, profile=profile))
            children = [('main', base, {}, 'unscaled', 'main50_separate_from_comparisons', None)]
            controls = replace(base, research=replace(base.research, models=('naive', 'ridge')))
            children += [(f'control_{seed:03d}', controls, {'membership_seed': seed}, 'unscaled', 'random_membership_control',
                          ('naive_lo_2', 'ridge_lo_2')) for seed in seeds]
            children += [(v.name, v.config, v.changes, v.curve, v.reason, REPRESENTATIVES) for v in sensitivity_variants(base)]
            for name, child, changes, curve, reason, strategies in children:
                run_id = f'{profile}/{name}'; configs[run_id] = child
                planned.append({'id': run_id, 'profile': profile, 'name': name, 'path': f'children/{run_id}',
                    'config': asdict(child), 'changes': changes, 'curve': curve, 'reason': reason,
                    'strategy_selection': list(strategies) if strategies is not None else 'all50',
                    'requested_period': [child.research.start_month, child.research.end_month]})
        writer.json('planned_runs.json', {'schema_version': SCHEMA_VERSION, 'runs': planned,
            'planned_count': len(planned), 'control_seeds': seeds, 'execution_policy': 'fail_entire_suite_on_any_child_failure'})
        ctx = context if context is not None else ExecutionContext(config)
        ctx.require_config(config)
        for item in planned:
            child = configs[item['id']]; start = ctx.feasible_start(child)
            actual = selected_months(replace(child, research=replace(child.research, start_month=start)), smoke)
            item['actual_months'] = actual
            item['warmup'] = {'requested_start': child.research.start_month, 'actual_start': start,
                'changed': start != child.research.start_month, 'reason': 'listing/public-vintage warmup preserving48targets' if start != child.research.start_month else 'no_warmup_change'}
        for profile in ('fixed_snapshot', 'vintage_lagged'):
            group = [x for x in planned if x['profile'] == profile]
            common = sorted(set.intersection(*(set(x['actual_months']) for x in group)))
            if not common:
                raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'no common period across declared variants')
            for item in group:
                item['suite_common_months'] = common
                item['common_period_reason'] = 'intersection_of_all_declared_variant_actual_months_no_posthoc_selection'
        writer.json('execution_plan.json', {'runs': planned, 'code_revision': revision, 'code_hashes': hashes})
        bases = {}
        with threadpool_limits(1):
            for item in planned:
                active_id = item['id']; child = configs[active_id]
                ctx.require_config(child)
                if _code_snapshot() != (revision, hashes):
                    raise ResearchError(ErrorCode.HASH_MISMATCH, 'source changed during suite')
                kwargs = {'smoke': smoke, 'context': ctx, 'experiment': {'suite_child': active_id,
                    'changes': item['changes'], 'reason': item['reason'], 'comparison_curve': item['curve']}, 'use_prior_cache': False}
                if item['strategy_selection'] != 'all50': kwargs['strategy_selection'] = item['strategy_selection']
                if item['name'].startswith('control_'):
                    profile = item['profile']
                    if profile not in bases:
                        bases[profile] = {month: ctx.state(child, month) for month in item['actual_months']}
                    kwargs['state_overrides'] = {month: random_state(bases[profile][month], item['changes']['membership_seed']) for month in item['actual_months']}
                if item['name'] == 'validation_forward':
                    from ..models.forward_validation import select_forward_lambdas
                    kwargs['fixed_lambdas'] = {month: select_forward_lambdas(child, ctx, month) for month in item['actual_months']}
                result = run_research(child, writer.path/item['path'], **kwargs)
                if result.get('status') != 'succeeded':
                    raise ResearchError(ErrorCode.VERIFICATION_FAILED, f'child did not succeed: {active_id}')
                completed.append({'id': active_id, 'status': 'succeeded', 'run_id': result['run_id'], 'path': item['path']})
                writer.json(f'child_status/{len(completed):04d}.json', completed[-1])
        comparisons, sensitivity = [], []
        for profile in ('fixed_snapshot', 'vintage_lagged'):
            group = [x for x in planned if x['profile'] == profile]
            main = next(x for x in group if x['name'] == 'main')
            baseline = _panel(writer.path/main['path'], REPRESENTATIVES, main['actual_months'])
            control_panels = [_panel(writer.path/next(x for x in group if x['name'] == f'control_{seed:03d}')['path'],
                              ('naive_lo_2', 'ridge_lo_2'), main['actual_months']) for seed in seeds]
            if smoke:
                stats = _smoke_comparisons(profile, seeds)
                stats['original_series'] = {'months': main['actual_months'], 'baseline': baseline.to_dict(orient='list'),
                    'controls': [{'seed': seed, 'returns': panel.to_dict(orient='list')} for seed, panel in zip(seeds, control_panels)]}
            else:
                pairs = {'naive_vs_random': (baseline.naive_lo_2.to_numpy(), np.array([p.naive_lo_2 for p in control_panels])),
                         'ridge_vs_random': (baseline.ridge_lo_2.to_numpy(), np.array([p.ridge_lo_2 for p in control_panels])),
                         'bl_vs_mvo': (baseline.bl_lo_2.to_numpy(), baseline.mvo_lo_2.to_numpy()[None, :])}
                stats = compare_profile(profile, main['actual_months'], pairs, seeds=seeds,
                    repetitions=config.suite.bootstrap_repetitions,
                    block_lengths=(config.suite.bootstrap_block_months, *config.suite.bootstrap_sensitivity_blocks),
                    seed=config.suite.bootstrap_seed, auxiliary_seed=config.suite.auxiliary_seed)
            writer.json(f'statistics/{profile}.json', stats); comparisons.extend(stats['comparisons'])
            for item in group:
                if item['name'] == 'main' or item['name'].startswith('control_'): continue
                common = sorted(set(main['actual_months']) & set(item['actual_months']))
                left = _panel(writer.path/main['path'], REPRESENTATIVES, main['actual_months'], curve=item['curve']).loc[common]
                right = _panel(writer.path/item['path'], REPRESENTATIVES, item['actual_months'], curve=item['curve']).loc[common]
                for strategy in REPRESENTATIVES:
                    if smoke:
                        metrics = {metric: {'value': None, 'reason': 'smoke_independent_months_not_a_continuous_performance_series'} for metric in METRICS}
                    else:
                        a, b = compute_metrics(common, left[strategy]), compute_metrics(common, right[strategy])
                        metrics = {metric: {'value': b.values[metric]-a.values[metric] if b.values[metric] is not None and a.values[metric] is not None else None,
                            'reason': b.reasons.get(metric, a.reasons.get(metric, ''))} for metric in METRICS}
                    sensitivity.append({'profile': profile, 'variant': item['name'], 'strategy': strategy, 'curve': item['curve'],
                        'changed_fields': item['changes'], 'warmup': item['warmup'], 'common_months': common,
                        'common_period_reason': 'intersection_baseline_variant_actual_months', 'difference_variant_minus_baseline': metrics})
        writer.json('comparisons.json', {'schema_version': SCHEMA_VERSION, 'comparisons': comparisons})
        writer.json('sensitivity.json', {'schema_version': SCHEMA_VERSION, 'comparisons': sensitivity})
        if len(completed) != len(planned):
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'planned child inventory incomplete')
        ctx.require_config(config)
        if _code_snapshot() != (revision, hashes): raise ResearchError(ErrorCode.HASH_MISMATCH, 'source changed during suite analysis')
        writer.json('suite_results.json', {'planned_count': len(planned), 'completed': completed, 'context_counts': ctx.counts,
            'checks': [{'name': 'all_planned_children_succeeded', 'status': 'pass'}, {'name': 'source_fixed_entire_suite', 'status': 'pass'}]})
        return writer.succeed()
    except BaseException as exc:
        writer.json('suite_failure.json', {'active_child': active_id, 'completed': completed,
            'reason': str(exc), 'details': exc.to_dict() if isinstance(exc, ResearchError) else {'exception_type': type(exc).__name__},
            'policy': 'failed_children_and_partial_outputs_retained; no_seed_replacements'})
        writer.fail(exc)
        raise
