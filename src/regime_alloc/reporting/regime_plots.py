"""Explicit, import-safe rendering of full-sample diagnostic artifacts."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import ErrorCode, ResearchError, atomic_write_bytes, sha256_file, write_json


def plot_regime_analysis(analysis, directory, *, language='ko', font_path=None, dpi=140):
    """Create PNG/SVG figures, Table 1 JSON, and a T09-consumable manifest.

    A new directory is required. SVG Date is omitted and element salts are
    fixed by analysis ID. Font/backend/rc changes occur only during this call.
    Korean requires an installed Korean font or an explicit font file; English
    is an explicit portable alternative. Importing this module performs no IO.
    """
    analysis.require_scope('full_sample')
    if language not in ('ko', 'en') or type(dpi) is not int or dpi < 72:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'language must be ko/en and dpi at least 72')
    directory = Path(directory)
    if directory.exists():
        raise ResearchError(ErrorCode.RUN_CONFLICT, f'plot output already exists: {directory}')
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib import colormaps

    if font_path is not None:
        font_path = Path(font_path)
        if not font_path.is_file():
            raise ResearchError(ErrorCode.INVALID_CONFIG, f'explicit font file absent: {font_path}')
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    elif language == 'ko':
        installed = {font.name for font in font_manager.fontManager.ttflist}
        font_name = next((name for name in ('Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans CJK KR')
                          if name in installed), '')
        if not font_name:
            raise ResearchError(ErrorCode.INVALID_CONFIG,
                                'Korean font unavailable; provide font_path or explicitly select language=en')
    else:
        font_name = 'DejaVu Sans'

    meta, a = analysis.metadata, analysis.arrays
    dates = pd.PeriodIndex(meta['months'], freq='M').to_timestamp()
    ids = meta['regime_ids']
    figures = []
    directory.mkdir(parents=True)
    def tr(ko, en):
        return ko if language == 'ko' else en

    indicator_names_ko = {'RPI': '실질 개인소득', 'UNRATE': '실업률', 'UMCSENTx': '소비심리',
                          'FEDFUNDS': '연방기금금리', 'CPIAUCSL': '소비자물가', 'S&P 500': '주가지수'}

    def new_figure(width=12, height=5):
        figure = Figure(figsize=(width, height), layout='constrained')
        FigureCanvasAgg(figure)
        return figure

    def save(figure, stem, paper_id, title, differences):
        outputs = {}
        for ext in ('png', 'svg'):
            buffer = io.BytesIO()
            metadata = {'Software': 'regime_alloc'} if ext == 'png' else {'Date': None, 'Creator': 'regime_alloc'}
            figure.savefig(buffer, format=ext, dpi=dpi, metadata=metadata)
            path = directory/f'{stem}.{ext}'
            atomic_write_bytes(path, buffer.getvalue())
            outputs[ext] = path.name
            outputs[f'{ext}_sha256'] = sha256_file(path)
        figures.append({'paper_id': paper_id, 'title': title, **outputs,
                        'scope': 'full_sample', 'reproduction_kind': 'method_reexecution',
                        'differences_from_paper': differences})
        figure.clear()

    def shade_nber(axis):
        recession = a['nber_usrec'].astype(bool)
        boundaries = np.flatnonzero(np.diff(np.r_[False, recession, False].astype(int)))
        for start, stop in zip(boundaries[::2], boundaries[1::2]):
            end = dates[stop] if stop < len(dates) else (dates[-1].to_period('M')+1).to_timestamp()
            axis.axvspan(dates[start], end, color='#718096', alpha=.18, linewidth=0)
        axis.set_xlim(dates[0], (dates[-1].to_period('M')+1).to_timestamp())
        axis.grid(axis='y', color='#d9dfe7', linewidth=.5)

    style = {'font.family': font_name, 'font.size': 10, 'axes.unicode_minus': False,
             'svg.hashsalt': analysis.analysis_id, 'svg.fonttype': 'path',
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.titlepad': 12, 'figure.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff'}
    with matplotlib.rc_context(style):
        f = new_figure(10, 5)
        ax = f.subplots()
        cumulative = np.cumsum(a['explained_variance_ratio'])
        k = meta['pca_components']
        ax.plot(np.arange(1, len(cumulative)+1), cumulative*100, color='#2463a0', linewidth=2)
        ax.axhline(meta['settings']['pca_variance']*100, color='#c46024', linestyle='--', label=tr('선택 기준', 'Selection threshold'))
        ax.scatter([k], [cumulative[k-1]*100], color='#c46024', zorder=4)
        ax.annotate(f'{k} PCs / {cumulative[k-1]*100:.2f}%', (k, cumulative[k-1]*100),
                    xytext=(8, -28), textcoords='offset points')
        ax.set(xlabel=tr('주성분 개수', 'Number of components'),
               ylabel=tr('누적 설명력 (%)', 'Cumulative explained variance (%)'), ylim=(0, 104),
               title=tr('PCA: 원자료의 변동을 얼마나 보존하는가', 'PCA: how much input variation is retained'))
        ax.legend(loc='lower right')
        save(f, 'figure01_pca', 'Figure 1', 'PCA cumulative variance',
             ['Component count is fitted from this snapshot; the paper value 61 is not imposed.',
              'Explained variance concerns macro input variation, not ETF forecast accuracy.'])

        f = new_figure(10, 4.5)
        ax = f.subplots()
        ax.plot([row['k_normal'] for row in meta['elbow']], [row['cosine_inertia'] for row in meta['elbow']],
                'o-', color='#2463a0')
        ax.axvline(5, color='#c46024', linestyle='--', label=tr('논문의 선택 r=5', 'Paper setting r=5'))
        ax.set(xticks=range(1, 11), xlabel=tr('일반 레짐 개수', 'Number of normal regimes'),
               ylabel=tr('코사인 거리 합', 'Sum of cosine distances'),
               title=tr('Elbow 진단: 같은 일반 표본에 1~10개 군집', 'Elbow: 1 to 10 clusters on the same normal sample'))
        ax.legend()
        save(f, 'supplement_elbow', 'Supplement elbow', 'Normal-regime elbow diagnostic',
             ['Diagnostic supplement for Algorithm 1; no automatic choice of r or forced visual elbow.'])

        f = new_figure(13, 6)
        axes = f.subplots(2, 1, sharex=True)
        for ax, key, title in zip(axes, ('labels', 'gmm_labels'), ('Layered k-means', 'GMM (centroid-aligned)')):
            shade_nber(ax)
            ax.step(dates, a[key], where='post', color='#256a93', linewidth=.9)
            ax.set(yticks=range(6), yticklabels=ids, ylim=(-.5, 5.5), ylabel=title)
        axes[0].set_title(tr('전체 표본 레짐 | 회색: 사후 NBER 경기침체', 'Full-sample regimes | gray: ex-post NBER recessions'))
        axes[-1].set_xlabel(tr('월 (고정 공개본, 사후 분석)', 'Month (fixed snapshot, ex-post analysis)'))
        save(f, 'figure02_regime_timeline', 'Figure 2', 'Layered/GMM/NBER timeline',
             ['GMM components are aligned by empirical PCA centroid distance without NBER labels.',
              'Equal regime numbers are descriptive alignment, not known economic ground truth.'])

        f = new_figure(13, 6)
        axes = f.subplots(2, 1, sharex=True)
        for ax, key, title in zip(axes, ('probabilities', 'gmm_probabilities'), ('Layered heuristic', 'GMM (aligned R0)')):
            shade_nber(ax)
            ax.plot(dates, a[key][:, 0], color='#a83a32', linewidth=1.0, label='R0')
            ax.plot(dates, 1-a[key][:, 0], color='#287695', linewidth=.9, alpha=.85, label='R1-R5')
            ax.set(ylabel=title, ylim=(-.02, 1.02))
            ax.legend(loc='upper right', ncol=2)
        axes[0].set_title(tr('이상치 레짐 R0 확률 | 경기침체 정답 확률을 뜻하지 않음',
                            'Outlier regime R0 probability | not a known recession probability'))
        axes[-1].set_xlabel(tr('월 | 회색: 사후 NBER 경기침체', 'Month | gray: ex-post NBER recessions'))
        save(f, 'figure03_r0_probability', 'Figure 3', 'Outlier/normal probability comparison',
             ['Uses the paper distance/log2 heuristic and a separately fitted full-covariance GMM.',
              'An aligned GMM R0 is an outlier-cluster comparison, not a recession-trained component.'])

        for unit in ('raw', 'tcode'):
            f = new_figure(10, max(4.5, len(meta['settings']['indicators'])*.6+1.8))
            ax = f.subplots()
            values = np.ma.masked_where(~a[f'indicator_{unit}_normalized_observed'], a[f'indicator_{unit}_normalized'])
            color = colormaps['YlGnBu'].copy()
            color.set_bad('#e2e5e9')
            im = ax.imshow(values, vmin=0, vmax=1, cmap=color, aspect='auto')
            labels = [(f"{indicator_names_ko[row['series']]} ({row['series']})"
                       if language == 'ko' and row['series'] in indicator_names_ko else row['series'])
                      +(' *' if row['constant'] else '') for row in meta['indicators'][unit]['records']]
            units_ko = '원자료' if unit == 'raw' else 't-code 변환값'
            ax.set(xticks=range(6), xticklabels=ids, yticks=range(len(labels)), yticklabels=labels,
                   xlabel=tr('레짐 | *: 상수 행, 회색: 관측값 없음', 'Regime | *: constant row, gray: no observations'),
                   title=tr(f'사후 경제지표: {units_ko} 평균을 지표별 0~1로 표시',
                            f'Post-hoc indicators: {unit} means scaled 0 to 1 within each row'))
            for i in range(values.shape[0]):
                for j in range(6):
                    valid = not np.ma.is_masked(values[i, j])
                    value = float(values[i, j]) if valid else 0
                    ax.text(j, i, f'{value:.2f}' if valid else 'NA', ha='center', va='center',
                            color='white' if valid and value > .58 else '#172635')
            f.colorbar(im, ax=ax, label=tr('행별 min-max 값', 'Row min-max value'), shrink=.8)
            save(f, f'figure04_indicators_{unit}', 'Figure 4' if unit == 'raw' else 'Supplement transformed indicators',
                 f'Indicator heatmap ({unit})', meta['indicators']['limitations'])

        f = new_figure(15, 5)
        axes = f.subplots(1, 3)
        for ax, key, title in zip(axes, ('transition_matrix', 'literal_transition_matrix', 'conditional_departures'),
                                  (tr('출발 전이 수 분모', 'Outgoing-count denominator'),
                                   tr('원문: 전체 출현 수 분모', 'Literal occurrence denominator'),
                                   tr('다른 레짐으로 이동한다면', 'Conditional on departure'))):
            values = a[key]
            im = ax.imshow(values, vmin=0, vmax=1, cmap='YlGnBu')
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center',
                            color='white' if values[i,j] > .58 else '#172635', fontsize=8)
            row_ids = [label+('*' if key == 'conditional_departures' and a['no_departure'][i] else '')
                       for i, label in enumerate(ids)]
            ax.set(xticks=range(6), xticklabels=ids, yticks=range(6), yticklabels=row_ids,
                   title=title, xlabel=tr('도착 레짐', 'Destination'), ylabel=tr('출발 레짐', 'Origin'))
        f.colorbar(im, ax=list(axes), label=tr('확률', 'Probability'), shrink=.75)
        f.suptitle(tr('전이행렬 | * 이탈 없음: 0행은 조건부 분포가 아님',
                      'Transitions | * no departure: zero row is not a conditional distribution'))
        save(f, 'figure05_transitions', 'Figure 5', 'Transition matrices and conditional departures',
             ['Rows are origin and columns destination, explicitly labelled.',
              'Outgoing denominator is primary; literal paper occurrence denominator is shown separately.',
              'Self-transition one yields a flagged zero departure row.'])

        f = new_figure(9, 7)
        ax = f.subplots()
        positions = np.column_stack([np.cos(np.pi/2-np.arange(6)*2*np.pi/6),
                                     np.sin(np.pi/2-np.arange(6)*2*np.pi/6)])
        for i in range(6):
            for j in range(6):
                weight = a['conditional_departures'][i, j]
                if i != j and weight > 0:
                    arrow = FancyArrowPatch(positions[i], positions[j], arrowstyle='-|>',
                                            connectionstyle='arc3,rad=.12', shrinkA=23, shrinkB=24,
                                            mutation_scale=9+5*weight, linewidth=.5+3*weight,
                                            color=colormaps['Blues'](.25+.75*weight), alpha=.35+.65*weight)
                    ax.add_patch(arrow)
        for i, position in enumerate(positions):
            ax.add_patch(Circle(position, .12, facecolor='#f4f8fc', edgecolor='#345b78', zorder=4))
            ax.text(*position, ids[i], ha='center', va='center', zorder=5, fontweight='bold')
            ax.text(position[0], position[1]-.21, f"n={a['occurrences'][i]}", ha='center', fontsize=9,
                    zorder=6, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
        ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect='equal',
               title=tr('레짐 이탈 네트워크 | 화살표: 출발 → 도착', 'Regime departure network | arrows: origin to destination'))
        ax.axis('off')
        f.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='Blues'), ax=ax, shrink=.7,
                   label=tr('이탈을 조건으로 한 전이 확률', 'Transition probability conditional on departure'))
        save(f, 'figure06_departure_network', 'Figure 6', 'Conditional departure network',
             ['Fixed circular layout, every positive off-diagonal edge, and actual fitted regime IDs.',
              'No paper economic label or emergent path layout is imposed.'])

    write_json(directory/'table01_regime_descriptions.json',
               {'paper_id': 'Table 1', 'analysis_id': analysis.analysis_id, 'records': meta['table1'],
                'evidence': meta['indicators'], 'interpretation': 'sample-dependent descriptive summaries, not economic truths'})
    manifest = {'schema_version': '1.0', 'analysis_id': analysis.analysis_id, 'model_id': analysis.model_id,
                'scope': 'full_sample', 'language': language, 'font_name': font_name, 'dpi': dpi,
                'render_policy': {'backend': 'FigureCanvasAgg', 'svg_date': 'omitted',
                                  'svg_hashsalt': analysis.analysis_id, 'svg_fonts': 'paths',
                                  'timestamp_metadata': 'excluded', 'matplotlib_version': matplotlib.__version__},
                'figures': figures, 'tables': [{'paper_id': 'Table 1', 'json': 'table01_regime_descriptions.json',
                                               'sha256': sha256_file(directory/'table01_regime_descriptions.json')}],
                'nber_use': meta['nber_purpose']}
    write_json(directory/'figures_manifest.json', manifest)
    return manifest
