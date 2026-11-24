"""用已有实验汇总生成中文展示图。"""
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTFont
import pandas as pd

from .plotting import COLORS

LABELS = {'A': 'A 全量模拟', 'B': 'B 分块模拟', 'C': 'C 分块＋控制变量'}
STYLES = {'A': ':o', 'B': '--s', 'C': '-^'}


def _series(ax, data, method, x, y, x_multiplier=1):
    rows = data.loc[data['method'].eq(method)].sort_values(x)
    ax.plot(rows[x] * x_multiplier, rows[y], STYLES[method],
            color=COLORS[method], label=LABELS[method], linewidth=2, markersize=6)


def _save(fig, target):
    fig.savefig(target, bbox_inches='tight', pad_inches=0.08, facecolor='white')
    plt.close(fig)


def _cjk_family(root):
    font_path = os.environ.get('SLIDES_CJK_FONT')
    if not font_path:
        family = 'Noto Sans CJK SC'
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            pass

        # Matplotlib 只扫描字体集合的第一张字体。由 Fontconfig 选出简中字体，
        # 再将对应字体单独提取到构建目录，供图表嵌入字形。
        match = subprocess.check_output(
            ['fc-match', '-f', '%{file}\\n%{index}\\n%{family}\\n', family],
            text=True,
        )
        filename, index, matched_family = match.splitlines()
        if family not in matched_family.split(','):
            raise RuntimeError('请安装 Noto Sans CJK SC，或将 SLIDES_CJK_FONT 指向中文字体文件。')
        font_path = filename
        if Path(filename).suffix.lower() in {'.ttc', '.otc'}:
            font_path = root / 'docs' / '.build' / 'slides-cjk.otf'
            font_path.parent.mkdir(parents=True, exist_ok=True)
            with TTFont(filename, fontNumber=int(index)) as font:
                font.save(font_path)
    font_manager.fontManager.addfont(font_path)
    return font_manager.FontProperties(fname=font_path).get_name()


def create_slide_figures(root):
    raw = Path(root) / 'results' / 'raw'
    target = Path(root) / 'results' / 'figures' / 'slides_zh'
    target.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(raw / 'main_summary.csv')
    sensitivity = pd.read_csv(raw / 'sensitivity_summary.csv')
    family = _cjk_family(Path(root))
    style = {'font.family': family, 'font.size': 11, 'axes.labelsize': 11,
             'axes.titlesize': 12, 'axes.spines.top': False,
             'axes.spines.right': False, 'axes.grid': True, 'axes.axisbelow': True,
             'grid.alpha': .24, 'legend.frameon': False, 'legend.fontsize': 10,
             'axes.unicode_minus': False, 'pdf.fonttype': 3}
    with plt.rc_context(style):
        fig, (memory, error) = plt.subplots(1, 2, figsize=(9.3, 3.0), layout='constrained')
        for method in 'AB':
            _series(memory, main, method, 'n', 'median_peak_rss_mib')
        for method in 'BC':
            _series(error, main, method, 'n', 'median_standard_error')
        memory.set(ylabel='峰值内存中位数（MiB）', ylim=(0, 520))
        error.set(ylabel='标准误中位数', yscale='log')
        for ax in (memory, error):
            ax.set(xscale='log', xlabel='正式模拟路径数')
            ax.set_xticks([10_000, 100_000, 1_000_000], ['1 万', '10 万', '100 万'])
            ax.legend(loc='upper left' if ax is memory else 'lower left')
        _save(fig, target / 'same_paths.pdf')

        fig, ax = plt.subplots(figsize=(7.0, 3.55), layout='constrained')
        for method in 'BC':
            _series(ax, main, method, 'median_wall_seconds', 'median_standard_error', 1000)
        ax.set(xscale='log', yscale='log', xlabel='单次耗时中位数（毫秒）', ylabel='标准误中位数')
        cross = pd.read_csv(raw / 'precision_comparison.csv').iloc[0]
        selected = (
            ('B', int(cross['baseline_n']), 'B 的 100 万条', (-7, 11), 'right'),
            ('C', int(cross['n']), 'C 的 1 万条', (8, 9), 'left'),
        )
        for method, count, label, offset, align in selected:
            row = main.loc[main['method'].eq(method) & main['n'].eq(count)].iloc[0]
            point = (row['median_wall_seconds'] * 1000, row['median_standard_error'])
            ax.scatter(*point, s=130, edgecolor=COLORS[method], facecolors='none', zorder=4)
            ax.annotate(label, point, xytext=offset, textcoords='offset points',
                        ha=align, color=COLORS[method], fontsize=11)
        ax.legend(loc='upper right')
        _save(fig, target / 'time_precision.pdf')

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9.3, 2.65), layout='constrained')
        for ax, factor, label in zip(axes, ('strike', 'sigma', 'steps'),
                                     ('执行价', '年化波动率', '观察次数')):
            panel = sensitivity.loc[sensitivity['factor'].eq(factor)]
            for method in 'BC':
                _series(ax, panel, method, 'value', 'median_standard_error')
            ticks = sorted(panel['value'].unique())
            if factor == 'steps':
                ax.set_xscale('log')
            tick_labels = [f'{v:.0%}' if factor == 'sigma' else f'{v:g}' for v in ticks]
            ax.set(yscale='log', xlabel=label)
            ax.set_xticks(ticks, tick_labels)
        axes[0].set_ylabel('标准误中位数')
        axes[-1].legend(loc='center right')
        _save(fig, target / 'sensitivity.pdf')
