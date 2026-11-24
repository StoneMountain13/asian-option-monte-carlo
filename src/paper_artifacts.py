"""从既有实验记录生成论文专用数值与彩色图，不运行定价实验。"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from .plotting import _main_records, COLORS
from .slide_figures import _cjk_family

B = chr(92)


def _sci(value):
    mantissa, exponent = f'{float(value):.2e}'.split('e')
    return '$' + mantissa + B + 'times10^{' + str(int(exponent)) + '}$'


def _table(spec, headers, rows):
    lines = [B+'begin{tabular}{'+spec+'}', B+'toprule',
             ' & '.join(headers)+' '+B*2, B+'midrule']
    lines += [' & '.join(row)+' '+B*2 for row in rows]
    return '\n'.join(lines+[B+'bottomrule', B+'end{tabular}'])


def write_paper_values(root):
    """用正式汇总表生成独立宏文件，避免覆盖展示稿的数值源。"""
    root = Path(root)
    raw = root/'results/raw'
    summary = pd.read_csv(raw/'main_summary.csv')
    sensitivity = pd.read_csv(raw/'sensitivity_summary.csv')
    cross = pd.read_csv(raw/'precision_comparison.csv').iloc[0]
    meta = json.loads((raw/'metadata.json').read_text(encoding='utf-8'))
    main = summary.set_index(['n', 'method'])
    a, b, c = [main.loc[(1000000, method)] for method in 'ABC']
    values = {
        'PaperMemoryA': f'{a.median_peak_rss_mib:.1f}',
        'PaperMemoryB': f'{b.median_peak_rss_mib:.1f}',
        'PaperMemoryC': f'{c.median_peak_rss_mib:.1f}',
        'PaperMemoryRatio': f'{a.median_peak_rss_mib/b.median_peak_rss_mib:.2f}',
        'PaperTimeB': f'{b.median_wall_seconds:.4f}',
        'PaperTimeC': f'{c.median_wall_seconds:.4f}',
        'PaperSERatio': f'{b.median_standard_error/c.median_standard_error:.1f}',
        'PaperSEB': _sci(b.median_standard_error),
        'PaperSEC': _sci(c.median_standard_error),
        'PaperSmallSEC': _sci(cross.median_standard_error_c),
        'PaperVarianceRatio': f'{c.variance_reduction_vs_b:,.1f}',
        'PaperCostRatio': _sci(c.time_se2_ratio_vs_b),
        'PaperCrossTime': f'{cross.matched_precision_time_ratio:.2f}',
        'PaperRuntime': f'{meta["full_execution"]["wall_seconds"]/60:.2f}',
    }
    controlled = sensitivity.loc[sensitivity.method.eq('C')]
    # 每个端点都从原始汇总行取值，正文只引用宏而不手工抄数。
    for factor, low, high, prefix in [
            ('strike',80,120,'Strike'), ('sigma',.1,.4,'Vol')]:
        for endpoint, value in [('Low',low),('High',high)]:
            row = controlled.loc[controlled.factor.eq(factor) & controlled.value.eq(value)].iloc[0]
            values['Paper'+prefix+endpoint+'VR'] = f'{row.variance_reduction_vs_b:,.1f}'
    step = controlled.loc[controlled.factor.eq('steps')].sort_values('value')
    values['PaperStepLowTime'] = f'{step.iloc[0].median_wall_seconds:.4f}'
    values['PaperStepHighTime'] = f'{step.iloc[-1].median_wall_seconds:.4f}'
    values['PaperStepVRMin'] = f'{step.variance_reduction_vs_b.min():,.1f}'
    values['PaperStepVRMax'] = f'{step.variance_reduction_vs_b.max():,.1f}'
    rows = []
    for n in sorted(summary.n.unique()):
        for method in 'ABC':
            row = main.loc[(n,method)]
            count = '$10^{'+str(int(round(math.log10(n))))+'}$' if method=='A' else ''
            rows.append([count,method,f'{row.estimate_mean:.6f}',
                         _sci(row.median_standard_error),_sci(row.across_seed_sd),
                         f'{row.median_wall_seconds:.4f}'])
    values['PaperDefaultTable'] = _table(
        'llrrrr',['路径数','方法','价格均值','标准误','种子间标准差','耗时（秒）'],rows)
    pilot = controlled.loc[controlled.factor.eq('pilot_n')].sort_values('value')
    values['PaperPilotTable'] = _table(
        'rrrrr',['预备路径数',r'系数 $\widehat\beta$','方差比',
                 '耗时（秒）',r'$\eta_C/\eta_B$'],
        [[f'{r.value:,.0f}',f'{r.median_beta:.4f}',f'{r.variance_reduction_vs_b:,.1f}',
          f'{r.median_wall_seconds:.4f}',_sci(r.time_se2_ratio_vs_b)]
         for r in pilot.itertuples()])
    text = ''.join(B+'newcommand{'+B+key+'}{'+value+'}\n' for key,value in values.items())
    (root/'docs/paper-values.tex').write_text(text,encoding='utf-8')
    return values


def _axis(ax):
    ax.grid(True, color='.87', linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)


def _path_axis(ax, counts):
    ax.set_xscale('log')
    ax.set_xticks(counts)
    ax.set_xticklabels([rf'$10^{{{int(np.log10(n))}}}$' for n in counts])
    ax.set_xlabel('正式路径数 $N$')


def _save(fig, target):
    fig.savefig(target, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_paper_figures(root):
    """使用颜色、线型和标记重绘四幅论文图，保留原展示稿图文件。"""
    root = Path(root)
    raw = root/'results/raw'
    target = root/'results/figures/paper'
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((raw/'metadata.json').read_text())['config']
    summary = pd.read_csv(raw/'main_summary.csv')
    sensitivity = pd.read_csv(raw/'sensitivity_summary.csv')
    records = _main_records(pd.read_csv(raw/'runs.csv'), config)
    independent = records.sort_values('repeat').drop_duplicates(['n','method','seed'])
    counts = sorted(config['experiment']['path_counts'])
    seed = config['experiment']['seeds'][0]
    style = {
        'font.family':_cjk_family(root), 'font.size':9,
        'mathtext.fontset':'cm', 'axes.labelsize':9, 'axes.titlesize':10,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
        'lines.linewidth':1.1, 'pdf.fonttype':3, 'savefig.facecolor':'white',
    }
    # 与展示稿统一配色，同时保留线型和标记供黑白打印辨认。
    colors = dict(COLORS)
    marks = {'A':'o','B':'s','C':'^'}
    lines = {'A':':','B':'--','C':'-'}
    labels = {'A':'A 全量模拟','B':'B 分块模拟','C':'C 控制变量'}
    with plt.rc_context(style):
        # A/B 的价格和 SE 一致，合并曲线可避免完全重叠的重复图例。
        fig, axes = plt.subplots(1,2,figsize=(6.35,2.42),constrained_layout=True)
        for method in ['B','C']:
            label = 'A/B 普通蒙特卡洛' if method=='B' else labels[method]
            one = independent.loc[independent.method.eq(method) & independent.seed.eq(seed)].sort_values('n')
            axes[0].errorbar(one.n,one.estimate,
                yerr=np.vstack([one.estimate-one.ci_low,one.ci_high-one.estimate]),
                color=colors[method],marker=marks[method],linestyle=lines[method],
                markersize=4,capsize=3,label=label)
            stats = independent.loc[independent.method.eq(method)].groupby('n').standard_error.agg(['median','min','max'])
            axes[1].errorbar(stats.index,stats['median'],
                yerr=np.vstack([stats['median']-stats['min'],stats['max']-stats['median']]),
                color=colors[method],marker=marks[method],linestyle=lines[method],
                markersize=4,capsize=3,label=label)
        axes[0].set_title('（a）价格估计')
        axes[0].set_ylabel('期权价格估计')
        axes[0].legend(frameon=False,loc='best')
        axes[1].set_title('（b）估计精度')
        axes[1].set_ylabel('估计标准误')
        axes[1].set_yscale('log')
        for ax in axes:
            _path_axis(ax,counts)
            _axis(ax)
        _save(fig,target/'price_and_se.pdf')

        fig, ax = plt.subplots(figsize=(5.25,2.65),constrained_layout=True)
        for method in 'ABC':
            group = summary.loc[summary.method.eq(method)].sort_values('n')
            ax.errorbar(group.n,group.median_peak_rss_mib,
                yerr=np.vstack([group.median_peak_rss_mib-group.min_peak_rss_mib,
                                group.max_peak_rss_mib-group.median_peak_rss_mib]),
                color=colors[method],marker=marks[method],linestyle=lines[method],
                markersize=4,capsize=3,label=labels[method],
                markerfacecolor='white' if method=='B' else colors[method])
        _path_axis(ax,counts)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('峰值常驻内存（MiB）')
        ax.legend(frameon=False,loc='upper left')
        _axis(ax)
        _save(fig,target/'memory_scaling.pdf')

        fig, ax = plt.subplots(figsize=(5.25,2.7),constrained_layout=True)
        for method in 'ABC':
            group = summary.loc[summary.method.eq(method)].sort_values('n')
            ax.loglog(group.median_wall_seconds,group.median_standard_error,
                color=colors[method],marker=marks[method],linestyle=lines[method],
                markersize=4,label=labels[method],
                markerfacecolor='white' if method=='B' else colors[method])
            if method=='C':
                for row in group.itertuples():
                    ax.annotate(rf'$N=10^{{{int(np.log10(row.n))}}}$',
                        (row.median_wall_seconds,row.median_standard_error),
                        xytext=(4,6),textcoords='offset points',fontsize=8)
        baseline = summary.loc[summary.method.eq('B') & summary.n.eq(1000000),'median_standard_error'].iloc[0]
        ax.axhline(baseline,color='.5',linestyle=':',linewidth=.8,label=r'B 的基准标准误')
        ax.set_xlabel('单次耗时（秒，含预备计算）')
        ax.set_ylabel('估计标准误')
        ax.margins(x=.12,y=.18)
        ax.legend(frameon=False,loc='upper right')
        _axis(ax)
        _save(fig,target/'precision_vs_cost.pdf')

        fig, axes = plt.subplots(1,3,figsize=(6.35,2.4),constrained_layout=True)
        for ax,factor,title,xlabel in zip(axes,['strike','sigma','steps'],
                ['（a）执行价','（b）波动率','（c）观察次数'],
                [r'执行价 $K$',r'波动率 $\sigma$',r'观察次数 $m$']):
            panel = sensitivity.loc[sensitivity.factor.eq(factor)]
            for method in ['B','C']:
                group = panel.loc[panel.method.eq(method)].sort_values('value')
                ax.plot(group.value,group.median_standard_error,
                    color=colors[method],marker=marks[method],linestyle=lines[method],
                    markersize=4,label='B 分块' if method=='B' else 'C 控制变量')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_yscale('log')
            if factor=='steps':
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xticks(sorted(panel.value.unique()))
            _axis(ax)
        axes[0].set_ylabel('估计标准误')
        axes[-1].legend(frameon=False,loc='best')
        _save(fig,target/'model_sensitivity.pdf')
    return sorted(target.glob('*.pdf'))
