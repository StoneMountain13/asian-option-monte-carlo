"""读取正式实验记录生成英文图表，不额外运行任何定价样本。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from .experiments import _tasks, _key

COLORS = {'A':'#88939F','B':'#286E9A','C':'#AE0B2A'}
MARKERS = {'A':'o','B':'s','C':'^'}
LABELS = {'A':'A: Full array','B':'B: Blocked MC','C':'C: Control variate'}


def _style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,
        'axes.titlesize':18,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,
        'legend.fontsize':13,'lines.linewidth':2.0,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})


def _finish(fig, destination):
    fig.savefig(destination,dpi=280,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def _grid(ax):
    ax.grid(True,which='major',color='#DDDDDD',linewidth=.7)
    ax.set_axisbelow(True)


def _path_ticks(ax, counts):
    ax.set_xscale('log')
    ax.set_xticks(counts)
    ax.set_xticklabels([rf'$10^{{{int(np.log10(n))}}}$' for n in counts])
    ax.set_xlabel('Main paths, N')


def _main_records(runs, config):
    wanted = {_key(task) for task in _tasks(config,'main')}
    # 参数匹配去重，避免将断点续跑旧记录或时间重复当成新样本。
    selected = {}
    for row in runs.loc[runs.status.eq('completed')].to_dict('records'):
        key = _key(row)
        if key in wanted:
            selected[key] = row
    return pd.DataFrame(selected.values())


def create_figures(root: Path):
    root = Path(root)
    raw, destination = root/'results/raw', root/'results/figures'
    destination.mkdir(parents=True,exist_ok=True)
    metadata = json.loads((raw/'metadata.json').read_text(encoding='utf-8'))
    config = metadata['config']
    summary = pd.read_csv(raw/'main_summary.csv')
    sensitivity = pd.read_csv(raw/'sensitivity_summary.csv')
    main = _main_records(pd.read_csv(raw/'runs.csv'),config)
    independent = main.sort_values('repeat').drop_duplicates(['n','method','seed'])
    representative_seed = int(config['experiment']['seeds'][0])
    counts = sorted(int(n) for n in config['experiment']['path_counts'])
    _style()
    output = []
    # 左图只展示一个种子的原始CI，右图展示独立种子的SE范围。
    fig,axes = plt.subplots(1,2,figsize=(12.4,4.8),constrained_layout=True)
    for method in ['A','B','C']:
        one = independent.loc[(independent.method==method) & (independent.seed==representative_seed)].sort_values('n')
        axes[0].errorbar(one.n,one.estimate,yerr=np.vstack([one.estimate-one.ci_low,one.ci_high-one.estimate]),
            color=COLORS[method],marker=MARKERS[method],markersize=7,capsize=4,label=LABELS[method],
            markerfacecolor='white' if method=='B' else COLORS[method])
        stats = independent.loc[independent.method.eq(method)].groupby('n').standard_error.agg(['median','min','max'])
        axes[1].errorbar(stats.index,stats['median'],yerr=np.vstack([stats['median']-stats['min'],stats['max']-stats['median']]),
            color=COLORS[method],marker=MARKERS[method],markersize=7,capsize=4,label=LABELS[method])
    axes[0].set_title(f'Price and 95% CI (seed {representative_seed})')
    axes[0].set_ylabel('Discounted option price')
    axes[1].set_title('Standard error across independent seeds')
    axes[1].set_ylabel('Standard error (median and range)')
    axes[1].set_yscale('log')
    for ax in axes:
        _path_ticks(ax,counts)
        _grid(ax)
    axes[0].legend(loc='best',frameon=False)
    name='price_and_se.png'
    _finish(fig,destination/name)
    output.append(name)

    fig,ax = plt.subplots(figsize=(8.8,5.2),constrained_layout=True)
    for method in ['A','B','C']:
        group=summary.loc[summary.method.eq(method)].sort_values('n')
        ax.loglog(group.median_wall_seconds,group.median_standard_error,
                  marker=MARKERS[method],color=COLORS[method],markersize=8,label=LABELS[method])
        if method=='C':
            for row in group.itertuples():
                ax.annotate(rf'$N=10^{{{int(np.log10(row.n))}}}$',
                    (row.median_wall_seconds,row.median_standard_error),xytext=(7,7),textcoords='offset points',fontsize=12)
    target=summary.loc[summary.method.eq('B') & summary.n.eq(1000000),'median_standard_error']
    if len(target):
        ax.axhline(float(target.iloc[0]),color='#666666',linestyle=':',linewidth=1.5,label=r'B, $N=10^6$ SE')
    ax.set_title('Precision versus cost')
    ax.set_xlabel('Time per estimate (s), including pilot')
    ax.set_ylabel('Standard error')
    ax.legend(frameon=False,loc='best')
    _grid(ax)
    name='precision_vs_cost.png'
    _finish(fig,destination/name)
    output.append(name)

    fig,ax = plt.subplots(figsize=(8.8,5.2),constrained_layout=True)
    for method in ['A','B','C']:
        group=summary.loc[summary.method.eq(method)].sort_values('n')
        ax.errorbar(group.n,group.median_peak_rss_mib,
            yerr=np.vstack([group.median_peak_rss_mib-group.min_peak_rss_mib,group.max_peak_rss_mib-group.median_peak_rss_mib]),
            color=COLORS[method],marker=MARKERS[method],markersize=8,capsize=4,label=LABELS[method])
    _path_ticks(ax,counts)
    ax.set_ylim(bottom=0)
    ax.set_title('Peak resident memory in fresh workers')
    ax.set_ylabel('Peak RSS (MiB), median and range')
    ax.legend(frameon=False)
    _grid(ax)
    name='memory_scaling.png'
    _finish(fig,destination/name)
    output.append(name)

    # 参数敏感性突出B/C，A与B统计估计相同，不重复重叠曲线。
    fig,axes = plt.subplots(1,3,figsize=(14.6,4.7),constrained_layout=True)
    for ax,factor,title,xlabel in zip(axes,['strike','sigma','steps'],
            ['Strike','Volatility','Monitoring frequency'],['Strike, K','Volatility, sigma','Monitoring dates, m']):
        panel=sensitivity.loc[sensitivity.factor.eq(factor)]
        for method in ['B','C']:
            group=panel.loc[panel.method.eq(method)].sort_values('value')
            ax.plot(group.value,group.median_standard_error,marker=MARKERS[method],
                    markersize=8,color=COLORS[method],label=LABELS[method])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yscale('log')
        if factor=='steps':
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(sorted(panel.value.unique()))
        _grid(ax)
    axes[0].set_ylabel('Median standard error')
    axes[-1].legend(frameon=False,loc='best')
    name='model_sensitivity.png'
    _finish(fig,destination/name)
    output.append(name)

    pilot=sensitivity.loc[sensitivity.factor.eq('pilot_n') & sensitivity.method.eq('C')].sort_values('value')
    fig,axes=plt.subplots(1,3,figsize=(14.6,4.7),constrained_layout=True)
    for ax,field,title,ylabel in zip(axes,
            ['median_beta','variance_reduction_vs_b','time_se2_ratio_vs_b'],
            ['Pilot coefficient','Variance reduction','Time and sampling variance'],
            ['Median beta','B variance / C variance','C / B ratio of time x SE squared']):
        ax.plot(pilot.value,pilot[field],color=COLORS['C'],marker='^',markersize=8)
        ax.set_xscale('log')
        ax.set_xticks(pilot.value.tolist())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Independent pilot paths')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if field != 'median_beta':
            ax.set_yscale('log')
        if field == 'time_se2_ratio_vs_b':
            ax.axhline(1.0,color='#777777',linestyle=':',linewidth=1.3)
        _grid(ax)
    name='pilot_sensitivity.png'
    _finish(fig,destination/name)
    output.append(name)
    return output
