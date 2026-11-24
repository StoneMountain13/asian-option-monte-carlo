"""读取正式 Asian Monte Carlo 结果，在服务器生成英文论文和展示稿。"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
B = chr(92)


def read_csv(name):
    # 从随项目保存的正式 CSV 读取文档数据。
    with (ROOT/'results'/'raw'/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def num(row,key):
    return float(row[key])


def fmt(value,decimals=2):
    x=float(value)
    return f'{x:,.{decimals}f}' if math.isfinite(x) else '--'


def table(spec,headers,rows):
    # 将 CSV 数值转换为可编辑的 LaTeX 表格。
    lines=[B+'begin{tabular}{'+spec+'}',B+'toprule',' & '.join(headers)+' '+B*2,B+'midrule']
    lines+=[' & '.join(str(x) for x in row)+' '+B*2 for row in rows]
    return '\n'.join(lines+[B+'bottomrule',B+'end{tabular}'])


def command(name,body):
    return B+'newcommand{'+B+name+'}{'+body+'}\n'


def collect_values():
    """从正式记录读取展示所用的统计量。"""
    summary = read_csv('main_summary.csv')
    sensitivity = read_csv('sensitivity_summary.csv')
    cross = read_csv('precision_comparison.csv')[0]
    with (ROOT/'configs'/'default.json').open(encoding='utf-8') as f:
        pilot_paths = json.load(f)['experiment']['pilot_n']
    by = {(int(row['n']), row['method']): row for row in summary}
    largest = max(n for n, _ in by)
    a, b, c = [by[largest, method] for method in 'ABC']
    pilots = sorted(
        [row for row in sensitivity if row['factor']=='pilot_n' and row['method']=='C'],
        key=lambda row: num(row, 'value'),
    )
    return {
        'SlideLargest': f'{largest:,}',
        'SlideMemoryA': fmt(num(a, 'median_peak_rss_mib'), 1),
        'SlideMemoryB': fmt(num(b, 'median_peak_rss_mib'), 1),
        'SlideSERatio': fmt(num(b, 'median_standard_error') / num(c, 'median_standard_error'), 0),
        'SlideTimeB': fmt(num(b, 'median_wall_seconds'), 4),
        'SlideTimeC': fmt(num(c, 'median_wall_seconds'), 4),
        'SlideCrossRatio': fmt(num(cross, 'nominal_time_ratio_b_over_c'), 2),
        'SlidePilotPaths': f'{pilot_paths:,}',
        'SlideCrossTable': table(
            '@{}lrr@{}', ['指标', 'B', 'C'],
            [
                ['正式路径', fmt(num(cross, 'baseline_n'), 0), fmt(num(cross, 'n'), 0)],
                ['标准误', fmt(num(cross, 'median_standard_error_b'), 5),
                 fmt(num(cross, 'median_standard_error_c'), 5)],
                ['耗时（毫秒）', fmt(1000*num(cross, 'median_wall_seconds_b'), 1),
                 fmt(1000*num(cross, 'median_wall_seconds_c'), 2)],
            ],
        ),
        'SlidePilotTable': table(
            'rrr', ['预备路径数', 'B/C 支付方差比', 'C 的耗时（毫秒）'],
            [
                [fmt(num(row, 'value'), 0), fmt(num(row, 'variance_reduction_vs_b'), 1),
                 fmt(1000*num(row, 'median_wall_seconds'), 2)]
                for row in pilots
            ],
        ),
    }


def write_parallel_values(docs):
    """论文与展示稿共用同一份配对并行实验数据。"""
    rows = read_csv('parallel_runs.csv')
    serial = [row for row in rows if int(row['workers']) == 1]
    parallel = [row for row in rows if int(row['workers']) == 6]
    serial_time = median(num(row, 'wall_seconds') for row in serial)
    parallel_time = median(num(row, 'wall_seconds') for row in parallel)
    values = {
        'ParallelSerialMs': fmt(1000*serial_time, 2),
        'ParallelSixMs': fmt(1000*parallel_time, 2),
        'ParallelSpeedup': fmt(serial_time/parallel_time, 2),
        'ParallelPrice': fmt(num(serial[0], 'estimate'), 6),
        'ParallelSE': fmt(num(serial[0], 'standard_error'), 6),
    }
    (docs/'parallel-values.tex').write_text(
        ''.join(command(key,value) for key,value in values.items()),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=('report','slides','all'),default='all',
                        help='选择论文、展示稿或两者；默认保持全部构建')
    parser.add_argument('--sources-only',action='store_true',help='只更新所选文档的数值宏')
    parser.add_argument('--render',action='store_true',help='渲染所选PDF的全部页面')
    parser.add_argument('--tectonic',default=shutil.which('tectonic') or '/opt/miniconda3/envs/R/bin/tectonic')
    args = parser.parse_args()
    targets = ('report','slides') if args.target=='all' else (args.target,)
    docs = ROOT/'docs'
    write_parallel_values(docs)

    # 报告拥有独立的数值源和彩色图，单独重建时不触碰展示稿。
    if 'report' in targets:
        from src.paper_artifacts import write_paper_values,create_paper_figures
        write_paper_values(ROOT)
        if not args.sources_only:
            create_paper_figures(ROOT)
    if 'slides' in targets:
        values = collect_values()
        (docs/'values.tex').write_text(
            ''.join(command(k,v) for k,v in values.items()),encoding='utf-8')
        if not args.sources_only:
            from src.slide_figures import create_slide_figures
            create_slide_figures(ROOT)
    # 讲稿按叙事顺序单独维护，不在构建时追加自动生成的结果段落。
    if args.sources_only:
        print('Refreshed numerical sources for '+', '.join(targets)+'.')
        return

    build = docs/'.build'
    build.mkdir(exist_ok=True)
    for name in targets:
        log_path = build/(name+'-build.log')
        with log_path.open('w',encoding='utf-8') as log:
            done = subprocess.run(
                [args.tectonic,'--outdir',str(build),name+'.tex'],
                cwd=docs,stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:
            print(log_path.read_text(encoding='utf-8',errors='replace')[-6000:])
            raise RuntimeError(name+'.tex compilation failed')
        shutil.copy2(build/(name+'.pdf'),ROOT/(name+'.pdf'))
        info = subprocess.check_output(['pdfinfo',str(ROOT/(name+'.pdf'))],text=True)
        pages = next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:'))
        print(f'{name}.pdf: {pages} pages')
        if args.render:
            target = build/(name+'-pages')
            target.mkdir(exist_ok=True)
            for stale in target.glob(name+'-*.png'):
                stale.unlink()
            subprocess.run(
                ['pdftoppm','-r','120','-png',str(ROOT/(name+'.pdf')),str(target/name)],
                check=True)
    print('Built '+', '.join(targets)+' from existing formal records.')


if __name__=='__main__':
    main()
