"""串行独立进程实验、断点续跑和按独立随机种子汇总。"""
from __future__ import annotations
import csv
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

MODEL = ['s0','strike','rate','sigma','maturity','steps']
TASK = MODEL + ['n','seed','repeat','method','block_size','pilot_n','min_seconds']
RESULT = ['estimate','standard_error','ci_low','ci_high','beta','pilot_zero_variance',
          'sample_variance','wall_seconds','pilot_seconds','main_seconds',
          'geometric_analytic_price','pilot_paths','actual_method','peak_rss_mib',
          'internal_repeats','cpu_affinity']
RUN_COLUMNS = ['stage','case_id'] + TASK + RESULT + ['process_seconds','status','error']
AGG_COLUMNS = ['independent_seeds','timing_repeats','worker_runs','estimate_mean',
               'across_seed_sd','median_standard_error','median_sample_variance',
               'median_wall_seconds','median_pilot_seconds','median_main_seconds',
               'median_peak_rss_mib','min_peak_rss_mib','max_peak_rss_mib','median_beta',
               'variance_reduction_vs_b','time_se2_ratio_vs_b','speedup_vs_b',
               'pilot_zero_variance_count','actual_methods']


def _key(task):
    model = task.get('model', task)
    method = str(task['method'])
    # 参数元组标识一次任务；A/B 不使用 pilot，所以可跨 pilot 配置复用。
    return tuple(float(model[k]) for k in MODEL) + (
        int(task['n']),int(task['seed']),int(task['repeat']),method,
        int(task['block_size']),int(task['pilot_n']) if method == 'C' else 0,
        float(task['min_seconds']))


def _sensitivity_specs(config):
    base, spec = config['model'], config['sensitivity']
    cases = []
    for plural, factor in [('strikes','strike'),('sigmas','sigma'),('steps','steps')]:
        for value in spec[plural]:
            model = dict(base, **{factor:value})
            case_id = 'base' if model == base else f'{factor}_{value}'
            cases.append((factor,value,case_id,model,int(config['experiment']['pilot_n']),config['experiment']['methods']))
    for pilot in spec['pilot_sizes']:
        case_id = 'base' if int(pilot) == int(config['experiment']['pilot_n']) else f'pilot_{pilot}'
        cases.append(('pilot_n',pilot,case_id,dict(base),int(pilot),['C']))
    return cases


def _tasks(config, stage):
    exp = config['experiment']
    if stage == 'main':
        specifications = [('base',config['model'],int(n),exp['seeds'],int(exp['pilot_n']),exp['methods'])
                          for n in exp['path_counts']]
    else:
        specifications = [(case,model,int(config['sensitivity']['n']),config['sensitivity']['seeds'],pilot,methods)
                          for _,_,case,model,pilot,methods in _sensitivity_specs(config)]
    output, seen = [], set()
    for case_id,model,n,seeds,pilot,methods in specifications:
        for seed_index,seed in enumerate(seeds):
            for repeat in range(1,int(exp['repeats'])+1):
                shift = (seed_index+repeat-1) % len(methods)
                for method in list(methods)[shift:] + list(methods)[:shift]:
                    task = {'case_id':case_id,'model':dict(model),'n':n,'seed':int(seed),
                            'repeat':repeat,'method':method,'block_size':int(exp['block_size']),
                            'pilot_n':pilot,'min_seconds':float(exp['min_seconds'])}
                    key = _key(task)
                    if key not in seen:
                        seen.add(key)
                        output.append(task)
    return output


def _flat(task, stage):
    return {'stage':stage, **{k:v for k,v in task.items() if k != 'model'}, **task['model']}


def _select(frame, model, n, seeds=None, pilot=None, method=None):
    if frame.empty:
        return frame.copy()
    mask = frame['n'].eq(int(n))
    for key in MODEL:
        mask &= frame[key].eq(model[key])
    if seeds is not None:
        mask &= frame['seed'].isin(seeds)
    if method is not None:
        mask &= frame['method'].eq(method)
    if pilot is not None:
        mask &= frame['method'].ne('C') | frame['pilot_n'].eq(int(pilot))
    return frame.loc[mask].copy()


def _aggregate(group, all_runs):
    # 每个独立种子保留一个价格估计；同种子的重复运行用于汇总耗时。
    unique = group.sort_values('repeat').drop_duplicates('seed')
    result = {'independent_seeds':len(unique),
              'timing_repeats':int(group.groupby('seed').size().min()),
              'worker_runs':len(group),'estimate_mean':float(unique.estimate.mean()),
              'across_seed_sd':float(unique.estimate.std(ddof=1)),
              'median_standard_error':float(unique.standard_error.median()),
              'median_sample_variance':float(unique.sample_variance.median()),
              'median_beta':float(unique.beta.median()),
              'pilot_zero_variance_count':int(unique.pilot_zero_variance.astype(str).str.lower().eq('true').sum()),
              'actual_methods':','.join(sorted(set(group.actual_method.astype(str))))}
    for field in ['wall_seconds','pilot_seconds','main_seconds','peak_rss_mib']:
        result['median_'+field] = float(group[field].median())
    result['min_peak_rss_mib'] = float(group.peak_rss_mib.min())
    result['max_peak_rss_mib'] = float(group.peak_rss_mib.max())
    join = MODEL + ['n','seed','repeat']
    reference = all_runs.loc[all_runs.method.eq('B'),join+['sample_variance','standard_error','wall_seconds']]
    paired = group.merge(reference,on=join,suffixes=('','_b'))
    if paired.empty:
        result.update(variance_reduction_vs_b=np.nan,time_se2_ratio_vs_b=np.nan,speedup_vs_b=np.nan)
    else:
        one_seed = paired.sort_values('repeat').drop_duplicates('seed')
        result['variance_reduction_vs_b'] = float((one_seed.sample_variance_b / one_seed.sample_variance.clip(lower=1e-300)).median())
        result['time_se2_ratio_vs_b'] = float((paired.wall_seconds*paired.standard_error**2 /
            (paired.wall_seconds_b*paired.standard_error_b**2).clip(lower=1e-300)).median())
        result['speedup_vs_b'] = float((paired.wall_seconds_b/paired.wall_seconds).median())
    return result


def _write_summaries(config, root, completed):
    wanted = {_key(t) for s in ['main','sensitivity'] for t in _tasks(config,s)}
    rows = [row for key,row in completed.items() if key in wanted]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    exp, sens = config['experiment'], config['sensitivity']
    main_rows, sensitivity_rows = [], []
    for n in exp['path_counts']:
        selected = _select(frame,config['model'],n,exp['seeds'],exp['pilot_n'])
        for method,group in selected.groupby('method',sort=True):
            main_rows.append({'n':int(n),'method':method,**_aggregate(group,frame)})
    for factor,value,case_id,model,pilot,methods in _sensitivity_specs(config):
        selected = _select(frame,model,sens['n'],sens['seeds'],pilot)
        for method in methods:
            group = selected.loc[selected.method.eq(method)]
            if not group.empty:
                sensitivity_rows.append({'factor':factor,'value':value,'case_id':case_id,
                    'n':int(sens['n']),'method':method,**_aggregate(group,frame)})
    raw = root/'results/raw'
    pd.DataFrame(main_rows,columns=['n','method']+AGG_COLUMNS).to_csv(raw/'main_summary.csv',index=False)
    pd.DataFrame(sensitivity_rows,columns=['factor','value','case_id','n','method']+AGG_COLUMNS).to_csv(raw/'sensitivity_summary.csv',index=False)
    # 比较 C 的小样本与 B 的大样本：同时保存 SE、实测耗时及逐种子关系。
    small = _select(frame,config['model'],10000,exp['seeds'],exp['pilot_n'],'C')
    large = _select(frame,config['model'],1000000,exp['seeds'],exp['pilot_n'],'B')
    precision = []
    if not small.empty and not large.empty:
        pairs = small.merge(large,on=['seed','repeat'],suffixes=('_c','_b'))
        single = pairs.sort_values('repeat').drop_duplicates('seed')
        se_c, se_b = float(single.standard_error_c.median()), float(single.standard_error_b.median())
        target = se_c <= se_b
        ratio = float((pairs.wall_seconds_b/pairs.wall_seconds_c).median())
        precision.append({'method':'C','n':10000,'baseline_method':'B','baseline_n':1000000,
            'independent_seeds':len(single),'median_standard_error_c':se_c,'median_standard_error_b':se_b,
            'standard_error_ratio_c_over_b':se_c/se_b,'precision_target_met':target,
            'median_wall_seconds_c':float(small.wall_seconds.median()),
            'median_wall_seconds_b':float(large.wall_seconds.median()),
            'nominal_time_ratio_b_over_c':ratio,'matched_precision_time_ratio':ratio if target else np.nan,
            'paired_seed_precision_fraction':float((single.standard_error_c<=single.standard_error_b).mean())})
    pd.DataFrame(precision, columns=['method','n','baseline_method','baseline_n','independent_seeds','median_standard_error_c','median_standard_error_b','standard_error_ratio_c_over_b','precision_target_met','median_wall_seconds_c','median_wall_seconds_b','nominal_time_ratio_b_over_c','matched_precision_time_ratio','paired_seed_precision_fraction']).to_csv(raw/'precision_comparison.csv',index=False)


def _metadata_write(path, metadata, targets, completed, config):
    metadata['progress'] = {'completed':sum(key in completed for key in targets),
                            'total':len(targets),
                            'main_completed':sum(_key(t) in completed for t in _tasks(config,'main')),
                            'main_total':len(_tasks(config,'main')),
                            'sensitivity_completed':sum(_key(t) in completed for t in _tasks(config,'sensitivity')),
                            'sensitivity_total':len(_tasks(config,'sensitivity'))}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def run_experiments(config, root: Path, stage='all'):
    if stage not in ['all','main','sensitivity']:
        raise ValueError('stage must be all, main or sensitivity')
    root = Path(root)
    raw = root/'results/raw'
    raw.mkdir(parents=True,exist_ok=True)
    run_path, meta_path = raw/'runs.csv', raw/'metadata.json'
    completed = {}
    if run_path.exists() and run_path.stat().st_size:
        previous = pd.read_csv(run_path)
        for row in previous.to_dict('records'):
            if row.get('status') == 'completed':
                completed[_key(row)] = row
    phases = ['main','sensitivity'] if stage == 'all' else [stage]
    plans = {phase:_tasks(config,phase) for phase in phases}
    targets = {_key(t) for tasks in plans.values() for t in tasks}
    cpu_model = next((line.split(':',1)[1].strip() for line in
                     Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
    metadata = {'state':'running','stage_requested':stage,'config':config,
        'software':{name:version(name) for name in ['numpy','scipy','pandas','matplotlib','threadpoolctl']},
        'python':platform.python_version(),'platform':platform.platform(),
        'hardware':{'cpu_model':cpu_model,'worker_cpu':0,'numerical_threads':1,
                    'max_authorized_cores':6,'parent_cpu_affinity':sorted(os.sched_getaffinity(0))},
        'timing':{'wall_seconds':'Mean per-estimate pricing time, including the independent CV pilot and all main paths; excludes subprocess startup, warmup, I/O and summary calculations.',
                  'internal_repeats':'Same seed repeated until at least min_seconds cumulative pricing time; this improves timing only and does not increase N.',
                  'process_seconds':'Fresh worker lifetime including startup, warmup and internal timing repetitions.',
                  'memory':'Fresh-process ru_maxrss converted to MiB; includes interpreter and warmup, not a baseline-subtracted allocation.'},
        'statistics':{'independent_units':'One estimate per unique seed; timing repeat is not a new independent sample.',
                      'variance_reduction_vs_b':'Median B sample variance / method sample variance, paired by identical model,N,seed,repeat; one repeat per seed for statistical ratios.',
                      'time_se2_ratio_vs_b':'Median method(time*SE^2) / B(time*SE^2), paired by model,N,seed,repeat; lower is better.',
                      'price_ci_figure':'Representative seed 2026, first timing repeat; its individual 95% CI is not attached to the mean of ten seeds.'},
        'new_worker_calls':0,'skipped_matching_requests':0}
    _metadata_write(meta_path,metadata,targets,completed,config)
    env = dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
        env[key] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    with run_path.open('a',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=RUN_COLUMNS,extrasaction='ignore')
        if stream.tell() == 0:
            writer.writeheader()
            stream.flush()
        for phase in phases:
            for task in plans[phase]:
                key = _key(task)
                if key in completed:
                    metadata['skipped_matching_requests'] += 1
                    continue
                metadata['current'] = {'stage':phase,**task}
                _metadata_write(meta_path,metadata,targets,completed,config)
                record = _flat(task,phase)
                started = perf_counter()
                try:
                    # 每个测量在全新进程中完成，RSS峰值不会沿用上一个大任务。
                    process = subprocess.run([sys.executable,'-B',str(root/'worker.py'),'--task-json',json.dumps(task)],
                        cwd=root,env=env,text=True,capture_output=True,timeout=900,check=True)
                    result = json.loads(process.stdout)
                    record.update(result)
                    record['cpu_affinity'] = json.dumps(result['cpu_affinity'])
                    record.update(process_seconds=perf_counter()-started,status='completed',error='')
                except Exception as exc:
                    record.update(process_seconds=perf_counter()-started,status='failed',error=str(exc))
                    writer.writerow(record)
                    stream.flush()
                    metadata.update(state='failed',error=str(exc))
                    _metadata_write(meta_path,metadata,targets,completed,config)
                    raise RuntimeError(f'Worker failed: {phase}/{task["case_id"]}/{task["method"]}') from exc
                writer.writerow(record)
                stream.flush()
                os.fsync(stream.fileno())
                completed[key] = record
                metadata['new_worker_calls'] += 1
                _metadata_write(meta_path,metadata,targets,completed,config)
                done = metadata['progress']
                print(f'[{done["completed"]}/{done["total"]}] {phase} {task["case_id"]} N={task["n"]} {task["method"]} seed={task["seed"]} repeat={task["repeat"]}',flush=True)
            _write_summaries(config,root,completed)
    metadata.update(state='completed',current=None)
    _metadata_write(meta_path,metadata,targets,completed,config)
    return metadata
