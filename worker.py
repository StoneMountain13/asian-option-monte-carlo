"""单核独立进程 worker：只向标准输出写一行 JSON。"""
from __future__ import annotations

import os
# 环境与 CPU 约束须先于 NumPy/SciPy 导入。
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
os.sched_setaffinity(0, {0})

import argparse
import json
import resource
from src.pricing import estimate


def run_task(task: dict) -> dict:
    model = task['model']
    method, n, seed = task['method'], int(task['n']), int(task['seed'])
    block_size = int(task.get('block_size', 20000))
    pilot_n = int(task.get('pilot_n', 5000))
    minimum = max(0.0, float(task.get('min_seconds', 0.2)))
    # 独立小样本预热不进入核心计时，也不复用作正式结果。
    estimate(model, method, min(2048, n), seed + 2147483647,
             block_size=block_size, pilot_n=min(2048, pilot_n))
    totals = {'wall_seconds': 0.0, 'pilot_seconds': 0.0, 'main_seconds': 0.0}
    repeats = 0
    while repeats == 0 or totals['wall_seconds'] < minimum:
        result = estimate(model, method, n, seed, block_size=block_size, pilot_n=pilot_n)
        for key in totals:
            totals[key] += result[key]
        repeats += 1
    # 内部重复使用相同 seed，只平均耗时；价格与 SE 仍对应单次 n 条主路径。
    for key in totals:
        result[key] = totals[key] / repeats
    output = {key: value for key, value in task.items() if key != 'model'}
    output.update(model)
    output.update(result)
    output.update({
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        'internal_repeats': repeats,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
    })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-json', required=True)
    args = parser.parse_args()
    output = run_task(json.loads(args.task_json))
    print(json.dumps(output, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
