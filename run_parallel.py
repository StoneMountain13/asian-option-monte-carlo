"""固定一百万条路径，对比单核串行分块和六进程分块。"""
from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from statistics import median
from time import perf_counter

# 子进程在导入 NumPy 前继承这组设置，每个进程使用单线程数值库。
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from src.pricing import _payoffs

PATHS = 1_000_000
WORKERS = 6
REPEATS = 3


def _block_statistics(job):
    count, model, block_seed = job
    payoff, _ = _payoffs(np.random.default_rng(block_seed), count, model, False)
    mean = float(payoff.mean())
    centered = payoff - mean
    # 只返回三个统计量，不向主进程传输路径或逐条赔付。
    return count, mean, float(centered @ centered)


def _aggregate(blocks):
    blocks = list(blocks)
    total = sum(count for count, _, _ in blocks)
    mean = math.fsum(count * block_mean for count, block_mean, _ in blocks) / total
    m2 = math.fsum(
        block_m2 + count * (block_mean - mean) ** 2
        for count, block_mean, block_m2 in blocks
    )
    return total, mean, math.sqrt(m2 / (total * (total - 1)))


def _measure(dispatch, jobs, workers, cpus, seed, block_size):
    _aggregate(dispatch(_block_statistics, jobs))  # 预热在计时之外。
    rows = []
    for repeat in range(1, REPEATS + 1):
        start = perf_counter()
        count, price, se = _aggregate(dispatch(_block_statistics, jobs))
        elapsed = perf_counter() - start
        rows.append({
            "workers": workers, "repeat": repeat, "n": count, "seed": seed,
            "block_size": block_size, "blocks": len(jobs), "estimate": price,
            "standard_error": se, "wall_seconds": elapsed,
            "cpu_affinity": " ".join(map(str, cpus)),
        })
    return rows


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / "configs/default.json").read_text(encoding="utf-8"))
    model = config["model"]
    block_size = config["experiment"]["block_size"]
    seed = config["experiment"]["seeds"][0]
    counts = [min(block_size, PATHS - start) for start in range(0, PATHS, block_size)]
    # 随机流由块编号决定，两种执行方式使用同样的任务和随机路径。
    seeds = np.random.SeedSequence(seed).spawn(len(counts))
    jobs = [(count, model, rng_seed) for count, rng_seed in zip(counts, seeds)]

    original_affinity = os.sched_getaffinity(0)
    cpus = sorted(original_affinity)[:WORKERS]
    if len(cpus) < WORKERS:
        raise RuntimeError(f"The benchmark needs {WORKERS} available CPU cores.")
    try:
        os.sched_setaffinity(0, {cpus[0]})
        serial = _measure(map, jobs, 1, cpus[:1], seed, block_size)
        os.sched_setaffinity(0, cpus)
        # map 按提交顺序返回，串行与并行合并统计量时顺序相同。
        with mp.get_context("spawn").Pool(WORKERS) as pool:
            parallel = _measure(pool.map, jobs, WORKERS, cpus, seed, block_size)
    finally:
        os.sched_setaffinity(0, original_affinity)

    rows = serial + parallel
    for row in rows:
        if row["n"] != PATHS or any(
            not math.isclose(row[key], serial[0][key], rel_tol=0, abs_tol=1e-12)
            for key in ("estimate", "standard_error")
        ):
            raise RuntimeError("Serial and parallel estimates differ on the same paths.")

    result = root / "results/raw/parallel_runs.csv"
    with result.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    serial_time = median(row["wall_seconds"] for row in serial)
    parallel_time = median(row["wall_seconds"] for row in parallel)
    print(f"{PATHS:,} matched paths across {len(jobs)} blocks")
    print(f"Serial: {serial_time*1000:.2f} ms; parallel: {parallel_time*1000:.2f} ms; "
          f"speedup: {serial_time/parallel_time:.2f}x")
    print(f"Price: {serial[0]['estimate']:.12f}; SE: {serial[0]['standard_error']:.12f}")
    print(result.relative_to(root))


if __name__ == "__main__":
    main()
