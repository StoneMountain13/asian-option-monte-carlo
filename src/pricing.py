"""用模拟路径估计算术平均亚式看涨期权的当前价格。
A 保存全部路径；B 分块；C 在分块基础上用几何平均支付修正模拟噪声。
"""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np
from scipy.special import ndtr


def geometric_price(model: dict) -> float:
    """计算几何平均期权的解析价格，作为控制变量的已知期望。
    观察时点为 T/m,...,T；mu 和 variance 是对数几何平均值的均值和方差。
    """
    s0, strike = float(model['s0']), float(model['strike'])
    rate, sigma = float(model['rate']), float(model['sigma'])
    maturity, m = float(model['maturity']), int(model['steps'])
    mu = math.log(s0) + (rate - 0.5 * sigma**2) * maturity * (m + 1) / (2 * m)
    variance = sigma**2 * maturity * (m + 1) * (2 * m + 1) / (6 * m**2)
    discount = math.exp(-rate * maturity)
    if variance == 0.0:
        return discount * max(math.exp(mu) - strike, 0.0)
    std = math.sqrt(variance)
    d2 = (mu - math.log(strike)) / std
    return discount * (math.exp(mu + variance / 2) * ndtr(d2 + std) - strike * ndtr(d2))


def _payoffs(rng, count, model, need_control):
    # 每行是一条路径，每列是一个观察时点；先生成标准正态数。
    # 累加正态增量得到几何布朗运动（GBM）的对数价格，三种方法共用此过程。
    m = int(model['steps'])
    dt = float(model['maturity']) / m
    rate, sigma = float(model['rate']), float(model['sigma'])
    paths = rng.standard_normal((count, m))
    paths *= sigma * math.sqrt(dt)
    paths += (rate - 0.5 * sigma**2) * dt
    np.cumsum(paths, axis=1, out=paths)
    paths += math.log(float(model['s0']))
    discount = math.exp(-rate * float(model['maturity']))
    strike = float(model['strike'])
    # 此时 paths 保存对数价格。先取几何平均，再原地转成价格求算术平均。
    # 两种支付都折现到时间零，因此可以直接组合成价格估计。
    control = (discount * np.maximum(np.exp(paths.mean(axis=1)) - strike, 0.0)
               if need_control else None)
    np.exp(paths, out=paths)
    arithmetic = discount * np.maximum(paths.mean(axis=1) - strike, 0.0)
    return arithmetic, control


def _merge_moments(count, mean, m2, values):
    """合并已处理样本与新块的统计量。
    count、mean、m2 分别表示样本数、均值、相对均值的离差平方和。
    合并后只保存这三个数，即可继续计算最终均值与样本方差。
    """
    size = len(values)
    block_mean = float(values.mean())
    centered = values - block_mean
    block_m2 = float(centered @ centered)
    delta = block_mean - mean
    total = count + size
    return (total, mean + delta * size / total,
            m2 + block_m2 + delta * delta * count * size / total)


def estimate(model: dict, method: str, n: int, seed: int,
             block_size: int = 20000, pilot_n: int = 5000) -> dict:
    """返回 n 条正式路径的价格、标准误、区间及耗时（秒）。
    C 先用 pilot_n 条独立预备路径估计修正系数 beta，再固定系数计算正式样本。
    n 只计正式路径；wall_seconds 包含预备样本和全部正式计算。
    """
    started = perf_counter()
    method, n = str(method).upper(), int(n)
    block_size, pilot_n = int(block_size), int(pilot_n)
    if method not in {'A', 'B', 'C'} or n < 2 or block_size < 1:
        raise ValueError('method must be A/B/C, n >= 2, and block_size >= 1')
    if method == 'C' and pilot_n < 2:
        raise ValueError('control-variate pilot_n must be at least 2')
    beta, pilot_seconds, zero_variance = 0.0, 0.0, False
    # 三种方法均计算并报告几何解析价；C 中它也属于 pilot/系数阶段。
    known_g = float(geometric_price(model))
    if method == 'C':
        # pilot 同时计算两种支付，用样本协方差/方差拟合 beta。
        pilot_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7]))
        py, pg = _payoffs(pilot_rng, pilot_n, model, True)
        cy, cg = py - py.mean(), pg - pg.mean()
        denominator = float(cg @ cg)
        zero_variance = bool(denominator == 0.0 or np.ptp(pg) == 0.0)
        beta = 0.0 if zero_variance else float((cy @ cg) / denominator)
        pilot_seconds = perf_counter() - started
    # 共同主随机流使 A/B/C 使用相同正式路径，pilot 另有独立随机流。
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 11]))
    count, mean, m2 = 0, 0.0, 0.0
    chunk = n if method == 'A' else block_size
    while count < n:
        y, g = _payoffs(rng, min(chunk, n - count), model, method == 'C')
        values = y - beta * (g - known_g) if method == 'C' else y
        count, mean, m2 = _merge_moments(count, mean, m2, values)
    # 标准误衡量 n 条正式路径的平均值有多大模拟波动。
    sample_variance = m2 / (n - 1)
    standard_error = math.sqrt(sample_variance / n)
    ci_low, ci_high = mean - 1.96 * standard_error, mean + 1.96 * standard_error
    wall_seconds = perf_counter() - started
    return {
        'estimate': mean, 'standard_error': standard_error,
        'ci_low': ci_low, 'ci_high': ci_high, 'beta': beta,
        'pilot_zero_variance': zero_variance, 'sample_variance': sample_variance,
        'wall_seconds': wall_seconds, 'pilot_seconds': pilot_seconds,
        'main_seconds': wall_seconds - pilot_seconds,
        'geometric_analytic_price': known_g,
        'pilot_paths': pilot_n if method == 'C' else 0,
        'actual_method': 'C_beta0' if method == 'C' and zero_variance else method,
    }
