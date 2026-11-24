# 基于控制变量的亚式期权高效蒙特卡洛定价

学号：**2026001426**。项目由本人独立完成，负责估价实现、实验分析、论文和汇报。

在风险中性几何布朗运动下估计算术平均亚式看涨期权价格。比较全量向量化、分块模拟和分块加几何控制变量，并在相同路径上比较单核串行与六进程并行。价格路径由模型模拟，实验设定见 `configs/default.json`。

## 运行与复现

正式实验环境为 Linux、Python 3.10.19、NumPy 1.26.4 和 SciPy 1.15.3。依赖版本见 `requirements.txt`。

```bash
python -m pip install -r requirements.txt
```

小规模核对。下面以相同种子比较全量与分块算法的估价和标准误。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B - <<'PY'
from math import isclose
from src.pricing import estimate

model = dict(s0=100, strike=100, rate=0.03, sigma=0.2, maturity=1, steps=52)
a, b = (estimate(model, method, n=2000, seed=31415, block_size=500)
        for method in ("A", "B"))
for field in ("estimate", "standard_error"):
    assert isclose(a[field], b[field], rel_tol=0, abs_tol=1e-12), field
print(b["estimate"], b["standard_error"])
PY
```

```bash
python -B run.py --stage all                  # 主实验、敏感性实验和绘图
python -B run_parallel.py                     # 单核与六进程分块对照
python -B build_documents.py --target all     # 由现有记录重建两份 PDF
```

`results/raw/` 保存主要实验记录，`results/figures/` 保存复现用图表。重建文档需安装 Tectonic、Poppler、Fontconfig 和 Noto CJK 字体（简体中文衬线与无衬线字体）。

## 资料与 AI 使用

几何控制变量参考 Kemna 和 Vorst（1990）及 QuantLib 的 AsianOption 示例，分块方差合并参考 Chan、Golub 和 LeVeque（1983）。路径模拟、解析价格和控制变量由本项目实现，数组运算、随机数与分布函数使用 NumPy 和 SciPy。

使用 ChatGPT（Codex）辅助代码编写。用小规模对照及相同路径的串行、并行实验核对价格和标准误，并将报告数值与实验记录逐项核对。

## 提交版本

仓库：[StoneMountain13/asian-option-monte-carlo](https://github.com/StoneMountain13/asian-option-monte-carlo)

提交包版本：`$Format:%H$`（由 Git 导出时填入）。
