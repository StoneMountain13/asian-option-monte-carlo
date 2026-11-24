"""服务器实验与图表的统一入口。"""
from __future__ import annotations
import os
# 数值库环境在导入计算模块前固定，所有子进程继承同一资源上限。
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Asian option Monte Carlo experiments")
    parser.add_argument("--stage",choices=["main","sensitivity","figures","all"],default="all")
    parser.add_argument("--config",default="configs/default.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root/args.config).read_text(encoding="utf-8"))
    # 主进程亲和性最多六核；正式估价worker另外固定到一个核。
    if hasattr(os,"sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0,set(allowed[:int(config["resources"]["max_cores"])]))
    (root/"logs").mkdir(exist_ok=True)
    started=time.perf_counter()
    start_utc=datetime.now(timezone.utc).isoformat()
    try:
        if args.stage!="figures":
            from src.experiments import run_experiments
            run_experiments(config,root,stage=args.stage)
        from src.plotting import create_figures
        create_figures(root)
        elapsed=time.perf_counter()-started
        state={"state":"complete","stage":args.stage,"started_at":start_utc,
               "finished_at":datetime.now(timezone.utc).isoformat(),"wall_seconds":elapsed}
        (root/"logs/completion.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        print("COMPLETE "+json.dumps(state),flush=True)
    except BaseException as error:
        state={"state":"failed","stage":args.stage,"error":str(error),
               "elapsed_seconds":time.perf_counter()-started}
        (root/"logs/completion.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        raise

if __name__=="__main__":
    main()
