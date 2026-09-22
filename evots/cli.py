from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .config import config_from
from .data import synthetic, save_dataset, prepare, import_csv, import_bee
from .io import write_json


def main(argv=None):
    p=argparse.ArgumentParser(description="Independent EvoTS-Agent experiments with OpenRouter")
    sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate",help="Create synthetic benchmark observations and boundary labels")
    g.add_argument("--kind",choices=["ou","mean_variance"],default="mean_variance")
    g.add_argument("--output",required=True); g.add_argument("--n",type=int,default=3000)
    g.add_argument("--count",type=int,default=3); g.add_argument("--seed",type=int,default=42)
    g.add_argument("--dimensions",type=int,default=1); g.add_argument("--segment-length",type=int,default=125)
    g=sub.add_parser("prepare",help="Create chronological 60/20/20 splits")
    g.add_argument("--data",required=True); g.add_argument("--output",required=True)
    g.add_argument("--ratios",type=float,nargs=3,default=[0.6,0.2,0.2])
    g=sub.add_parser("import-csv"); g.add_argument("--csv",required=True); g.add_argument("--labels",required=True)
    g.add_argument("--columns",nargs="+",required=True); g.add_argument("--output",required=True)
    g=sub.add_parser("import-bee"); g.add_argument("--mat",required=True); g.add_argument("--output",required=True)
    g=sub.add_parser("fetch-bee"); g.add_argument("--output",required=True); g.add_argument("--cache",default="data/bee-raw")
    g=sub.add_parser("stitch-adia"); g.add_argument("--manifest",required=True); g.add_argument("--output",required=True)
    g.add_argument("--half-window",type=int,default=150); g.add_argument("--bucket-width",type=float,default=1.0)
    g.add_argument("--windows-per-stream",type=int,default=6)
    for name in ["run","baseline","matrix"]:
        g=sub.add_parser(name); g.add_argument("--prepared",required=True); g.add_argument("--output",required=True)
        g.add_argument("--config"); g.add_argument("--seed",type=int); g.add_argument("--mock",action="store_true")
        if name=="run": g.add_argument("--resume",action="store_true")
        elif name=="baseline": g.add_argument("--method",default="random_search")
        else:
            g.add_argument("--seeds",type=int,nargs="+",default=[11,22,33])
            g.add_argument("--variants",nargs="+",default=["full","no_alternative","no_recombination"])
            g.add_argument("--models",nargs="+"); g.add_argument("--include-baselines",action="store_true")
    for name in ["evaluate","report"]:
        g=sub.add_parser(name); g.add_argument("--run",required=True)
        if name=="evaluate": g.add_argument("--prepared",help="Relocated prepared data; all fingerprints must match")
    g=sub.add_parser("compare"); g.add_argument("--matrix",required=True); g.add_argument("--a",default="full")
    g.add_argument("--b",default="no_alternative"); g.add_argument("--model",required=True); g.add_argument("--output",required=True)
    g=sub.add_parser("doctor"); g.add_argument("--list-models",action="store_true"); g.add_argument("--config")
    args=p.parse_args(argv)
    try:
        result=None
        if args.command=="generate":
            settings={"kind":args.kind,"n":args.n,"count":args.count,"seed":args.seed,"dimensions":args.dimensions,"segment_length":args.segment_length}
            save_dataset(args.output,synthetic(**settings),{"generator":settings,"author_exact":False})
        elif args.command=="prepare": prepare(args.data,args.output,args.ratios)
        elif args.command=="import-csv": import_csv(args.csv,args.labels,args.columns,args.output)
        elif args.command=="import-bee": import_bee(args.mat,args.output)
        elif args.command=="fetch-bee":
            from .datasets_extra import fetch_bee
            fetch_bee(args.output,args.cache)
        elif args.command=="stitch-adia":
            from .datasets_extra import stitch_adia
            stitch_adia(args.manifest,args.output,args.half_window,args.bucket_width,args.windows_per_stream)
        elif args.command in {"run","baseline","matrix"}:
            cfg=config_from(args.config)
            if args.seed is not None: cfg["seed"]=args.seed
            if args.mock: cfg["mock"]=True
            if args.command=="run":
                from .runner import run
                result=run(args.prepared,args.output,cfg,args.resume)
            elif args.command=="baseline":
                from .experiments import baseline
                baseline(args.prepared,args.output,cfg,args.method)
            else:
                from .experiments import matrix
                result=matrix(args.prepared,args.output,cfg,args.seeds,args.variants,args.models,args.include_baselines)
        elif args.command=="evaluate":
            from .runner import evaluate_frozen
            result=evaluate_frozen(args.run,args.prepared)
        elif args.command=="report":
            from .report import report_run
            result={"report":report_run(args.run)}
        elif args.command=="compare":
            from .experiments import paired_comparison
            result=paired_comparison(args.matrix,args.a,args.b,args.model)
            write_json(args.output,result)
        elif args.command=="doctor":
            from .bank import available_bank
            cfg=config_from(args.config)
            result={"python":sys.version.split()[0],"api_key_set":bool(os.environ.get("OPENROUTER_API_KEY")),
                    "requested_model":os.environ.get("OPENROUTER_MODEL") or cfg["llm"]["model"],"bank":available_bank()}
            if args.list_models:
                import httpx
                with httpx.Client(timeout=30) as client:
                    response=client.get("https://openrouter.ai/api/v1/models"); response.raise_for_status()
                    models=response.json()["data"]
                result["model_info"]=[{k:m.get(k) for k in ["id","context_length","supported_parameters","pricing"]} for m in models if m["id"]==result["requested_model"]]
                if not result["model_info"]: result["warning"]="Requested model not found in live catalog"
        if result is not None: print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    except Exception as error:
        text=str(error)
        key=os.environ.get("OPENROUTER_API_KEY")
        if key: text=text.replace(key,"[REDACTED]")
        print(f"Error: {text}",file=sys.stderr)
        return_code=1
        if argv is None: raise SystemExit(return_code) from None
        return return_code
    return 0
