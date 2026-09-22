from __future__ import annotations
from copy import deepcopy
from .io import read_json
from .bank import BANK


DEFAULT = {
    "seed":42,"tolerance":10,"mode":"spec","mock":False,
    "bank":["pelt","bottom_up","window","bayesian_offline","spectral"],
    "agent":{"k":3,"iterations":8,"epsilon":0.005,"alternative":True,"recombination":True,
             "eda":True,"vision":False,"repairs":1},
    "llm":{"model":"openai/gpt-4o","temperature":0.2,"max_tokens":6000,"send_seed":False,
           "json_mode":True,"reasoning":None,"provider":{"allow_fallbacks":False},
           "timeout_seconds":120,"retries":2,"max_retry_wait_seconds":60,"max_requests":60,"max_cost_usd":None},
    "execution":{"timeout_seconds":180,"image":"evots-worker:local","memory":"2g","cpus":2},
}


def config_from(path=None,overrides=None):
    cfg=deepcopy(DEFAULT)
    def merge(dst,src):
        for k,v in src.items():
            if k not in dst: raise ValueError(f"Unknown configuration field: {k}")
            if isinstance(dst[k],dict) and isinstance(v,dict):
                if k=="provider": dst[k]=v
                else: merge(dst[k],v)
            else: dst[k]=v
    if path: merge(cfg,read_json(path))
    if overrides: merge(cfg,overrides)
    if cfg["mode"] not in {"spec","python"}: raise ValueError("mode must be spec or python")
    if not cfg["bank"] or len(set(cfg["bank"]))!=len(cfg["bank"]) or any(m not in BANK for m in cfg["bank"]):
        raise ValueError("bank must contain unique supported detectors")
    for key in ("seed","tolerance"):
        if type(cfg[key]) is not int or cfg[key]<0: raise ValueError(f"{key} must be nonnegative integer")
    a=cfg["agent"]
    if type(a["k"]) is not int or not 1<=a["k"]<=len(cfg["bank"]): raise ValueError("k exceeds available bank")
    if type(a["iterations"]) is not int or not 1<=a["iterations"]<=1000: raise ValueError("iterations must be 1..1000")
    if not 0<=a["epsilon"]<=1: raise ValueError("epsilon must be 0..1")
    if type(a["repairs"]) is not int or not 0<=a["repairs"]<=5: raise ValueError("repairs must be 0..5")
    for key in ("alternative","recombination","eda","vision"):
        if type(a[key]) is not bool: raise ValueError(f"{key} must be boolean")
    if a["vision"] and not a["eda"]: raise ValueError("vision requires EDA")
    if type(cfg["mock"]) is not bool: raise ValueError("mock must be boolean")
    if cfg["execution"]["timeout_seconds"]<=0: raise ValueError("timeout must be positive")
    if cfg["llm"]["max_requests"]<1 or cfg["llm"]["max_tokens"]<1: raise ValueError("Invalid request budget")
    return cfg
