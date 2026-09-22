from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
import numpy as np

from .bank import script_for_spec
from .data import load_dataset
from .execution import execute, evaluate_predictions
from .io import digest, file_hash, read_json, write_json, source_hash
from .runner import run, evaluate_frozen, scientific_config


VARIANTS={"full":{},"no_alternative":{"alternative":False},"no_recombination":{"recombination":False},
          "revision_only":{"alternative":False,"recombination":False},"no_eda":{"eda":False,"vision":False}}


def random_spec(rng,bank):
    return {"model":str(rng.choice(bank)),"penalty":float(rng.choice([2,4,8,16,32,64])),
            "cost":str(rng.choice(["l2","rbf","normal"])),"window":int(rng.choice([10,20,30,50])),
            "min_size":int(rng.choice([5,10,20])),"threshold":float(rng.choice([1.5,2.5,4])),
            "representation":str(rng.choice(["raw","raw_squared","difference"])),
            "posterior_threshold":float(rng.choice([0.2,0.5,0.8])),"epochs":5}


def baseline(prepared,output,config,method="random_search"):
    prepared,output=Path(prepared).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    config=copy.deepcopy(config); config["mode"]="spec"
    config["mock"]=False
    identity={"config":scientific_config(config),"inputs":{s:file_hash(prepared/s/"dataset.json") for s in ["train","validation","test"]},
              "protocol":file_hash(prepared/"protocol.json"),"source_sha256":source_hash()}
    write_json(output/"manifest.json",{"identity":identity,"prepared":str(prepared),"baseline":method,"mock":False,"mode":"spec"})
    write_json(output/"config.json",config)
    train,val=load_dataset(prepared/"train"),load_dataset(prepared/"validation")
    rng=np.random.default_rng(config["seed"])
    budget=2*config["agent"]["k"]+config["agent"]["iterations"]
    specs=[random_spec(rng,config["bank"]) for _ in range(budget)] if method=="random_search" else [{"model":method}]
    trials=[]; best=None
    for i,spec in enumerate(specs):
        candidate={"model":spec["model"],"pipeline":spec,"summary":method,"code":script_for_spec(spec)}
        result=execute(candidate,train,val,"spec",config["execution"],config["seed"])
        metrics=evaluate_predictions(val,result["predictions"],config["tolerance"]) if result["ok"] else None
        q=metrics["aggregate"]["f1"] if metrics else None
        accepted=q is not None and (best is None or q>best["q"])
        trial={"id":i,"operator":method,"model":spec["model"],"candidate":candidate,"q":q,
               "metrics":metrics["aggregate"] if metrics else None,"accepted":accepted,"stagnant":False,
               "parents":[],"log":result["log"],"attempts":[{"candidate":candidate,"execution":result}]}
        trials.append(trial)
        write_json(output/"trajectories"/f"{i:04d}.json",trial)
        if accepted: best=trial
    if best is None: raise RuntimeError("Every baseline trial failed")
    write_json(output/"best.json",{"id":best["id"],"validation_f1":best["q"],"candidate":best["candidate"],
                                  "candidate_sha256":digest(best["candidate"]),"seed":config["seed"],"mode":"spec","mock":False})
    (output/"best.py").write_text(best["candidate"]["code"])
    write_json(output/"search_complete.json",{"complete":True,"best_trial":best["id"],"validation_f1":best["q"],
              "trials":len(trials),"valid_trials":sum(t["q"] is not None for t in trials),
              "trial_success_rate":sum(t["q"] is not None for t in trials)/len(trials),"mock":False})


def matrix(prepared,output,config,seeds,variants,models=None,include_baselines=False):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Matrix output already exists; choose a new path")
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds): raise ValueError("Seeds must be distinct nonnegative integers")
    if any(v not in VARIANTS for v in variants): raise ValueError("Unknown ablation variant")
    models=models or [os.environ.get("OPENROUTER_MODEL") or config["llm"]["model"]]
    if os.environ.get("OPENROUTER_MODEL") and models!=[os.environ["OPENROUTER_MODEL"]]:
        raise ValueError("Unset OPENROUTER_MODEL when supplying a multi-model matrix")
    output.mkdir(parents=True)
    records=[]
    for model_index,model in enumerate(models):
        for variant in variants:
            for seed in seeds:
                cfg=copy.deepcopy(config); cfg["seed"]=seed; cfg["llm"]["model"]=model
                cfg["agent"].update(VARIANTS[variant])
                folder=output/f"m{model_index}-{variant}-seed{seed}"
                record={"model":model,"variant":variant,"seed":seed,"path":str(folder),"search_ok":False}
                try: run(prepared,folder,cfg); record["search_ok"]=True
                except Exception as error: record["error"]=str(error)[:2000]
                records.append(record); write_json(output/"matrix.json",records)
    if include_baselines:
        for method in ["random_search"]+config["bank"]:
            for seed in seeds:
                cfg=copy.deepcopy(config); cfg["seed"]=seed
                folder=output/f"baseline-{method}-seed{seed}"
                record={"model":"none","variant":method,"seed":seed,"path":str(folder),"search_ok":False}
                try: baseline(prepared,folder,cfg,method); record["search_ok"]=True
                except Exception as error: record["error"]=str(error)[:2000]
                records.append(record); write_json(output/"matrix.json",records)
    # All search configurations are frozen before any test evaluation.
    for record in records:
        record["test_ok"]=False
        if record["search_ok"]:
            try:
                result=evaluate_frozen(record["path"])
                record["test_ok"]=result["execution"]["ok"]
                record["test"]=result["scores"]["aggregate"] if result["scores"] else None
            except Exception as error: record["error"]=str(error)[:2000]
        write_json(output/"matrix.json",records)
    summarize_matrix(output)
    return records


def summarize_matrix(output):
    output=Path(output); records=read_json(output/"matrix.json"); groups={}
    for r in records: groups.setdefault((r["model"],r["variant"]),[]).append(r)
    rows=[]
    for (model,variant),group in groups.items():
        valid=[r for r in group if r.get("test_ok")]
        row={"model":model,"variant":variant,"runs":len(group),"successful_runs":len(valid),
             "run_success_rate":len(valid)/len(group)}
        for metric in ["f1","precision","recall","hausdorff"]:
            values=[r["test"][metric] for r in valid if r["test"][metric] is not None]
            # Do not hide infinite Hausdorff cases by averaging only finite runs.
            if metric=="hausdorff" and len(values)!=len(valid): values=[]
            row[f"{metric}_mean"]=float(np.mean(values)) if values else None
            row[f"{metric}_std"]=float(np.std(values,ddof=1)) if len(values)>1 else None
        row["failure_adjusted_f1"]=sum(r["test"]["f1"] for r in valid)/len(group)
        rows.append(row)
    write_json(output/"summary.json",rows)
    if rows:
        with (output/"summary.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def paired_comparison(matrix_path,variant_a,variant_b,model,seed=42):
    records=read_json(matrix_path)
    def selected(variant):
        return {r["seed"]:r["test"]["f1"] for r in records if r["model"]==model and r["variant"]==variant and r.get("test_ok")}
    a,b=selected(variant_a),selected(variant_b)
    paired=sorted(a.keys() & b.keys())
    if len(paired)<2: raise ValueError("Need at least two paired successful seeds")
    deltas=np.array([a[s]-b[s] for s in paired])
    rng=np.random.default_rng(seed)
    means=rng.choice(deltas,(10000,len(deltas)),replace=True).mean(1)
    return {"a":variant_a,"b":variant_b,"model":model,"paired_seeds":paired,"mean_f1_difference":float(deltas.mean()),
            "bootstrap_95_percent_interval":np.quantile(means,[0.025,0.975]).tolist(),
            "note":"Exploratory paired bootstrap over run seeds; few seeds give weak uncertainty estimates. Failed runs excluded; inspect run success separately."}
