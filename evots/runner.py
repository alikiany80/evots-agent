from __future__ import annotations

import copy
import difflib
import importlib.metadata
import math
import os
import platform
from pathlib import Path

from .bank import available_bank
from .data import load_dataset
from .eda import profile, plot_unlabeled
from .execution import execute, evaluate_predictions, check_docker
from .io import digest, file_hash, read_json, write_json, source_hash
from .openrouter import OpenRouter, AuthenticationError, BudgetExceeded
from .proposer import Proposer, MockProposer


def incumbent(trajectories):
    valid=[t for t in trajectories if t.get("q") is not None]
    return max(valid,key=lambda t:t["q"]) if valid else None


def decision(score, previous, operator, epsilon):
    valid=score is not None and math.isfinite(score) and 0<=score<=1
    accept=valid and (previous is None or score>previous)
    stagnant=operator=="revision" and (not valid or (previous is not None and score-previous<=epsilon))
    return accept,stagnant


def choose_operator(trajectories,iteration,config):
    best=incumbent(trajectories)
    if best is None: raise RuntimeError("All warm-up experiments failed; inspect trajectories and configuration")
    a=config["agent"]
    if iteration==a["iterations"]-1 and a["recombination"]:
        ranked=sorted([t for t in trajectories if t["q"] is not None],key=lambda t:t["q"],reverse=True)
        parents=[best]
        for t in ranked:
            if t["id"]!=best["id"] and digest(t["candidate"])!=digest(best["candidate"]):
                parents.append(t); break
        return "recombine",parents,None
    consumed={t.get("consumes") for t in trajectories if t.get("consumes") is not None}
    stagnant=next((t for t in reversed(trajectories)
                   if t["stagnant"] and t["model"]==best["model"] and t["id"] not in consumed),None)
    if a["alternative"] and stagnant:
        parents=[best] if best["id"]==stagnant["id"] else [best,stagnant]
        return "alternative",parents,stagnant["id"]
    return "revision",[best],None


def evidence(t):
    # Only scalar validation feedback reaches planning; never ground-truth coordinates.
    return {k:t[k] for k in ("id","operator","model","candidate","q","metrics","stagnant","accepted","log")}


def scientific_config(config):
    c=copy.deepcopy(config)
    for k in ("max_requests","max_cost_usd","timeout_seconds","retries","max_retry_wait_seconds"):
        c["llm"].pop(k,None)
    return c


def run(prepared, output, config, resume=False, proposer=None):
    prepared,output=Path(prepared).resolve(),Path(output).resolve()
    config=copy.deepcopy(config)
    if not config["mock"]: config["llm"]["model"]=os.environ.get("OPENROUTER_MODEL") or config["llm"]["model"]
    # Reading a test manifest checksum is not reading its annotations or values.
    fingerprints={s:file_hash(prepared/s/"dataset.json") for s in ("train","validation","test")}
    identity={"config":scientific_config(config),"inputs":fingerprints,
              "protocol":file_hash(prepared/"protocol.json"),"source_sha256":source_hash()}
    if output.exists() and not resume: raise FileExistsError("Run directory exists. Use --resume or a new output directory")
    if resume:
        old=read_json(output/"manifest.json")
        if old["identity"]!=identity: raise ValueError("Resume refused: scientific config, source or input fingerprint changed")
        if (output/"search_complete.json").exists(): return read_json(output/"search_complete.json")
    else:
        output.mkdir(parents=True)
        versions={}
        for name in ("numpy","scipy","ruptures","httpx","changeforest","torch"):
            try: versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: pass
        write_json(output/"manifest.json",{"identity":identity,"prepared":str(prepared),"python":platform.python_version(),
                  "platform":platform.platform(),"versions":versions,"implementation":"independent-research-v0.1",
                  "mock":config["mock"],"mode":config["mode"]})
    write_json(output/"config.json",config)
    execution_config=copy.deepcopy(config["execution"])
    if config["mode"]=="python":
        meta=read_json(output/"manifest.json")
        image_id=check_docker(meta.get("docker_image_id",execution_config["image"]))
        meta["docker_image_id"]=image_id
        write_json(output/"manifest.json",meta)
        execution_config["image"]=image_id
    else:
        status=available_bank()
        missing=[m for m in config["bank"] if not status[m]["available"]]
        if missing: raise RuntimeError(f"Missing optional detector dependencies: {missing}; install .[all] or adjust bank")
    train=load_dataset(prepared/"train")
    validation=load_dataset(prepared/"validation")
    if [s.name for s in train]!=[s.name for s in validation]: raise ValueError("Mismatched train/validation names")
    if proposer is None:
        proposer=MockProposer(config) if config["mock"] else Proposer(OpenRouter(config["llm"],output/"llm"),config)
    if (output/"eda.json").exists(): eda=read_json(output/"eda.json")
    else:
        eda=profile(train,config["seed"]) if config["agent"]["eda"] else {"disabled":True}
        write_json(output/"eda.json",eda)
        if config["agent"]["eda"]: plot_unlabeled(train[eda["sample_index"]].x,output/"eda.png")
    if (output/"selection.json").exists(): selection=read_json(output/"selection.json")
    else:
        selection=proposer.select(eda,output/"eda.png" if config["agent"]["vision"] else None)
        write_json(output/"selection.json",selection)
    trial_dir=output/"trajectories"; trial_dir.mkdir(exist_ok=True)
    trajectories=[read_json(p) for p in sorted(trial_dir.glob("*.json"))]
    k=config["agent"]["k"]; total=2*k+config["agent"]["iterations"]

    def trial(index,operator,parents,consumes,required_model=None):
        best=incumbent(trajectories)
        global_previous=best["q"] if best else None
        local_previous=parents[0]["q"] if parents and index<2*k else global_previous
        model=required_model or (best["model"] if best else selection["primary"][0])
        recent=[t for t in trajectories if t["model"]==model and t["q"] is not None][-4:]
        context={"operator":operator,"required_model":required_model,"selection":selection,"eda":eda,
                 "parents":[evidence(p) for p in parents],"recent_relevant":[evidence(t) for t in recent],
                 "attempted_families":sorted({t["model"] for t in trajectories}),
                 "trial":index,"total_trials":total,"primary_metric":"macro boundary F1 on validation",
                 "tolerance":config["tolerance"]}
        candidate=None; repair=None; attempts=[]; result=None; scores=None
        for repair_index in range(config["agent"]["repairs"]+1):
            attempt_count_before=len(attempts)
            try:
                candidate=proposer.propose(context,f"trial:{index}:repair:{repair_index}",repair)
                if required_model and candidate["model"]!=required_model:
                    raise ValueError(f"Warm-up must use required family {required_model}")
                result=execute(candidate,train,validation,config["mode"],execution_config,config["seed"])
                attempts.append({"candidate":candidate,"execution":result})
                if not result["ok"]: raise ValueError(result["log"])
                scores=evaluate_predictions(validation,result["predictions"],config["tolerance"])
                break
            except (BudgetExceeded,AuthenticationError): raise
            except (ValueError,SyntaxError,RuntimeError,KeyError,TypeError) as error:
                repair={"previous_candidate":candidate,"error":str(error)[-8000:]}
                if len(attempts)==attempt_count_before:
                    attempts.append({"candidate":candidate,"error":str(error)[-8000:]})
        q=scores["aggregate"]["f1"] if scores else None
        accepted,_=decision(q,global_previous,operator,config["agent"]["epsilon"])
        _,stagnant=decision(q,local_previous,operator,config["agent"]["epsilon"])
        before=parents[0]["candidate"]["code"] if parents and parents[0].get("candidate") else ""
        after=candidate["code"] if candidate else ""
        diff="".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="parent.py",tofile="candidate.py"))
        item={"id":index,"stage":"warmup" if index<2*k else "optimization","operator":operator,
              "parents":[p["id"] for p in parents],"consumes":consumes,"model":candidate["model"] if candidate else model,
              "candidate":candidate,"q":q,"metrics":scores["aggregate"] if scores else None,
              "accepted":accepted,"stagnant":stagnant,"code_diff":diff,
              "log":result["log"] if result else (repair or {}).get("error",""),"attempts":attempts,
              "validation_details":scores,"context":context}
        write_json(trial_dir/f"{index:04d}.json",item)
        if candidate: (trial_dir/f"{index:04d}.py").write_text(after,encoding="utf-8")
        trajectories.append(item)
        current=incumbent(trajectories)
        if current: write_json(output/"incumbent.json",{"id":current["id"],"q":current["q"],"candidate":current["candidate"]})
        print(f"[{index+1}/{total}] {operator} {item['model']} validation_f1={q} accepted={accepted}",flush=True)

    try:
        for index in range(len(trajectories),total):
            if index<2*k:
                family=selection["primary"][index//2]
                parents=[] if index%2==0 else [trajectories[index-1]]
                trial(index,"initial" if index%2==0 else "revision",parents,None,family)
            else:
                operator,parents,consumes=choose_operator(trajectories,index-2*k,config)
                trial(index,operator,parents,consumes)
    except BaseException as error:
        write_json(output/"status.json",{"complete":False,"completed_trials":len(trajectories),"error":str(error)[:2000]})
        raise
    best=incumbent(trajectories)
    if best is None: raise RuntimeError("No successful candidate")
    frozen={"id":best["id"],"validation_f1":best["q"],"candidate":best["candidate"],"seed":config["seed"],
            "candidate_sha256":digest(best["candidate"]),"mode":config["mode"],"mock":config["mock"]}
    write_json(output/"best.json",frozen)
    (output/"best.py").write_text(best["candidate"]["code"],encoding="utf-8")
    summary={"complete":True,"best_trial":best["id"],"validation_f1":best["q"],"trials":len(trajectories),
             "valid_trials":sum(t["q"] is not None for t in trajectories),
             "trial_success_rate":sum(t["q"] is not None for t in trajectories)/len(trajectories),
             "first_attempt_success_rate":sum(bool(t["attempts"] and t["attempts"][0].get("execution",{}).get("ok")) for t in trajectories)/len(trajectories),
             "mock":config["mock"]}
    write_json(output/"search_complete.json",summary)
    write_json(output/"status.json",summary)
    return summary


def evaluate_frozen(output,prepared_override=None):
    output=Path(output).resolve()
    if not (output/"search_complete.json").exists(): raise ValueError("Search is not complete; test evaluation is unavailable")
    frozen=read_json(output/"best.json"); manifest=read_json(output/"manifest.json"); config=read_json(output/"config.json")
    if digest(frozen["candidate"])!=frozen["candidate_sha256"]: raise ValueError("Frozen candidate changed")
    if source_hash()!=manifest["identity"]["source_sha256"]: raise ValueError("Source changed after search; create a new experiment")
    prepared=Path(prepared_override).resolve() if prepared_override else Path(manifest["prepared"])
    for split in ("train","validation","test"):
        if file_hash(prepared/split/"dataset.json")!=manifest["identity"]["inputs"][split]:
            raise ValueError("Prepared dataset changed after search")
    report_path=output/"test_results.json"
    if report_path.exists(): return read_json(report_path)
    train,test=load_dataset(prepared/"train"),load_dataset(prepared/"test")
    execution_config=copy.deepcopy(config["execution"])
    if frozen["mode"]=="python":
        execution_config["image"]=check_docker(manifest["docker_image_id"])
    result=execute(frozen["candidate"],train,test,frozen["mode"],execution_config,frozen["seed"])
    report={"candidate_sha256":frozen["candidate_sha256"],"mock":frozen["mock"],"execution":result,
            "scores":evaluate_predictions(test,result["predictions"],config["tolerance"]) if result["ok"] else None}
    write_json(report_path,report)
    return report
