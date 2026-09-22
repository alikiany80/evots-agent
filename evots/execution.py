from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
import numpy as np

from .io import read_json, write_json
from .metrics import boundaries, score_boundaries, aggregate


def check_docker(image):
    if shutil.which("docker") is None:
        raise RuntimeError("Python-generation mode needs Docker. Install Docker and build the supplied Dockerfile, or use mode=spec.")
    proc=subprocess.run(["docker","image","inspect",image],capture_output=True,text=True,timeout=20)
    if proc.returncode:
        raise RuntimeError(f"Docker image {image!r} unavailable. Run: docker build -t {image} .")
    details=json.loads(proc.stdout)
    if not details or not details[0].get("Id"):
        raise RuntimeError("Docker returned no immutable image ID")
    return details[0]["Id"]


def execute(candidate, train, evaluation, mode, config, seed):
    if len(train)!=len(evaluation) or [s.name for s in train]!=[s.name for s in evaluation]:
        raise ValueError("Train/evaluation series do not align")
    started=time.monotonic()
    name="evots-"+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="evots-") as tmp:
        root=Path(tmp); inp=root/"input"; out=root/"output"
        inp.mkdir(); out.mkdir()
        arrays={"count":len(train)}
        for i,(tr,ev) in enumerate(zip(train,evaluation)):
            arrays[f"train_{i}"]=tr.x; arrays[f"x_{i}"]=ev.x
        np.savez_compressed(inp/"arrays.npz",**arrays)
        if mode=="spec": write_json(inp/"candidate.json",candidate["pipeline"])
        else: (inp/"candidate.py").write_text(candidate["code"],encoding="utf-8")
        result_file=out/"predictions.json"
        base=["-m","evots.worker","--mode",mode,"--seed",str(seed)]
        if mode=="spec":
            cmd=[sys.executable,*base,"--input",str(inp),"--output",str(result_file)]
            # Trusted detector code only. Never exec LLM Python in this process.
            env={k:v for k,v in os.environ.items() if k in {"PATH","SYSTEMROOT","WINDIR","PYTHONPATH"}}
            package_root=str(Path(__file__).resolve().parent.parent)
            paths=[str(Path(p or '.').resolve()) for p in env.get("PYTHONPATH","").split(os.pathsep)]
            env["PYTHONPATH"]=os.pathsep.join([package_root,*paths])
            env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        else:
            worker_uid=os.getuid() if hasattr(os,"getuid") and os.getuid()!=0 else 65534
            worker_gid=os.getgid() if hasattr(os,"getgid") and os.getgid()!=0 else 65534
            out.chmod(0o777)
            cmd=["docker","run","--rm","--name",name,"--network=none","--read-only",
                 "--cap-drop=ALL","--security-opt=no-new-privileges","--pids-limit=128",
                 "--memory",config["memory"],"--cpus",str(config["cpus"]),
                 "--tmpfs","/tmp:rw,noexec,nosuid,size=256m",
                 "--user",f"{worker_uid}:{worker_gid}",
                 "--mount",f"type=bind,src={inp},dst=/input,readonly",
                 "--mount",f"type=bind,src={out},dst=/output",
                 config["image"],"python",*base,"--input","/input","--output","/output/predictions.json"]
            env=None
        try:
            # Drain continuously but retain only a bounded tail, including for
            # noisy generated code. No unlimited host log file is created.
            tail=bytearray()
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            def drain():
                while True:
                    chunk=proc.stdout.read(4096)
                    if not chunk: break
                    tail.extend(chunk)
                    if len(tail)>16000: del tail[:-16000]
            reader=threading.Thread(target=drain,daemon=True); reader.start()
            try: proc.wait(timeout=config["timeout_seconds"])
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); raise
            finally:
                reader.join(timeout=1)
            text=bytes(tail).decode(errors="replace")
            if proc.returncode: raise RuntimeError(f"Worker exited {proc.returncode}: {text}")
            if not result_file.exists() or result_file.stat().st_size>2_000_000:
                raise ValueError("Worker result missing or larger than 2 MB")
            predictions=read_json(result_file)
            if not isinstance(predictions,list) or len(predictions)!=len(evaluation):
                raise ValueError("Worker returned wrong number of predictions")
            predictions=[boundaries(p,len(s.x)) for p,s in zip(predictions,evaluation)]
            return {"ok":True,"predictions":predictions,"log":text[-8000:],"seconds":time.monotonic()-started}
        except (subprocess.TimeoutExpired,RuntimeError,ValueError,OSError) as error:
            return {"ok":False,"predictions":None,"log":str(error)[-8000:],"seconds":time.monotonic()-started}
        finally:
            if mode=="python":
                try:
                    subprocess.run(["docker","rm","-f",name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
                except (OSError,subprocess.TimeoutExpired): pass


def evaluate_predictions(evaluation,predictions,tolerance):
    rows=[{"name":s.name,**score_boundaries(s.cps,p,len(s.x),tolerance)} for s,p in zip(evaluation,predictions)]
    return {"aggregate":aggregate(rows),"per_series":rows}
