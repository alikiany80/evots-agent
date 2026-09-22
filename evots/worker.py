"""Worker contract: only train arrays and unlabeled evaluation arrays enter."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

from .bank import detect as bank_detect
from .metrics import boundaries


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=["spec","python"],required=True)
    p.add_argument("--seed",type=int,required=True)
    args=p.parse_args(); inp=Path(args.input)
    if args.mode=="spec":
        spec=json.loads((inp/"candidate.json").read_text())
        predict=lambda train,x,seed: bank_detect(train,x,spec,seed)
    else:
        module_spec=importlib.util.spec_from_file_location("candidate",inp/"candidate.py")
        module=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        predict=module.detect
    outputs=[]
    with np.load(inp/"arrays.npz",allow_pickle=False) as data:
        count=int(data["count"])
        for i in range(count):
            train,x=data[f"train_{i}"].copy(),data[f"x_{i}"].copy()
            points=boundaries(predict(train,x,args.seed+i),len(x))
            outputs.append(points)
    Path(args.output).write_text(json.dumps(outputs,allow_nan=False))


if __name__=="__main__": main()
