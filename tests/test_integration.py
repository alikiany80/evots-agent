import json
import os
from pathlib import Path
import httpx
import numpy as np
import pytest

from evots.config import config_from
from evots.data import synthetic, save_dataset, prepare, Series
from evots.openrouter import OpenRouter
from evots.proposer import Proposer
from evots.runner import run, evaluate_frozen
from evots.experiments import matrix, paired_comparison
from evots.datasets_extra import stitch_adia
from evots.io import write_json, read_json


def test_openrouter_planner_end_to_end_with_mock_http(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY","TEST_SECRET")
    monkeypatch.delenv("OPENROUTER_MODEL",raising=False)
    save_dataset(tmp_path/"data",synthetic(n=500,count=1,segment_length=40))
    prepare(tmp_path/"data",tmp_path/"prepared")
    cfg=config_from(overrides={"bank":["pelt","window"],"agent":{"k":1,"iterations":3,"repairs":0}})
    requests=[]
    def handle(request):
        payload=json.loads(request.content); requests.append(payload)
        prompt=payload["messages"][-1]["content"]
        if isinstance(prompt,list):
            answer={"primary":["pelt"],"alternatives":["window"],"summary":"fixture"}
        else:
            context=json.loads(prompt)["context"]
            answer={"model":"pelt","summary":"fixture","pipeline":{"model":"pelt","penalty":8}}
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(answer)},"finish_reason":"stop"}],
                                       "usage":{"cost":0,"prompt_tokens":10,"completion_tokens":10}})
    client=OpenRouter(cfg["llm"],tmp_path/"api",httpx.Client(transport=httpx.MockTransport(handle)))
    result=run(tmp_path/"prepared",tmp_path/"run",cfg,proposer=Proposer(client,cfg))
    assert result["trials"]==5 and len(requests)==6
    assert evaluate_frozen(tmp_path/"run")["execution"]["ok"]
    assert "TEST_SECRET" not in json.dumps(requests)


def test_matrix_and_pairing_with_mock_planner(tmp_path):
    save_dataset(tmp_path/"data",synthetic(n=400,count=1,segment_length=40))
    prepare(tmp_path/"data",tmp_path/"prepared")
    cfg=config_from(overrides={"mock":True,"bank":["pelt"],"agent":{"k":1,"iterations":1,"repairs":0}})
    records=matrix(tmp_path/"prepared",tmp_path/"matrix",cfg,[11,22],["full","no_recombination"])
    assert len(records)==4 and all(r["test_ok"] for r in records)
    assert (tmp_path/"matrix"/"summary.csv").exists()
    stats=paired_comparison(tmp_path/"matrix"/"matrix.json","full","no_recombination",cfg["llm"]["model"])
    assert stats["paired_seeds"]==[11,22]


def test_adia_stitching_records_artificial_seams(tmp_path):
    x=np.r_[np.zeros(30),np.ones(30)*2]
    np.savetxt(tmp_path/"one.csv",x,delimiter=",",header="value",comments="")
    np.savetxt(tmp_path/"two.csv",x,delimiter=",",header="value",comments="")
    write_json(tmp_path/"input.json",[{"file":"one.csv","value_column":"value","break_index":30},
                                    {"file":"two.csv","value_column":"value","break_index":30}])
    stitch_adia(tmp_path/"input.json",tmp_path/"out",half_window=20)
    metadata=read_json(tmp_path/"out"/"dataset.json")["metadata"]
    assert metadata["streams"][0]["annotated_breaks"]==[20,60]
    assert metadata["streams"][0]["artificial_join_boundaries"]==[40]


def test_python_mode_requires_real_sandbox(monkeypatch):
    from evots.execution import check_docker
    import shutil
    monkeypatch.setattr(shutil,"which",lambda _:None)
    with pytest.raises(RuntimeError,match="Docker"): check_docker("image")
