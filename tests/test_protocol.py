import numpy as np
import pytest
from evots.data import Series, chronological_split, save_dataset, prepare
from evots.bank import transform, DEFAULTS
from evots.runner import decision, choose_operator
from evots.config import config_from


def test_split_edges_removed_and_labels_rebased():
    result=chronological_split([Series("one",np.arange(100),[20,60,70,80,90])])
    assert [len(result[s][0].x) for s in result]==[60,20,20]
    assert result["train"][0].cps==[]
    assert result["validation"][0].cps==[10]
    assert result["test"][0].cps==[10]


def test_transform_uses_training_statistics_only():
    train=np.arange(100,dtype=float)[:,None]
    x=np.array([[50.],[np.nan],[1e8]])
    _,one=transform(train,x,DEFAULTS)
    _,two=transform(train,x[:2],DEFAULTS)
    np.testing.assert_allclose(one[:2],two)
    assert abs(one[1,0])<1e-8


def test_small_positive_gain_accepted_but_stagnant():
    assert decision(.501,.5,"revision",.005)==(True,True)
    assert decision(.5,.5,"revision",.005)==(False,True)
    assert decision(None,.5,"revision",.005)==(False,True)
    assert decision(.6,.5,"alternative",.005)==(True,False)


def test_alternative_consumes_stagnation_once_and_final_overrides():
    cfg=config_from(overrides={"agent":{"iterations":4}})
    t={"id":0,"q":.5,"model":"pelt","stagnant":True,"candidate":{"code":"a"},"consumes":None}
    op,parents,consumes=choose_operator([t],0,cfg)
    assert op=="alternative" and consumes==0
    next_t={**t,"id":1,"q":.4,"stagnant":False,"consumes":0}
    assert choose_operator([t,next_t],1,cfg)[0]=="revision"
    assert choose_operator([t],3,cfg)[0]=="recombine"


def test_search_never_loads_test_values_or_labels(tmp_path,monkeypatch):
    from evots import runner
    from evots.data import synthetic
    from evots.io import read_json
    save_dataset(tmp_path/"data",synthetic(n=500,count=1,segment_length=40))
    prepare(tmp_path/"data",tmp_path/"prepared")
    # Keep the manifest; remove every test data file. Search must still work.
    for p in (tmp_path/"prepared"/"test").glob("*.npz"): p.unlink()
    cfg=config_from(overrides={"mock":True,"bank":["pelt","bottom_up"],"agent":{"k":1,"iterations":2,"repairs":0}})
    summary=runner.run(tmp_path/"prepared",tmp_path/"run",cfg)
    assert summary["trials"]==4 and summary["complete"]
    assert runner.run(tmp_path/"prepared",tmp_path/"run",cfg,resume=True)==summary
    trials=[read_json(p) for p in (tmp_path/"run"/"trajectories").glob("*.json")]
    accepted=[t["q"] for t in sorted(trials,key=lambda t:t["id"]) if t["accepted"]]
    assert accepted==sorted(set(accepted))
    assert all("cps" not in str(t["context"]) for t in trials)
    with pytest.raises(FileNotFoundError): runner.evaluate_frozen(tmp_path/"run")
