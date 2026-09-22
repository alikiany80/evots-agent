import importlib.util
import numpy as np
import pytest
from evots.bank import detect, validate_spec


@pytest.mark.parametrize("model",["pelt","bottom_up","window","spectral","bayesian_offline","changeforest_rf","changeforest_knn"])
def test_detector_contract_and_simple_mean_shift(model):
    if model.startswith("changeforest") and importlib.util.find_spec("changeforest") is None: pytest.skip("optional changeforest")
    rng=np.random.default_rng(0)
    x=np.r_[rng.normal(0,.2,60),rng.normal(4,.2,60),rng.normal(-3,.2,60)][:,None]
    pred=detect(x,x,{"model":model,"window":20},42)
    assert pred==sorted(set(pred)) and all(0<p<len(x) for p in pred)
    if model in {"pelt","bottom_up","window","bayesian_offline"}:
        assert any(abs(p-60)<=5 for p in pred) and any(abs(p-120)<=5 for p in pred)


def test_klcpd_executes_and_repeats_on_cpu():
    if importlib.util.find_spec("torch") is None: pytest.skip("optional torch")
    rng=np.random.default_rng(4)
    x=rng.normal(size=(70,2))
    spec={"model":"klcpd","window":5,"epochs":1,"critic_steps":1,"hidden":4,"batch_size":16}
    assert detect(x,x,spec,11)==detect(x,x,spec,11)


@pytest.mark.parametrize("spec",[{"model":"shell"},{"model":"pelt","code":"evil"},{"model":"pelt","jump":0},{"model":"pelt","penalty":float('nan')}])
def test_spec_rejects_unbounded_or_unknown_parameters(spec):
    with pytest.raises(ValueError): validate_spec(spec)
