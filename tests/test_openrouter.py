import json
import httpx
import pytest
from evots.config import config_from
from evots.openrouter import OpenRouter, AuthenticationError, BudgetExceeded


def response(content='{"ok":true}',finish="stop"):
    return {"id":"fake-id","model":"test/model","choices":[{"message":{"content":content},"finish_reason":finish}],
            "usage":{"prompt_tokens":12,"completion_tokens":6,"cost":.002}}


def client(tmp_path,monkeypatch,handler,**overrides):
    monkeypatch.setenv("OPENROUTER_API_KEY","secret-test-value")
    monkeypatch.delenv("OPENROUTER_MODEL",raising=False)
    cfg=config_from()["llm"]; cfg.update(overrides)
    waits=[]
    obj=OpenRouter(cfg,tmp_path,httpx.Client(transport=httpx.MockTransport(handler)),waits.append)
    return obj,waits


def test_retry_cache_and_usage(tmp_path,monkeypatch):
    calls=[]
    def handle(req):
        calls.append(req)
        if len(calls)==1: return httpx.Response(429,json={"error":{"code":429,"message":"slow"}},headers={"Retry-After":"2"})
        return httpx.Response(200,json=response())
    obj,waits=client(tmp_path,monkeypatch,handle)
    messages=[{"role":"user","content":"test"}]
    assert obj.request(messages,"one",42)=={"ok":True}
    assert obj.request(messages,"one",42)=={"ok":True}
    assert len(calls)==2 and waits==[2]
    assert obj.ledger["attempts"]==2 and obj.ledger["reported_cost_usd"]==.002
    payload=json.loads(calls[1].content)
    assert payload["provider"]["require_parameters"] is True
    assert payload["provider"]["allow_fallbacks"] is False
    assert all("secret-test-value" not in p.read_text() for p in tmp_path.glob("*.json"))


def test_auth_does_not_retry(tmp_path,monkeypatch):
    obj,waits=client(tmp_path,monkeypatch,lambda r:httpx.Response(401,json={"error":{"code":401,"message":"bad"}}))
    with pytest.raises(AuthenticationError): obj.request([],"one",1)
    assert obj.ledger["attempts"]==1 and not waits


def test_error_inside_http_200_is_not_success(tmp_path,monkeypatch):
    obj,_=client(tmp_path,monkeypatch,lambda r:httpx.Response(200,json={"error":{"code":503,"message":"down"}}),retries=0)
    with pytest.raises(RuntimeError): obj.request([],"one",1)
    assert not list(tmp_path.glob("*.response.json"))


def test_truncation_is_accounted_and_cached(tmp_path,monkeypatch):
    obj,_=client(tmp_path,monkeypatch,lambda r:httpx.Response(200,json=response('{"unfinished":',"length")))
    with pytest.raises(ValueError): obj.request([],"one",1)
    with pytest.raises(ValueError): obj.request([],"one",1)
    assert obj.ledger["attempts"]==1 and obj.ledger["reported_cost_usd"]==.002


def test_request_budget_before_network(tmp_path,monkeypatch):
    obj,_=client(tmp_path,monkeypatch,lambda r:httpx.Response(200,json=response()),max_requests=1)
    obj.request([],"one",1)
    with pytest.raises(BudgetExceeded): obj.request([],"two",1)
