"""OpenRouter REST transport with request cache, retries and durable accounting."""
from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import httpx

from .io import digest, read_json, write_json


class BudgetExceeded(RuntimeError): pass
class AuthenticationError(RuntimeError): pass


def parse_object(content):
    if not isinstance(content,str) or not content.strip():
        raise ValueError("Empty model response")
    text=content.strip()
    if text.startswith("```"):
        text=re.sub(r"^```(?:json)?\s*", "", text, count=1)
        text=re.sub(r"\s*```$", "", text, count=1)
    value=json.loads(text)
    if not isinstance(value,dict): raise ValueError("Response must be one JSON object")
    return value


def retry_delay(value, attempt):
    if value:
        try: return max(0,float(value))
        except ValueError:
            try: return max(0,(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except (ValueError,TypeError): pass
    return min(30,2**attempt+random.random())


class OpenRouter:
    def __init__(self, config, directory, client=None, sleep=time.sleep):
        self.config=config
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.model=os.environ.get("OPENROUTER_MODEL") or config["model"]
        self.key=os.environ.get("OPENROUTER_API_KEY","")
        if not self.key: raise AuthenticationError("Set OPENROUTER_API_KEY locally; do not put it in configs or prompts")
        self.http=client or httpx.Client(timeout=config["timeout_seconds"],follow_redirects=False)
        self.sleep=sleep
        self.ledger_path=self.directory/"usage.json"
        self.ledger=read_json(self.ledger_path) if self.ledger_path.exists() else {"attempts":0,"reported_cost_usd":0.0,"unknown_cost_responses":0,"prompt_tokens":0,"completion_tokens":0}

    def clean(self,text): return str(text).replace(self.key,"[REDACTED]")

    def request(self,messages,scope,seed):
        cfg=self.config
        payload={"model":self.model,"messages":messages,"stream":False,"max_tokens":cfg["max_tokens"],
                 "provider":{**cfg["provider"],"require_parameters":True}}
        if cfg.get("temperature") is not None: payload["temperature"]=cfg["temperature"]
        if cfg.get("send_seed",False): payload["seed"]=seed
        if cfg.get("reasoning") is not None: payload["reasoning"]=cfg["reasoning"]
        if cfg["json_mode"]: payload["response_format"]={"type":"json_object"}
        key=digest({"request":payload,"scope":scope})
        response_file=self.directory/f"{key}.response.json"
        if response_file.exists():
            data=read_json(response_file)
            return self._content(data)
        write_json(self.directory/f"{key}.request.json",payload)
        for attempt in range(cfg["retries"]+1):
            if self.ledger["attempts"]>=cfg["max_requests"]: raise BudgetExceeded("OpenRouter request-attempt budget exhausted")
            limit=cfg.get("max_cost_usd")
            if limit is not None and (self.ledger["reported_cost_usd"]>=limit or self.ledger["unknown_cost_responses"]):
                raise BudgetExceeded("Reported-cost limit reached, or a response had unknown cost; check usage.json")
            self.ledger["attempts"]+=1
            write_json(self.ledger_path,self.ledger)
            try:
                r=self.http.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers={"Authorization":f"Bearer {self.key}","X-OpenRouter-Title":"EvoTS Research"},json=payload)
            except httpx.TransportError as e:
                # The server may still bill a timed-out request. Never call this free.
                self.ledger["unknown_cost_responses"]+=1; write_json(self.ledger_path,self.ledger)
                if attempt==cfg["retries"]: raise RuntimeError(self.clean(e)) from None
                self.sleep(retry_delay(None,attempt)); continue
            try: data=r.json()
            except ValueError: data={"error":{"code":r.status_code,"message":"Non-JSON response"}}
            if not isinstance(data,dict): raise RuntimeError("OpenRouter returned a non-object response")
            status=r.status_code
            if data.get("error"):
                try: status=int(data["error"].get("code",status))
                except (ValueError,TypeError): status=500
                if status<400: status=500
            if status>=400:
                message=self.clean(data.get("error",{}).get("message",f"HTTP {status}"))[:2000]
                write_json(self.directory/f"{key}.error-{self.ledger['attempts']}.json",{"status":status,"message":message})
                if status in (401,402,403): raise AuthenticationError(f"OpenRouter {status}: {message}")
                if status not in {408,429,500,502,503,504} or attempt==cfg["retries"]:
                    raise RuntimeError(f"OpenRouter {status}: {message}")
                delay=retry_delay(r.headers.get("retry-after"),attempt)
                if delay>cfg["max_retry_wait_seconds"]:
                    raise RuntimeError(f"Retry-After={delay:.0f}s exceeds configured wait; resume later")
                self.sleep(delay); continue
            usage=data.get("usage") or {}
            cost=usage.get("cost")
            if isinstance(cost,(float,int)) and cost>=0: self.ledger["reported_cost_usd"]+=cost
            else: self.ledger["unknown_cost_responses"]+=1
            for k in ("prompt_tokens","completion_tokens"):
                self.ledger[k]+=int(usage.get(k) or 0)
            write_json(self.ledger_path,self.ledger)
            # Persist even an unusable completion so resume does not silently resample.
            write_json(response_file,json.loads(self.clean(json.dumps(data))))
            return self._content(data)
        raise RuntimeError("Retry loop exhausted")

    def _content(self,data):
        choices=data.get("choices") or []
        if not choices: raise ValueError("OpenRouter response has no choices")
        choice=choices[0]
        if choice.get("finish_reason") in {"length","content_filter","error"}:
            raise ValueError(f"Unusable completion: finish_reason={choice.get('finish_reason')}")
        return parse_object(choice.get("message",{}).get("content"))
