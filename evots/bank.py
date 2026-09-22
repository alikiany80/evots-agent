"""Eight detector families, explicit implementations and bounded configurations."""
from __future__ import annotations

import importlib.util
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.special import gammaln, logsumexp

from .eda import fill_missing


BANK = {
    "pelt": "ruptures PELT: penalized offline segmentation, l2/rbf/normal costs",
    "bottom_up": "ruptures BottomUp: greedy merging of adjacent segments",
    "window": "ruptures Window: sliding discrepancy, l2/rbf/normal costs",
    "changeforest_rf": "changeforest package: random-forest distributional change detection",
    "changeforest_knn": "changeforest package: kNN distributional change detection",
    "bayesian_offline": "Independent exact offline Bayesian Gaussian segmentation with NIG prior and geometric durations",
    "spectral": "Independent adjacent-window log-power spectral discrepancy",
    "klcpd": "Modernized KL-CPD-style GRU generator/critic; exact multi-RBF MMD, see fidelity notes",
}

DEFAULTS = dict(cost="l2", penalty=8.0, min_size=10, jump=1, window=25,
                threshold=2.5, min_distance=10, scaling="standard", representation="raw",
                smooth=1, clip=0.0, hazard=0.02, posterior_threshold=0.5,
                n_estimators=100, alpha=0.05, epochs=30, batch_size=32, hidden=16,
                learning_rate=0.001, critic_steps=3, lambda_ae=0.001, lambda_real=0.1)


def available_bank():
    out = {}
    for name, desc in BANK.items():
        requirement = "changeforest" if name.startswith("changeforest") else "torch" if name == "klcpd" else "ruptures" if name in {"pelt","bottom_up","window"} else None
        out[name] = {"description": desc, "available": requirement is None or importlib.util.find_spec(requirement) is not None,
                     "dependency": requirement}
    return out


def validate_spec(spec, depth=0):
    if not isinstance(spec, dict):
        raise ValueError("pipeline must be an object")
    if spec.get("model") == "ensemble":
        if depth or set(spec)-{"model","members","votes","vote_tolerance"}:
            raise ValueError("No nested ensembles or unknown ensemble fields")
        members = spec.get("members", [])
        if not 2 <= len(members) <= 4:
            raise ValueError("An ensemble requires 2..4 members")
        for s in members:
            validate_spec(s, depth=1)
        for k, lo, hi in [("votes",1,len(members)),("vote_tolerance",0,100)]:
            v = spec.get(k, 2 if k == "votes" else 10)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"Invalid {k}")
        return spec
    if spec.get("model") not in BANK or set(spec)-({"model"}|set(DEFAULTS)):
        raise ValueError("Unknown detector or pipeline field")
    c = {**DEFAULTS, **spec}
    for k, choices in {"cost":["l2","rbf","normal"], "scaling":["none","standard","robust"],
                       "representation":["raw","difference","squared","raw_squared","rolling_std"]}.items():
        if c[k] not in choices:
            raise ValueError(f"Invalid {k}")
    limits = {"penalty":(1e-5,1e6), "min_size":(2,1000), "jump":(1,100), "window":(3,500),
              "threshold":(0,100), "min_distance":(1,1000), "smooth":(1,101), "clip":(0,100),
              "hazard":(1e-6,0.9), "posterior_threshold":(0.00001,1), "n_estimators":(10,1000),
              "alpha":(0.00001,0.5), "epochs":(1,2000), "batch_size":(2,256), "hidden":(2,256),
              "learning_rate":(1e-6,0.1), "critic_steps":(1,10), "lambda_ae":(0,100), "lambda_real":(0,100)}
    integers = {k for k,v in DEFAULTS.items() if type(v) is int}
    for k,(lo,hi) in limits.items():
        v = c[k]
        if type(v) not in (int,float) or not np.isfinite(v) or not lo <= v <= hi or (k in integers and type(v) is not int):
            raise ValueError(f"{k} must be {'integer' if k in integers else 'numeric'} in [{lo},{hi}]")
    return spec


def transform(train, x, c):
    train, median = fill_missing(train)
    x, _ = fill_missing(x, median)
    if c["scaling"] == "standard":
        center, scale = train.mean(0), train.std(0)
    elif c["scaling"] == "robust":
        center = np.median(train,axis=0)
        scale = np.quantile(train,0.75,axis=0)-np.quantile(train,0.25,axis=0)
    else:
        center, scale = 0, np.ones(train.shape[1])
    scale = np.maximum(scale,1e-8)
    def one(a):
        a = (a-center)/scale
        if c["clip"]:
            a = np.clip(a,-c["clip"],c["clip"])
        if c["smooth"] > 1:
            # Offline detector: centered smoothing is intentional; never online.
            a = uniform_filter1d(a, size=c["smooth"],axis=0,mode="nearest")
        rep = c["representation"]
        if rep == "difference":
            a = np.vstack([np.zeros(a.shape[1]),np.diff(a,axis=0)])
        elif rep == "squared": a = a*a
        elif rep == "raw_squared": a = np.column_stack([a,a*a])
        elif rep == "rolling_std":
            mean = uniform_filter1d(a,size=c["window"],axis=0,mode="nearest")
            a = np.sqrt(np.maximum(0,uniform_filter1d(a*a,size=c["window"],axis=0,mode="nearest")-mean**2))
        return np.ascontiguousarray(a,dtype=np.float64)
    return one(train), one(x)


def robust_peaks(scores, valid_positions, threshold, distance):
    vals = scores[valid_positions]
    median = np.median(vals)
    mad = np.median(np.abs(vals-median))
    cutoff = median + threshold*max(1.4826*mad,1e-12)
    peaks, _ = find_peaks(scores, height=cutoff, distance=distance)
    return [int(p) for p in peaks if p in valid_positions]


def spectral(x, c):
    n = len(x)
    w = min(c["window"],(n-1)//2)
    scores = np.zeros(n)
    taper = np.hanning(w)[:,None]
    for t in range(w,n-w+1):
        # Log power retains changes in amplitude as well as spectral shape.
        left = np.abs(np.fft.rfft(x[t-w:t]*taper,axis=0))**2/w
        right = np.abs(np.fft.rfft(x[t:t+w]*taper,axis=0))**2/w
        scores[t] = np.mean((np.log1p(left)-np.log1p(right))**2)
    return robust_peaks(scores,np.arange(w,n-w+1),c["threshold"],c["min_distance"])


def bayesian_offline(x,c):
    """Exact forward/backward sum over partitions, independent Gaussian features.

    NIG(mu=0,kappa=1,alpha=1,beta=1); duration=min_size+Geometric(h)-1.
    Right-censored last segment uses survival rather than a terminal boundary.
    Complexity O(n^2*d); no online run-length approximation.
    """
    n,d = x.shape
    m = c["min_size"]
    if n > 4000:
        raise ValueError("Bayesian offline O(n²) detector is limited to 4000 samples per split")
    cs = np.vstack([np.zeros(d),np.cumsum(x,axis=0)])
    css = np.vstack([np.zeros(d),np.cumsum(x*x,axis=0)])
    segments = np.full((n+1,n+1),-np.inf)
    log_survival, log_h = np.log1p(-c["hazard"]), np.log(c["hazard"])
    for s in range(n-m+1):
        ends = np.arange(s+m,n+1)
        lengths = ends-s
        l = lengths[:,None]
        sums = cs[ends]-cs[s]
        sumsq = css[ends]-css[s]
        means = sums/l
        k = 1+l
        a = 1+l/2
        b = 1+0.5*np.maximum(0,sumsq-sums*sums/l)+l*means**2/(2*k)
        marginal = (-l/2*np.log(2*np.pi)-0.5*np.log(k)+gammaln(a)-a*np.log(b)).sum(1)
        duration = (lengths-m)*log_survival+np.where(ends<n,log_h,0)
        segments[s,ends] = marginal+duration
    forward = np.full(n+1,-np.inf); forward[0]=0
    for t in range(m,n+1):
        forward[t]=logsumexp(forward[:t]+segments[:t,t])
    reverse = np.full(n+1,-np.inf); reverse[n]=0
    for s in range(n-m,-1,-1):
        reverse[s]=logsumexp(segments[s,s+1:]+reverse[s+1:])
    if not np.isfinite(forward[n]): return []
    posterior=np.exp(np.minimum(0,forward+reverse-forward[n]))
    posterior[:m]=0; posterior[n-m+1:]=0
    return find_peaks(posterior,height=c["posterior_threshold"],distance=c["min_distance"])[0].astype(int).tolist()


def vote(members, tolerance, votes):
    # A member gets at most one vote per cluster; use bounded diameter (no chaining).
    remaining=sorted((p,i) for i,ps in enumerate(members) for p in ps)
    result=[]
    while remaining:
        first=remaining[0][0]
        cluster=[v for v in remaining if v[0] <= first+2*tolerance]
        remaining=remaining[len(cluster):]
        by_member={}
        for p,i in cluster: by_member.setdefault(i,[]).append(p)
        if len(by_member)>=votes:
            result.append(int(round(np.median([np.median(ps) for ps in by_member.values()]))))
    return sorted(set(result))


def detect(train, x, spec, seed=42):
    validate_spec(spec)
    if spec["model"] == "ensemble":
        return vote([detect(train,x,s,seed) for s in spec["members"]],spec.get("vote_tolerance",10),spec.get("votes",2))
    c={**DEFAULTS,**spec}
    train,x=transform(train,x,c)
    model=c["model"]; n=len(x)
    if n < 2*c["min_size"]: return []
    if model in {"pelt","bottom_up","window"}:
        import ruptures as rpt
        opts=dict(model=c["cost"],min_size=c["min_size"],jump=c["jump"])
        if model=="pelt": alg=rpt.Pelt(**opts)
        elif model=="bottom_up": alg=rpt.BottomUp(**opts)
        else: alg=rpt.Window(width=min(2*c["window"],n//2),**opts)
        cps=alg.fit(x).predict(pen=c["penalty"])
    elif model.startswith("changeforest"):
        from changeforest import changeforest, Control
        control=Control(seed=seed, minimal_relative_segment_length=min(0.49,c["min_size"]/n),
                        random_forest_n_estimators=c["n_estimators"],random_forest_n_jobs=1,
                        model_selection_alpha=c["alpha"])
        result=changeforest(x,method="random_forest" if model.endswith("rf") else "knn",segmentation_type="bs",control=control)
        cps=result.split_points()
    elif model=="spectral": cps=spectral(x,c)
    elif model=="bayesian_offline": cps=bayesian_offline(x,c)
    else:
        from .klcpd import klcpd_scores
        scores,valid=klcpd_scores(train,x,c,seed)
        cps=robust_peaks(scores,valid,c["threshold"],c["min_distance"])
    return sorted({int(p) for p in cps if 0<int(p)<n})


def script_for_spec(spec):
    validate_spec(spec)
    return ("from evots.bank import detect as bank_detect\n"
            f"SPEC = {spec!r}\n\n"
            "def detect(train, x, seed):\n"
            "    return bank_detect(train, x, SPEC, seed)\n")
