from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
import numpy as np
from scipy.io import loadmat

from .data import Series, save_dataset
from .io import file_hash, read_json


KLCPD_COMMIT="b3a32ee4ce5a950cefa2319535d2b2f521e7176f"


def fetch_bee(output,cache):
    """Fetch six upstream Y/L files; do not concatenate independent recordings."""
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    series=[]; sources=[]
    for i in range(1,7):
        name=f"beedance-{i}.mat"
        url=f"https://raw.githubusercontent.com/OctoberChang/klcpd_code/{KLCPD_COMMIT}/data/beedance/{name}"
        file=cache/name
        with urlopen(url,timeout=60) as response: content=response.read(10_000_001)
        if len(content)>10_000_000: raise ValueError("Unexpected oversized dataset")
        file.write_bytes(content)
        data=loadmat(file); x=np.asarray(data["Y"]); labels=np.asarray(data["L"]).reshape(-1)
        if len(x)!=len(labels) or not np.isin(labels,[0,1]).all(): raise ValueError("Unexpected Bee-Dance labels")
        cps=[int(p) for p in np.flatnonzero(labels) if 0<p<len(x)]
        series.append(Series(name.removesuffix(".mat"),x,cps))
        sources.append({"url":url,"sha256":file_hash(file)})
    save_dataset(output,series,{"source":"KL-CPD upstream dataset","commit":KLCPD_COMMIT,"files":sources,
                              "note":"Six recordings split separately; exact EvoTS preprocessing is not specified in its paper."})


def stitch_adia(manifest_path,output,half_window=150,bucket_width=1.0,windows_per_stream=6):
    """Explicit independent reconstruction, NOT the unavailable author recipe.

    Input list: {file: local CSV, value_column: numeric column, break_index: int}.
    Groups by normalized signed mean shift and log volatility ratio, then sorts
    by source order. Artificial joins are labeled and recorded separately.
    """
    if half_window<15 or bucket_width<=0 or windows_per_stream<2: raise ValueError("Invalid stitching parameters")
    manifest_path=Path(manifest_path).resolve(); entries=read_json(manifest_path)
    groups={}; sources=[]
    for i,e in enumerate(entries):
        file=(manifest_path.parent/e["file"]).resolve()
        raw=np.genfromtxt(file,delimiter=",",names=True,encoding="utf-8")
        x=np.asarray(raw[e["value_column"]],dtype=float)
        b=e["break_index"]
        if type(b) is not int or not half_window<=b<=len(x)-half_window:
            raise ValueError(f"Not enough observations around annotated break in {file.name}")
        w=x[b-half_window:b+half_window]
        if not np.isfinite(w).all(): raise ValueError("ADIA windows must be finite")
        left,right=w[:half_window],w[half_window:]
        mu=(right.mean()-left.mean())/(w.std()+1e-8)
        volatility=np.log((right.std()+1e-8)/(left.std()+1e-8))
        group=(int(np.floor(mu/bucket_width)),int(np.floor(volatility/bucket_width)))
        groups.setdefault(group,[]).append((i,w))
        sources.append({"source_id":i,"file":file.name,"sha256":file_hash(file),"break_index":b})
    output_series=[]; stream_meta=[]
    for group,items in sorted(groups.items()):
        for offset in range(0,len(items),windows_per_stream):
            chunk=items[offset:offset+windows_per_stream]
            x=np.concatenate([w for _,w in chunk])
            real=[2*half_window*j+half_window for j in range(len(chunk))]
            seams=[2*half_window*j for j in range(1,len(chunk))]
            name=f"adia_stream_{len(output_series):04d}"
            output_series.append(Series(name,x,sorted(real+seams)))
            stream_meta.append({"name":name,"source_ids":[i for i,_ in chunk],"group":list(group),
                                "annotated_breaks":real,"artificial_join_boundaries":seams})
    if not output_series: raise ValueError("Empty ADIA input")
    save_dataset(output,output_series,{"recipe":"independent ADIA stitching, not author-exact",
                  "half_window":half_window,"bucket_width":bucket_width,"sources":sources,"streams":stream_meta,
                  "seam_policy":"all artificial window joins count as boundaries"})
