from __future__ import annotations

import html
import json
from pathlib import Path
from .io import read_json


def report_run(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=Path(directory)
    trials=[read_json(p) for p in sorted((directory/"trajectories").glob("*.json"))]
    if not trials: raise ValueError("No trajectories found")
    best=0; line=[]
    for t in trials:
        if t["q"] is not None: best=max(best,t["q"])
        line.append(best)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(range(1,len(line)+1),line,label="Best validation F1",color="#176b87",linewidth=2)
    for op in sorted({t["operator"] for t in trials}):
        vals=[t for t in trials if t["operator"]==op and t["q"] is not None]
        ax.scatter([t["id"]+1 for t in vals],[t["q"] for t in vals],label=op,s=35)
    ax.set(xlabel="Trial",ylabel="Validation F1",ylim=(-0.03,1.03),title="Validation-guided search")
    ax.legend(fontsize=8); ax.grid(alpha=0.15); fig.tight_layout()
    fig.savefig(directory/"search.png",dpi=160); plt.close(fig)
    rows="".join(f"<tr><td>{t['id']}</td><td>{html.escape(t['operator'])}</td><td>{html.escape(t['model'])}</td><td>{t['q']}</td><td>{t['accepted']}</td><td>{t.get('stagnant',False)}</td><td>{t.get('parents',[])}</td></tr>" for t in trials)
    summary=read_json(directory/"search_complete.json") if (directory/"search_complete.json").exists() else {"complete":False}
    test=read_json(directory/"test_results.json") if (directory/"test_results.json").exists() else None
    usage=read_json(directory/"llm"/"usage.json") if (directory/"llm"/"usage.json").exists() else {"reported_cost_usd":0,"mode":"no API calls"}
    manifest=read_json(directory/"manifest.json")
    banner="MOCK RUN — infrastructure demonstration; no LLM evidence" if manifest.get("mock") else "Independent research implementation — see docs/FIDELITY.md"
    body=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>EvoTS experiment report</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#18313e;background:#f5f8fa}}h1{{font-size:30px}}section{{background:white;padding:22px;margin:18px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}img{{width:100%}}.banner{{background:#fff0c8;padding:14px}}</style>
    <h1>EvoTS experiment report</h1><p class="banner">{html.escape(banner)}</p>
    <section><h2>Search</h2><pre>{html.escape(json.dumps(summary,indent=2))}</pre><img src="search.png" alt="Validation search progress"></section>
    <section><h2>Held-out test</h2><pre>{html.escape(json.dumps(test['scores'] if test else 'Not evaluated. Run evots evaluate after search.',indent=2))}</pre></section>
    <section><h2>API accounting</h2><pre>{html.escape(json.dumps(usage,indent=2))}</pre></section>
    <section><h2>Trajectory history</h2><table><tr><th>Trial</th><th>Operator</th><th>Model</th><th>F1</th><th>Accepted</th><th>Stagnant</th><th>Parents</th></tr>{rows}</table></section>
    <p>Scores are computed outside candidate execution. Primary selection uses validation F1 only. Training and test annotation access is excluded from the planner.</p></html>"""
    (directory/"report.html").write_text(body,encoding="utf-8")
    return str(directory/"report.html")
