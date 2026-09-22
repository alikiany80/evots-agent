from __future__ import annotations

import numpy as np
from .data import Series


def fill_missing(x, medians=None):
    x = np.asarray(x, dtype=float).copy()
    if medians is None:
        medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in x.T])
    return np.where(np.isnan(x), medians, x), medians


def describe(x, window=25):
    missing = float(np.isnan(x).mean())
    x, _ = fill_missing(x)
    n, d = x.shape
    w = min(window, max(2, n//4))
    scale = x.std(0)+1e-8
    centered = x-x.mean(0)
    lag = np.sum(centered[:-1]*centered[1:], axis=0)/(np.sum(centered**2,axis=0)+1e-8)
    t = np.arange(n, dtype=float)
    t -= t.mean()
    slope = t @ centered/(t@t+1e-8)
    residual = centered-t[:,None]*slope
    trend = np.clip(1-residual.var(0)/(x.var(0)+1e-8), 0, 1)
    power = np.abs(np.fft.rfft(centered, axis=0))**2
    power[0] = 0
    concentration = power.max(0)/(power.sum(0)+1e-8)
    dominant_bin = power.argmax(0)
    period = n/np.maximum(dominant_bin, 1)
    cs = np.vstack([np.zeros(d), np.cumsum(x, axis=0)])
    css = np.vstack([np.zeros(d), np.cumsum(x*x, axis=0)])
    positions = np.arange(w, n-w+1)
    left = (cs[positions]-cs[positions-w])/w
    right = (cs[positions+w]-cs[positions])/w
    ls = np.sqrt(np.maximum(0, (css[positions]-css[positions-w])/w-left**2))
    rs = np.sqrt(np.maximum(0, (css[positions+w]-css[positions])/w-right**2))
    return {"length": n, "dimensions": d, "missing_ratio": missing, "window": w,
            "lag1_autocorrelation": lag.tolist(), "trend_strength": trend.tolist(),
            "spectral_concentration": concentration.tolist(), "dominant_period_samples": period.tolist(),
            "local_mean_discrepancy_max": np.max(np.abs(right-left)/scale,axis=0).tolist(),
            "local_std_discrepancy_max": np.max(np.abs(rs-ls)/scale,axis=0).tolist(),
            "nonstationarity_proxy": (np.abs(x[:n//2].mean(0)-x[n//2:].mean(0))/scale).tolist()}


def profile(series: list[Series], seed=42):
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(len(series)))
    return {"sample_id": series[idx].name, "source_split": "train", "sample_index": idx,
            "n_series": len(series), "features": describe(series[idx].x)}


def plot_unlabeled(x, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(x[:, :min(x.shape[1],6)], linewidth=0.65)
    ax.set(xlabel="Sample", ylabel="Value", title="Training series — no labels")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
