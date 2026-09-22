"""One-to-one boundary matching; endpoints 0 and n are never events."""
from __future__ import annotations

import numbers
import numpy as np


def boundaries(points, n: int) -> list[int]:
    out = []
    for p in points:
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, numbers.Real):
            raise ValueError("Boundaries must be integer sample indices")
        if not np.isfinite(p) or float(p) != int(p) or not 0 < int(p) < n:
            raise ValueError(f"Boundary {p!r} is outside 1..{n-1}")
        out.append(int(p))
    if len(set(out)) != len(out):
        raise ValueError("Duplicate boundaries are invalid")
    return sorted(out)


def score_boundaries(truth, prediction, n: int, tolerance: int = 10) -> dict:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    truth, prediction = boundaries(truth, n), boundaries(prediction, n)
    # Earliest feasible matching is maximum-cardinality for sorted equal-radius
    # intervals. Nearest-neighbor greedy matching need not maximize cardinality.
    i = j = 0
    matches = []
    while i < len(truth) and j < len(prediction):
        if prediction[j] < truth[i] - tolerance:
            j += 1
        elif truth[i] < prediction[j] - tolerance:
            i += 1
        else:
            matches.append([truth[i], prediction[j]])
            i += 1
            j += 1
    tp = len(matches)
    precision = tp / len(prediction) if prediction else float(not truth)
    recall = tp / len(truth) if truth else float(not prediction)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if truth and prediction:
        distances = np.abs(np.subtract.outer(truth, prediction))
        hausdorff = float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))
    else:
        # Infinity has no portable JSON encoding. Keep null + explicit flag.
        hausdorff = 0.0 if not truth and not prediction else None
    return dict(f1=f1, precision=precision, recall=recall, hausdorff=hausdorff,
                hausdorff_infinite=hausdorff is None, tp=tp,
                fp=len(prediction)-tp, fn=len(truth)-tp, matches=matches)


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No sequences to score")
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("f1", "precision", "recall")}
    out["hausdorff_infinite"] = any(r["hausdorff_infinite"] for r in rows)
    out["hausdorff"] = None if out["hausdorff_infinite"] else float(np.mean([r["hausdorff"] for r in rows]))
    out["n_sequences"] = len(rows)
    return out
