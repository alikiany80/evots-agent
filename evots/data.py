"""Explicit chronological data protocol. No random shuffling across time."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .io import file_hash, read_json, write_json
from .metrics import boundaries


@dataclass
class Series:
    name: str
    x: np.ndarray
    cps: list[int]

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim == 1:
            self.x = self.x[:, None]
        if self.x.ndim != 2 or len(self.x) < 15 or self.x.shape[1] < 1:
            raise ValueError("Each series must be a numeric [time, features] array with >=15 samples")
        if np.isinf(self.x).any():
            raise ValueError("Infinite observations are invalid; NaN is supported")
        self.cps = boundaries(self.cps, len(self.x))


def synthetic(kind="mean_variance", n=3000, count=3, seed=42, dimensions=1, segment_length=125):
    if kind not in {"mean_variance", "ou"}:
        raise ValueError("Synthetic kind must be mean_variance or ou")
    if n < 100 or count < 1 or dimensions < 1 or segment_length < 10:
        raise ValueError("Invalid synthetic dimensions")
    result = []
    for i in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        cps = list(range(segment_length, n, segment_length))
        x = np.zeros((n, dimensions))
        for j, (start, end) in enumerate(zip([0] + cps, cps + [n])):
            mu = (-1 if j % 2 else 1) * rng.uniform(0.8, 2.5, dimensions)
            sigma = rng.uniform(0.2, 0.6, dimensions) if j % 2 else rng.uniform(0.8, 1.4, dimensions)
            if kind == "mean_variance":
                x[start:end] = rng.normal(mu, sigma, (end-start, dimensions))
            else:
                theta = rng.uniform(0.06, 0.3, dimensions)
                decay = np.exp(-theta)
                innovation = sigma * np.sqrt((1-decay**2)/(2*theta))
                previous = x[start-1] if start else mu.copy()
                for t in range(start, end):
                    previous = mu + decay*(previous-mu) + innovation*rng.normal(size=dimensions)
                    x[t] = previous
        result.append(Series(f"{kind}_{i}", x, cps))
    return result


def save_dataset(path, series, metadata=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    entries = []
    for i, s in enumerate(series):
        file = f"series_{i:04d}.npz"
        np.savez_compressed(path / file, x=s.x, cps=np.asarray(s.cps, dtype=np.int64))
        entries.append({"name": s.name, "file": file, "sha256": file_hash(path/file)})
    write_json(path / "dataset.json", {"format_version": 1, "metadata": metadata or {}, "series": entries})


def load_dataset(path):
    path = Path(path).resolve()
    manifest = read_json(path / "dataset.json")
    result = []
    for item in manifest["series"]:
        file = (path/item["file"]).resolve()
        if not file.is_relative_to(path):
            raise ValueError("Dataset file escapes dataset directory")
        if file_hash(file) != item["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {item['name']}")
        with np.load(file, allow_pickle=False) as a:
            result.append(Series(item["name"], a["x"], a["cps"].tolist()))
    if not result or len({s.name for s in result}) != len(result):
        raise ValueError("Dataset must contain distinct named series")
    return result


def chronological_split(series: list[Series], ratios=(0.6, 0.2, 0.2)):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive split fractions must sum to 1")
    output = {"train": [], "validation": [], "test": []}
    for s in series:
        n = len(s.x)
        edges = [0, int(n*ratios[0]), int(n*(ratios[0]+ratios[1])), n]
        for name, start, end in zip(output, edges[:-1], edges[1:]):
            # A break exactly at a split edge is not an internal boundary.
            cps = [] if name == "train" else [p-start for p in s.cps if start < p < end]
            output[name].append(Series(s.name, s.x[start:end].copy(), cps))
    return output


def prepare(dataset, output, ratios=(0.6, 0.2, 0.2)):
    parts = chronological_split(load_dataset(dataset), ratios)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split, data in parts.items():
        save_dataset(output/split, data, {"split": split, "ratios": list(ratios)})
    write_json(output/"protocol.json", {"source_manifest_sha256": file_hash(Path(dataset)/"dataset.json"),
               "ratios": list(ratios), "boundary_convention": "0-based index of first observation of new segment",
               "train_labels_removed": True})


def import_csv(csv_path, labels_path, columns, output):
    array = np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8")
    if not columns or any(c not in array.dtype.names for c in columns):
        raise ValueError("Supply exact numeric feature column names; never include timestamp or label columns")
    labels = read_json(labels_path)
    save_dataset(output, [Series(Path(csv_path).stem, np.column_stack([array[c] for c in columns]), labels)],
                 {"csv_sha256": file_hash(csv_path), "labels_sha256": file_hash(labels_path), "columns": columns})


def import_bee(mat_path, output):
    from scipy.io import loadmat
    mat = loadmat(mat_path)
    if "Y" not in mat or "L" not in mat:
        raise ValueError("Expected KL-CPD .mat format with Y [time, features] and L boundary indicator")
    x, labels = np.asarray(mat["Y"]), np.asarray(mat["L"]).reshape(-1)
    if len(labels) != len(x) or not np.isin(labels, [0, 1]).all():
        raise ValueError("L must be a binary boundary indicator, not regime labels")
    cps = np.flatnonzero(labels).tolist()
    cps = [c for c in cps if 0 < c < len(x)]
    save_dataset(output, [Series("beedance", x, cps)], {"source_sha256": file_hash(mat_path), "source_format": "KL-CPD Y/L"})
