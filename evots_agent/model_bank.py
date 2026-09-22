"""Model bank of change-point detection methods."""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional

# Type alias for detector functions
DetectorFn = Callable[[np.ndarray], List[int]]


class ChangePointModel:
    """A change-point detection model."""

    def __init__(
        self,
        name: str,
        description: str,
        strengths: List[str],
        constraints: str,
        baseline_script: str,
    ):
        self.name = name
        self.description = description
        self.strengths = strengths
        self.constraints = constraints
        self.baseline_script = baseline_script


# Default baseline scripts for each detector

PELT_SCRIPT = '''"""PELT (Pruned Exact Linear Time) change-point detection."""
import numpy as np
from ruptures import Pelt

def detect_change_points(data: np.ndarray) -> list[int]:
    """Detect change points using PELT with RBF cost model."""
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    model = Pelt(model="rbf", min_size=10).fit(signal)
    # Use penalty determined by BIC heuristic
    n = len(signal)
    penalty = np.log(n) * signal.shape[1]
    result = model.predict(pen=penalty)
    if result and result[-1] == n:
        result = result[:-1]
    return result
'''

BINSEG_SCRIPT = '''"""Binary Segmentation change-point detection."""
import numpy as np
from ruptures import Binseg

def detect_change_points(data: np.ndarray) -> list[int]:
    """Detect change points using Binary Segmentation."""
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    model = Binseg(model="l2", min_size=10).fit(signal)
    n = len(signal)
    penalty = 3 * np.log(n)
    result = model.predict(n_bkps=penalty)
    if result and result[-1] == n:
        result = result[:-1]
    return result
'''

WIN_SCRIPT = '''"""Window-based (WIN) change-point detection."""
import numpy as np
from ruptures import Window

def detect_change_points(data: np.ndarray) -> list[int]:
    """Detect change points using sliding window approach."""
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    model = Window(width=40, model="l2", min_size=10).fit(signal)
    n = len(signal)
    penalty = np.log(n) * signal.shape[1]
    result = model.predict(pen=penalty)
    if result and result[-1] == n:
        result = result[:-1]
    return result
'''

KERNEL_SCRIPT = '''"""Kernel-based (MMD) change-point detection."""
import numpy as np
from ruptures import Dynp

def detect_change_points(data: np.ndarray) -> list[int]:
    """Detect change points using kernel-based MMD method."""
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    # Kernel-based with RBF kernel via dynamic programming
    model = Dynp(model="rbf", min_size=10).fit(signal)
    n = len(signal)
    penalty = 2 * np.log(n)
    result = model.predict(pen=penalty)
    if result and result[-1] == n:
        result = result[:-1]
    return result
'''

BAYESIAN_SCRIPT = '''"""Bayesian Online change-point detection."""
import numpy as np

def detect_change_points(data: np.ndarray) -> list[int]:
    """Detect change points using Bayesian online approach."""
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    n, d = signal.shape
    change_points = []
    
    # Simple Bayesian approach: detect where posterior hazard rate exceeds threshold
    hazard = 1 / 100  # Prior: change point every 100 observations
    mu_prior = np.mean(signal[:50], axis=0) if n > 50 else np.mean(signal, axis=0)
    var_prior = np.var(signal[:50], axis=0) + 1e-6 if n > 50 else np.var(signal, axis=0) + 1e-6
    
    mu_pred = mu_prior.copy()
    var_pred = var_prior.copy()
    run_length = 0
    threshold = 0.8  # Probability threshold
    
    for t in range(1, n):
        x = signal[t]
        # Predictive probability under current run
        z_score = np.abs(x - mu_pred) / np.sqrt(var_pred + 1e-8)
        pred_prob = np.exp(-0.5 * np.mean(z_score ** 2))
        
        # Hazard probability
        H = hazard
        growth_prob = (1 - H) * pred_prob
        change_prob = H
        
        # Normalize
        total = growth_prob + change_prob
        
        if change_prob > threshold * total and t > 10:
            change_points.append(t)
            mu_pred = x.copy()
            var_pred = var_prior.copy()
            run_length = 0
        else:
            mu_pred = (run_length * mu_pred + x) / (run_length + 1)
            var_pred = var_prior / (run_length + 1)
            run_length += 1
    
    return change_points
'''

SPECTRAL_SCRIPT = '''"""Spectral (CSS) change-point detection for frequency-domain changes."""
import numpy as np
from scipy.signal import welch
from scipy.stats import chi2

def detect_change_points(data: np.ndarray) -> list[int]:
    """Detect change points via spectral density comparison."""
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    n, d = signal.shape
    change_points = []
    
    # Compare spectral density in adjacent windows
    window_size = min(n // 5, 100)
    step = max(1, window_size // 4)
    
    prev_freq = None
    for i in range(0, n - window_size, step):
        segment = signal[i:i + window_size]
        freq, psd = welch(segment.flatten(), nperseg=min(window_size, 128))
        
        if prev_freq is not None:
            # KL divergence between current and previous spectrum
            p = psd / (np.sum(psd) + 1e-10)
            q = prev_freq / (np.sum(prev_freq) + 1e-10)
            kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
            
            if kl > 0.5 and i > 10:
                change_points.append(i)
        
        prev_freq = psd.copy()
    
    return change_points
'''

DEEP_LEARNING_SCRIPT = '''"""Deep Learning (KL-CPD style) change-point detection."""
import numpy as np

def detect_change_points(data: np.ndarray) -> list[int]:
    """Detect change points via latent representation differences (simplified KL-CPD)."""
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    n, d = signal.shape
    change_points = []
    
    # Simple latent representation: moving PCA components
    window = min(n // 5, 100)
    step = max(1, window // 4)
    
    prev_repr = None
    for i in range(0, n - window, step):
        segment = signal[i:i + window]
        # Centered representation
        centered = segment - np.mean(segment, axis=0)
        if centered.shape[0] > 1:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            repr_vec = eigvecs[:, -1]  # Top eigenvector
        else:
            repr_vec = centered[0]
        
        if prev_repr is not None:
            cosine_sim = np.abs(np.dot(repr_vec, prev_repr)) / (np.linalg.norm(repr_vec) * np.linalg.norm(prev_repr) + 1e-10)
            if cosine_sim < 0.7 and i > 10:
                change_points.append(i)
        
        prev_repr = repr_vec.copy()
    
    return change_points
'''


class ModelBank:
    """Comprehensive model bank of change-point detection methods."""

    def __init__(self):
        self.models: Dict[str, ChangePointModel] = {}
        self._initialize()

    def _initialize(self):
        """Register all models."""
        self.models["PELT"] = ChangePointModel(
            name="PELT",
            description="Pruned Exact Linear Time - optimal segmentation with RBF cost. Best for mean/variance shifts in univariate series.",
            strengths=["mean_shift", "variance_shift", "univariate", "abrupt_changes"],
            constraints=["requires ruptures library", "sensitive to penalty parameter"],
            baseline_script=PELT_SCRIPT,
        )

        self.models["BinSeg"] = ChangePointModel(
            name="BinSeg",
            description="Binary Segmentation - fast approximate method splitting at strongest change point.",
            strengths=["fast", "mean_shift", "univariate", "scalable"],
            constraints=["suboptimal for closely-spaced changes"],
            baseline_script=BINSEG_SCRIPT,
        )

        self.models["WIN"] = ChangePointModel(
            name="WIN",
            description="Window-based detection using sliding window comparison.",
            strengths=["gradual_changes", "robust", "any_cost_model"],
            constraints=["window size tuning needed"],
            baseline_script=WIN_SCRIPT,
        )

        self.models["Kernel"] = ChangePointModel(
            name="Kernel",
            description="Kernel-based MMD (Maximum Mean Discrepancy) for complex distribution changes.",
            strengths=["distribution_change", "nonparametric", "multivariate"],
            constraints=["computational cost for large n", "kernel bandwidth tuning"],
            baseline_script=KERNEL_SCRIPT,
        )

        self.models["Bayesian"] = ChangePointModel(
            name="Bayesian",
            description="Bayesian online change-point detection with hazard rate estimation.",
            strengths=["online", "uncertainty_quantification", "sequential"],
            constraints=["prior specification needed", "sensitive to hazard rate"],
            baseline_script=BAYESIAN_SCRIPT,
        )

        self.models["Spectral"] = ChangePointModel(
            name="Spectral",
            description="Spectral density comparison for frequency-domain structural changes.",
            strengths=["frequency_domain", "periodic_changes", "spectrum_shift"],
            constraints=["requires stationarity within windows", "window size sensitive"],
            baseline_script=SPECTRAL_SCRIPT,
        )

        self.models["DeepCPD"] = ChangePointModel(
            name="DeepCPD",
            description="Deep learning representation-based detection (KL-CPD style).",
            strengths=["high_dimensional", "latent_changes", "representation_learning"],
            constraints=["requires sufficient data", "computational overhead"],
            baseline_script=DEEP_LEARNING_SCRIPT,
        )

    def get(self, name: str) -> Optional[ChangePointModel]:
        """Get a model by name."""
        return self.models.get(name)

    def get_all(self) -> Dict[str, ChangePointModel]:
        """Get all registered models."""
        return self.models

    def get_script(self, name: str) -> str:
        """Get the baseline script for a model."""
        model = self.get(name)
        if model is None:
            raise ValueError(f"Unknown model: {name}")
        return model.baseline_script

    def get_descriptions(self) -> str:
        """Get formatted descriptions of all models for LLM prompts."""
        lines = []
        for name, model in self.models.items():
            lines.append(f"  {name}: {model.description}")
            lines.append(f"    Strengths: {', '.join(model.strengths)}")
            lines.append(f"    Constraints: {', '.join(model.constraints)}")
        return "\n".join(lines)

    def filter_by_strength(self, strengths: List[str]) -> List[str]:
        """Filter models by desired strengths."""
        matching = []
        for name, model in self.models.items():
            if any(s in model.strengths for s in strengths):
                matching.append(name)
        return matching
