"""Exploratory Data Analysis (EDA) for change-point detection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DatasetProfile:
    """Dataset characterization from EDA."""
    sequence_length: int
    dimensionality: int
    lag1_autocorrelation: float
    trend_strength: float
    nonstationarity: float
    spectral_concentration: float
    periodicity: float
    missing_ratio: float
    local_mean_discrepancy: float
    local_variance_discrepancy: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "sequence_length": self.sequence_length,
            "dimensionality": self.dimensionality,
            "lag1_autocorrelation": self.lag1_autocorrelation,
            "trend_strength": self.trend_strength,
            "nonstationarity": self.nonstationarity,
            "spectral_concentration": self.spectral_concentration,
            "periodicity": self.periodicity,
            "missing_ratio": self.missing_ratio,
            "local_mean_discrepancy": self.local_mean_discrepancy,
            "local_variance_discrepancy": self.local_variance_discrepancy,
        }


def compute_eda(data: np.ndarray) -> DatasetProfile:
    """
    Compute curated meta-features for change-point detection.
    
    Implements Section 4.1 of the EvoTS-Agent paper:
    - Lag-1 autocorrelation
    - Trend strength
    - Nonstationarity (ADF-like statistic)
    - Spectral concentration
    - Periodicity (dominant frequency)
    - Missing ratio
    - Local mean/variance discrepancy (Equations 9-11)
    """
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    n, d = signal.shape

    # Missing ratio
    missing_ratio = float(np.mean(np.isnan(signal)))
    
    # Replace NaN for further computations
    signal = np.nan_to_num(signal, nan=0.0)

    # Lag-1 autocorrelation (mean across dimensions)
    if n > 1:
        corrs = []
        for j in range(d):
            x = signal[:, j]
            var = np.var(x)
            if var > 1e-10:
                corr = np.corrcoef(x[:-1], x[1:])[0, 1]
                corrs.append(corr)
        lag1_autocorrelation = float(np.mean(corrs)) if corrs else 0.0
    else:
        lag1_autocorrelation = 0.0

    # Trend strength (linear regression R²)
    x_axis = np.arange(n)
    if n > 1 and np.var(signal) > 1e-10:
        slope = np.polyfit(x_axis, signal.flatten()[:n], 1)[0]
        trend_strength = float(min(abs(slope) * n / (np.std(signal.flatten()[:n]) * np.sqrt(n) + 1e-8), 1.0))
    else:
        trend_strength = 0.0

    # Nonstationarity: simplified ADF-like statistic
    if n > 1:
        diffs = np.diff(signal, axis=0)
        levels = signal[:-1]
        diff_var = np.var(diffs)
        level_var = np.var(levels)
        nonstationarity = float(1.0 - diff_var / (level_var + 1e-8))
        nonstationarity = max(0.0, min(1.0, nonstationarity))
    else:
        nonstationarity = 0.0

    # Spectral concentration (FFT energy in top frequency)
    if n > 4:
        fft_vals = np.abs(np.fft.rfft(signal.flatten()[:n] - np.mean(signal.flatten()[:n])))
        psd = fft_vals ** 2
        total_energy = np.sum(psd) + 1e-10
        sorted_psd = np.sort(psd)[::-1]
        # Top-2 frequency ratio
        spectral_concentration = float(sorted_psd[:2].sum() / total_energy) if len(sorted_psd) > 1 else 1.0
    else:
        spectral_concentration = 1.0

    # Periodicity (dominant period)
    if n > 4:
        fft_vals = np.abs(np.fft.rfft(signal.flatten()[:n] - np.mean(signal.flatten()[:n])))
        # Exclude DC component
        if len(fft_vals) > 1:
            dominant_idx = np.argmax(fft_vals[1:]) + 1
            period = n / dominant_idx if dominant_idx > 0 else float(n)
            periodicity = float(min(period, n / 2))
        else:
            periodicity = float(n)
    else:
        periodicity = float(n)

    # Local mean and variance discrepancy (Equations 9-11)
    window = max(n // 20, 5)  # Window width
    local_mean_disc = 0.0
    local_var_disc = 0.0
    sigma_x = np.std(signal.flatten()[:n]) + 1e-8

    for t in range(window, n - window):
        left = signal[t - window:t]
        right = signal[t:t + window]
        
        # Equation 9: Δμ(t)
        delta_mu = abs(np.mean(right) - np.mean(left)) / (sigma_x + 1e-8)
        if delta_mu > local_mean_disc:
            local_mean_disc = float(delta_mu)
        
        # Equation 10: Δσ(t)
        delta_sigma = abs(np.std(right) - np.std(left)) / (sigma_x + 1e-8)
        if delta_sigma > local_var_disc:
            local_var_disc = float(delta_sigma)

    return DatasetProfile(
        sequence_length=n,
        dimensionality=d,
        lag1_autocorrelation=lag1_autocorrelation,
        trend_strength=trend_strength,
        nonstationarity=nonstationarity,
        spectral_concentration=spectral_concentration,
        periodicity=periodicity,
        missing_ratio=missing_ratio,
        local_mean_discrepancy=local_mean_disc,
        local_variance_discrepancy=local_var_disc,
    )
