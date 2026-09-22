"""Prompt templates for EvoTS-Agent."""

EDA_SYSTEM_PROMPT = """You are an expert financial time series analyst. You perform exploratory data analysis to characterize dataset properties and select candidate change-point detection models.

You have access to the following meta-features computed from the dataset:
- lag1_autocorrelation: Lag-1 autocorrelation
- trend_strength: Strength of linear trend
- nonstationarity: ADF test p-value
- spectral_concentration: Ratio of dominant spectral power
- periodicity: Dominant period from FFT
- missing_ratio: Fraction of missing values
- sequence_length: Number of observations
- dimensionality: Number of variables
- local_mean_discrepancy: Maximum local mean shift (S_mu)
- local_variance_discrepancy: Maximum local variance shift (S_sigma)
"""

EDA_USER_PROMPT = """Dataset profile:
- Sequence length: {sequence_length}
- Dimensionality: {dimensionality}
- Lag-1 autocorrelation: {lag1_autocorrelation:.3f}
- Trend strength: {trend_strength:.3f}
- Nonstationarity (ADF p-value): {nonstationarity:.3f}
- Spectral concentration: {spectral_concentration:.3f}
- Periodicity: {periodicity:.1f}
- Missing ratio: {missing_ratio:.3f}
- Local mean discrepancy (S_mu): {local_mean_discrepancy:.3f}
- Local variance discrepancy (S_sigma): {local_variance_discrepancy:.3f}

Available models:
{model_descriptions}

Select top-{k} models most suitable for this dataset. Return ONLY a JSON list of model names."""

MODEL_SELECTION_PROMPT = """Based on the EDA results above, select the top-K most suitable change-point detection models.

Consider:
- Mean-shift detectors for high S_mu
- Variance-shift detectors for high S_sigma
- Kernel-based methods for complex distribution changes
- Bayesian methods when uncertainty matters
- Spectral methods for frequency-domain changes
- Deep learning for high-dimensional data

Return JSON: {{"models": ["model_name_1", "model_name_2", ...], "rationale": "brief explanation"}}"""

REVISION_PROMPT = """You are refining the best current change-point detection pipeline.

INCUMBENT SCRIPT:
```python
{incumbent_script}
```

INCUMBENT PERFORMANCE:
- F1: {f1_score:.3f}
- Hausdorff distance: {hausdorff:.2f}
- Change points found: {num_change_points}

RECENT TRAJECTORY CONTEXT:
{recent_context}

REVISION INSTRUCTIONS:
1. Modify the incumbent script to improve validation F1
2. Changes may include: preprocessing, hyperparameters, post-processing, feature engineering
3. Preserve the same input/output interface
4. Return ONLY the modified executable Python script"""

ALTERNATIVE_STRATEGY_PROMPT = """The current approach has stagnated (F1 improvement < {threshold}).

INCUMBENT SCRIPT:
```python
{incumbent_script}
```

STAGNANT REVISION ATTEMPTS:
{stagnant_context}

ALTERNATIVE MODELS TO CONSIDER:
{alternative_models}

ALTERNATIVE STRATEGY INSTRUCTIONS:
1. Select a fundamentally different modeling direction OR an untried EDA-recommended model
2. Implement a complete alternative pipeline
3. Return ONLY the new executable Python script"""

RECOMBINATION_PROMPT = """Synthesize the best ideas from multiple successful trajectories.

INCUMBENT SCRIPT:
```python
{incumbent_script}
```

HIGH-PERFORMING TRAJECTORIES:
{strong_trajectories}

RECOMBINATION INSTRUCTIONS:
1. Analyze what made each trajectory successful
2. Combine complementary components (e.g., preprocessing from one, detector from another, post-processing from a third)
3. Create a final pipeline that integrates these successful ideas
4. Return ONLY the recombined executable Python script"""

EXECUTE_VALIDATE_PROMPT = """Execute the following change-point detection script and return the detected change points.

Script:
```python
{script}
```

Expected function signature:
```python
def detect_change_points(data: np.ndarray) -> list[int]:
    ...
    return change_points  # List of integer indices
```

Return JSON: {{"success": true/false, "change_points": [...], "error": "..."}}"""
