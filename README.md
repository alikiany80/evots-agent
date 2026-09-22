# EvoTS-Agent

**A Self-Evolving LLM Agent for Financial Time Series Change Point Detection**

Based on the paper: [EvoTS-Agent: A Self-Evolving LLM Agent for Financial Time Series Change Point Detection](https://arxiv.org/abs/2608.17933)

## Overview

EvoTS-Agent autonomously detects structural change points in financial time series by combining:
- **Curated EDA** → dataset characterization and model initialization
- **Model Bank** → diverse change-point detection algorithms
- **Trajectory Evolution** → three self-evolution operators (Revision, Alternative Strategy, Recombination)
- **Validation-Guided Search** → boundary-aware F1 optimization

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    EvoTS-Agent                          │
├─────────────────────────────────────────────────────────┤
│  EDA Module → Meta-features → Model Selection (top-K)  │
│                          ↓                              │
│  Warm-up: 2×K trajectories (initial + revision)        │
│                          ↓                              │
│  Optimization Loop:                                     │
│    ┌─ Revision (exploit incumbent)                      │
│    ├─ Alternative Strategy (explore when stagnated)     │
│    └─ Recombine (final synthesis)                      │
│                          ↓                              │
│  Best validated model → Final change-point detection    │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from evots_agent import EvoTSAgent

agent = EvoTSAgent(llm_config={"provider": "openai", "model": "gpt-4o"})
result = agent.detect(data, validation_boundaries=val_bp)

print(f"Change points: {result.change_points}")
print(f"Best model: {result.model_name}")
print(f"Validation F1: {result.f1_score}")
```

## Model Bank

- PELT (Pruned Exact Linear Time)
- Binary Segmentation
- Window-based (WIN)
- Kernel-based (MMD)
- Bayesian Online
- Spectral methods
- Deep Learning (KL-CPD)

## License

MIT
