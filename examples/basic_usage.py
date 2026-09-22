"""Example: Using EvoTS-Agent for financial time series change-point detection."""

import numpy as np
from evots_agent import EvoTSAgent, EvoTSConfig

# Generate synthetic financial data with change points
np.random.seed(42)
n = 500

# Create a price series with volatility regimes
returns = np.concatenate([
    np.random.randn(150) * 0.01,      # Low volatility
    np.random.randn(100) * 0.05,      # High volatility (crisis)
    np.random.randn(150) * 0.015,     # Medium volatility
    np.random.randn(100) * 0.04,      # Another crisis
])
prices = 100 * np.exp(np.cumsum(returns))

# True change points
true_change_points = [150, 250, 400]

# Run EvoTS-Agent
print("=" * 60)
print("EvoTS-Agent: Self-Evolving Change Point Detection")
print("=" * 60)

agent = EvoTSAgent(config=EvoTSConfig(
    budget=10,
    top_k=3,
    model_name="gpt-4o",
))

result = agent.detect(
    data=prices,
    validation_boundaries=true_change_points,
)

print(f"\nResults:")
print(f"  Model: {result.model_name}")
print(f"  F1 Score: {result.f1_score:.3f}")
print(f"  Precision: {result.precision:.3f}")
print(f"  Recall: {result.recall:.3f}")
print(f"  Detected change points: {len(result.change_points)}")
print(f"  Iterations: {result.num_iterations}")
