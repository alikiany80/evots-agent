"""EvoTS-Agent: Self-Evolving LLM Agent for Financial Time Series Change Point Detection."""

from .agent import EvoTSAgent, EvoTSConfig, EvoTSResult
from .eda import DatasetProfile, compute_eda
from .executor import CodeExecutor, validate_boundaries
from .model_bank import ChangePointModel, ModelBank
from .trajectory import ExperimentTrajectory, TrajectoryPool

__version__ = "1.0.0"

__all__ = [
    "EvoTSAgent",
    "EvoTSConfig",
    "EvoTSResult",
    "DatasetProfile",
    "compute_eda",
    "CodeExecutor",
    "validate_boundaries",
    "ChangePointModel",
    "ModelBank",
    "ExperimentTrajectory",
    "TrajectoryPool",
]
