"""Experiment trajectory recording and management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentTrajectory:
    """Single executable experiment trajectory (τ_k from paper)."""

    step: int
    parent_ids: List[int] = field(default_factory=list)
    operation: str = ""  # Revision | AlternativeStrategy | Recombination
    plan: str = ""
    summary: str = ""
    model_name: str = ""
    script: str = ""
    validation_score: float = 0.0
    execution_log: str = ""
    code_diff: str = ""
    decision: str = "Pending"  # Accept | Reject
    is_stagnant: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "parent_ids": self.parent_ids,
            "operation": self.operation,
            "summary": self.summary,
            "model_name": self.model_name,
            "validation_score": self.validation_score,
            "decision": self.decision,
            "is_stagnant": self.is_stagnant,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentTrajectory:
        traj = cls(
            step=data.get("step", 0),
            parent_ids=data.get("parent_ids", []),
            operation=data.get("operation", ""),
            plan=data.get("plan", ""),
            summary=data.get("summary", ""),
            model_name=data.get("model_name", ""),
            script=data.get("script", ""),
            validation_score=data.get("validation_score", 0.0),
            execution_log=data.get("execution_log", ""),
            code_diff=data.get("code_diff", ""),
            decision=data.get("decision", "Pending"),
            is_stagnant=data.get("is_stagnant", False),
            timestamp=data.get("timestamp", 0.0),
        )
        return traj


class TrajectoryPool:
    """Trajectory pool (𝒯) for storing and retrieving experiment trajectories."""

    def __init__(self):
        self.trajectories: List[ExperimentTrajectory] = []
        self._next_step = 0

    def add(self, trajectory: ExperimentTrajectory) -> None:
        trajectory.step = self._next_step
        self.trajectories.append(trajectory)
        self._next_step += 1

    def get_incumbent(self) -> Optional[ExperimentTrajectory]:
        """Get the trajectory with the highest validation score."""
        if not self.trajectories:
            return None
        accepted = [t for t in self.trajectories if t.decision == "Accept"]
        if not accepted:
            return max(self.trajectories, key=lambda t: t.validation_score)
        return max(accepted, key=lambda t: t.validation_score)

    def get_strong_trajectories(self, model_name: str, top_n: int = 3) -> List[ExperimentTrajectory]:
        """Get top-N performing trajectories for a given model."""
        model_trajs = [t for t in self.trajectories if t.model_name == model_name]
        model_trajs.sort(key=lambda t: t.validation_score, reverse=True)
        return model_trajs[:top_n]

    def get_recent(self, model_name: str, n: int = 4) -> List[ExperimentTrajectory]:
        """Get recent trajectories for a given model."""
        model_trajs = [t for t in self.trajectories if t.model_name == model_name]
        return model_trajs[-n:]

    def get_stagnant_for_model(self, model_name: str) -> Optional[ExperimentTrajectory]:
        """Get the most recent stagnant trajectory for a model."""
        stagnant = [t for t in self.trajectories if t.model_name == model_name and t.is_stagnant]
        if stagnant:
            return stagnant[-1]
        return None

    def save(self, path: str) -> None:
        """Save trajectory pool to JSON."""
        data = [t.to_dict() for t in self.trajectories]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load trajectory pool from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.trajectories = [ExperimentTrajectory.from_dict(d) for d in data]
        self._next_step = len(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __repr__(self) -> str:
        return f"TrajectoryPool(size={len(self.trajectories)}, incumbent_f1={self.get_incumbent().validation_score:.3f if self.get_incumbent() else 'N/A'})"
