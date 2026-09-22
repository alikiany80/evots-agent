"""Main EvoTS-Agent orchestrator."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .eda import DatasetProfile, compute_eda
from .executor import CodeExecutor, validate_boundaries
from .model_bank import ModelBank
from .prompts import (
    ALTERNATIVE_STRATEGY_PROMPT,
    EDA_USER_PROMPT,
    RECOMBINATION_PROMPT,
    REVISION_PROMPT,
)
from .trajectory import ExperimentTrajectory, TrajectoryPool

logger = logging.getLogger(__name__)

# Type alias for LLM callable
LLMCallable = Callable[[str, str], Tuple[str, str]]


@dataclass
class EvoTSConfig:
    """Configuration for EvoTS-Agent."""
    # Optimization budget
    budget: int = 20
    # Stagnation threshold
    stagnation_threshold: float = 0.01
    # Validation tolerance
    tolerance: float = 0.05
    # Top-K models to select
    top_k: int = 3
    # Number of alternatives
    num_alternatives: int = 2
    # Execution timeout
    timeout: int = 60
    # Model name for LLM
    model_name: str = "gpt-4o"


@dataclass
class EvoTSResult:
    """Result from EvoTS-Agent."""
    change_points: List[int]
    model_name: str
    f1_score: float
    precision: float
    recall: float
    hausdorff: float
    script: str
    num_iterations: int
    trajectories: List[Dict[str, Any]]

    def __repr__(self) -> str:
        return (
            f"EvoTSResult(model={self.model_name}, F1={self.f1_score:.3f}, "
            f"CPs={len(self.change_points)}, iter={self.num_iterations})"
        )


class EvoTSAgent:
    """
    Self-evolving LLM Agent for financial time series change-point detection.
    Implements the full pipeline from the paper:
    1. EDA → meta-features → model selection
    2. Warm-up: 2×K initial trajectories
    3. Optimization: Revision / Alternative / Recombination
    """

    def __init__(
        self,
        llm: Optional[LLMCallable] = None,
        config: Optional[EvoTSConfig] = None,
    ):
        self.llm = llm or self._default_llm
        self.config = config or EvoTSConfig()
        self.model_bank = ModelBank()
        self.executor = CodeExecutor(timeout=self.config.timeout)
        self.pool = TrajectoryPool()

    def detect(
        self,
        data: np.ndarray,
        validation_boundaries: Optional[List[int]] = None,
        profile: Optional[DatasetProfile] = None,
    ) -> EvoTSResult:
        """
        Main entry point: detect change points in financial time series.
        
        Args:
            data: Time series data (n,) or (n, d) numpy array
            validation_boundaries: Reference boundaries for validation
            profile: Pre-computed dataset profile (optional)
        
        Returns:
            EvoTSResult with detected change points and metadata
        """
        start_time = time.time()
        
        # Step 1: EDA
        if profile is None:
            profile = compute_eda(data)
        logger.info(f"EDA complete: {profile.to_dict()}")
        
        # Step 2: Model selection via LLM
        selected_models = self._select_models(profile)
        logger.info(f"Selected models: {selected_models}")
        
        # Step 3: Warm-up trajectories (2×K)
        self._warmup(data, validation_boundaries, selected_models)
        logger.info(f"Warm-up complete: {len(self.pool)} trajectories")
        
        # Step 4: Optimization loop
        incumbent = self._optimize(data, validation_boundaries, selected_models)
        logger.info(f"Optimization complete: incumbent F1={incumbent.validation_score:.3f}")
        
        # Step 5: Final result
        result = EvoTSResult(
            change_points=self._extract_change_points(incumbent),
            model_name=incumbent.model_name,
            f1_score=incumbent.validation_score,
            precision=0.0,  # TODO: store from validation
            recall=0.0,
            hausdorff=0.0,
            script=incumbent.script,
            num_iterations=len(self.pool),
            trajectories=[t.to_dict() for t in self.pool.trajectories],
        )
        
        elapsed = time.time() - start_time
        logger.info(f"EvoTS-Agent complete in {elapsed:.1f}s: {result}")
        
        return result

    def _select_models(self, profile: DatasetProfile) -> List[str]:
        """Use LLM to select top-K models based on EDA."""
        prompt = EDA_USER_PROMPT.format(
            sequence_length=profile.sequence_length,
            dimensionality=profile.dimensionality,
            lag1_autocorrelation=profile.lag1_autocorrelation,
            trend_strength=profile.trend_strength,
            nonstationarity=profile.nonstationarity,
            spectral_concentration=profile.spectral_concentration,
            periodicity=profile.periodicity,
            missing_ratio=profile.missing_ratio,
            local_mean_discrepancy=profile.local_mean_discrepancy,
            local_variance_discrepancy=profile.local_variance_discrepancy,
            model_descriptions=self.model_bank.get_descriptions(),
            k=self.config.top_k,
        )
        
        try:
            response, _ = self.llm("You are an expert financial time series analyst.", prompt)
            # Parse JSON from response
            models = self._parse_model_selection(response)
            if models:
                return models
        except Exception as e:
            logger.warning(f"Model selection failed: {e}, falling back to EDA-based")
        
        # Fallback: EDA-based selection
        return self._eda_based_selection(profile)

    def _eda_based_selection(self, profile: DatasetProfile) -> List[str]:
        """Fallback: select models based on EDA thresholds."""
        scores: Dict[str, float] = {}
        
        for name, model in self.model_bank.get_all().items():
            score = 0.0
            strengths = model.strengths
            
            if "mean_shift" in strengths and profile.local_mean_discrepancy > 1.0:
                score += 2.0
            if "variance_shift" in strengths and profile.local_variance_discrepancy > 0.5:
                score += 2.0
            if "distribution_change" in strengths and profile.nonstationarity > 0.7:
                score += 1.5
            if "frequency_domain" in strengths and profile.spectral_concentration > 0.3:
                score += 1.0
            if "multivariate" in strengths and profile.dimensionality > 1:
                score += 1.5
            if "high_dimensional" in strengths and profile.dimensionality > 10:
                score += 1.0
            if "online" in strengths:
                score += 0.5  # Slight preference
            
            scores[name] = score
        
        # Sort by score, return top-K
        sorted_models = sorted(scores, key=scores.get, reverse=True)
        return sorted_models[: self.config.top_k]

    def _warmup(
        self,
        data: np.ndarray,
        validation_boundaries: Optional[List[int]],
        selected_models: List[str],
    ) -> None:
        """Generate 2×K warm-up trajectories (initial + revision per model)."""
        for model_name in selected_models:
            script = self.model_bank.get_script(model_name)
            
            # Initial trajectory
            traj = self._create_trajectory(
                model_name=model_name,
                script=script,
                data=data,
                validation_boundaries=validation_boundaries,
                operation="Initial",
            )
            self.pool.add(traj)
            
            # Single revision
            revision_script = self._revise_script(
                script, traj, model_name, data, validation_boundaries
            )
            if revision_script:
                rev_traj = self._create_trajectory(
                    model_name=model_name,
                    script=revision_script,
                    data=data,
                    validation_boundaries=validation_boundaries,
                    operation="Revision",
                )
                self.pool.add(rev_traj)

    def _optimize(
        self,
        data: np.ndarray,
        validation_boundaries: Optional[List[int]],
        selected_models: List[str],
    ) -> ExperimentTrajectory:
        """Main optimization loop (Algorithm 1)."""
        incumbent = self.pool.get_incumbent()
        
        for iteration in range(self.config.budget):
            # Check if final iteration → Recombination
            if iteration == self.config.budget - 1:
                new_script = self._recombine(incumbent, selected_models, data, validation_boundaries)
                operation = "Recombination"
            # Check for stagnation → Alternative Strategy
            elif self._is_stagnated(incumbent):
                new_script = self._alternative_strategy(
                    incumbent, selected_models, data, validation_boundaries
                )
                operation = "AlternativeStrategy"
            else:
                new_script = self._revise_script(
                    incumbent.script, incumbent, incumbent.model_name,
                    data, validation_boundaries
                )
                operation = "Revision"
            
            if not new_script:
                logger.warning(f"Iteration {iteration}: no script generated, skipping")
                continue
            
            # Execute and validate
            traj = self._create_trajectory(
                model_name=incumbent.model_name,
                script=new_script,
                data=data,
                validation_boundaries=validation_boundaries,
                operation=operation,
            )
            
            # Check for stagnation
            if traj.validation_score <= incumbent.validation_score + self.config.stagnation_threshold:
                traj.is_stagnant = True
            
            # Accept/reject
            if traj.validation_score > incumbent.validation_score:
                traj.decision = "Accept"
                incumbent = traj
            else:
                traj.decision = "Reject"
            
            self.pool.add(traj)
            logger.info(
                f"Iteration {iteration}: {operation} → {traj.decision} "
                f"(F1={traj.validation_score:.3f}, incumbent={incumbent.validation_score:.3f})"
            )
        
        return incumbent

    def _is_stagnated(self, incumbent: ExperimentTrajectory) -> bool:
        """Check if revision has stagnated."""
        if self.pool.get_stagnant_for_model(incumbent.model_name) is not None:
            return True
        return False

    def _create_trajectory(
        self,
        model_name: str,
        script: str,
        data: np.ndarray,
        validation_boundaries: Optional[List[int]],
        operation: str,
    ) -> ExperimentTrajectory:
        """Create a trajectory by executing and validating."""
        traj = ExperimentTrajectory(
            step=len(self.pool),
            operation=operation,
            model_name=model_name,
            script=script,
        )
        
        if validation_boundaries is not None:
            result = self.executor.execute_and_validate(
                script, data, validation_boundaries,
                tolerance=self.config.tolerance,
            )
            traj.validation_score = result["metrics"].get("f1", 0.0)
            traj.execution_log = result.get("log", "")
        else:
            # No validation: just execute
            success, change_points, log = self.executor.execute(script, data)
            traj.execution_log = log
        
        return traj

    def _revise_script(
        self,
        incumbent_script: str,
        incumbent: ExperimentTrajectory,
        model_name: str,
        data: np.ndarray,
        validation_boundaries: Optional[List[int]],
    ) -> Optional[str]:
        """Generate revised script using LLM."""
        # Get recent trajectories for context
        recent = self.pool.get_recent(model_name, n=4)
        context = "\n".join(
            f"- Step {t.step}: {t.operation}, F1={t.validation_score:.3f}" for t in recent
        )
        
        prompt = REVISION_PROMPT.format(
            incumbent_script=incumbent_script,
            f1_score=incumbent.validation_score,
            hausdorff=0.0,  # TODO
            num_change_points=0,  # TODO
            recent_context=context,
        )
        
        try:
            response, _ = self.llm(
                "You are an expert at optimizing change-point detection algorithms.", prompt
            )
            return self._extract_script(response)
        except Exception as e:
            logger.warning(f"Revision failed: {e}")
            return None

    def _alternative_strategy(
        self,
        incumbent: ExperimentTrajectory,
        selected_models: List[str],
        data: np.ndarray,
        validation_boundaries: Optional[List[int]],
    ) -> Optional[str]:
        """Generate alternative strategy using LLM."""
        # Find alternative model
        alternative = None
        for model_name in selected_models:
            if model_name != incumbent.model_name:
                alternative = model_name
                break
        
        if not alternative:
            return None
        
        prompt = ALTERNATIVE_STRATEGY_PROMPT.format(
            incumbent_script=incumbent.script,
            threshold=self.config.stagnation_threshold,
            stagnant_context="Recent stagnant revisions failed to improve F1",
            alternative_models=self.model_bank.get_descriptions(),
        )
        
        try:
            response, _ = self.llm(
                "You are exploring alternative change-point detection strategies.", prompt
            )
            return self._extract_script(response)
        except Exception as e:
            logger.warning(f"Alternative strategy failed: {e}")
            return None

    def _recombine(
        self,
        incumbent: ExperimentTrajectory,
        selected_models: List[str],
        data: np.ndarray,
        validation_boundaries: Optional[List[int]],
    ) -> Optional[str]:
        """Recombine high-performing trajectories."""
        # Get strong trajectories
        strong_trajs = []
        for model_name in selected_models:
            trajs = self.pool.get_strong_trajectories(model_name, top_n=2)
            strong_trajs.extend(trajs)
        
        # Format context
        context_parts = []
        for t in strong_trajs:
            context_parts.append(
                f"Step {t.step} ({t.model_name}): F1={t.validation_score:.3f}\n"
                f"  Summary: {t.summary}\n"
            )
        
        prompt = RECOMBINATION_PROMPT.format(
            incumbent_script=incumbent.script,
            strong_trajectories="\n".join(context_parts) if context_parts else "No strong trajectories",
        )
        
        try:
            response, _ = self.llm(
                "You are synthesizing the best change-point detection approach from multiple successful experiments.",
                prompt,
            )
            return self._extract_script(response)
        except Exception as e:
            logger.warning(f"Recombination failed: {e}")
            return None

    @staticmethod
    def _default_llm(system: str, prompt: str) -> Tuple[str, str]:
        """Default LLM call: uses OpenAI-compatible API."""
        import os
        
        # Try OpenRouter first
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = "https://api.openai.com/v1"
        
        if not api_key:
            raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        
        # Make API call
        import urllib.request
        import json
        
        url = f"{base_url}/chat/completions"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        
        content = result["choices"][0]["message"]["content"]
        return content, ""

    @staticmethod
    def _parse_model_selection(response: str) -> Optional[List[str]]:
        """Parse LLM response to extract model names."""
        try:
            # Try JSON parsing
            data = json.loads(response)
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract model names from text
        import re
        models = re.findall(r'(PELT|BinSeg|WIN|Kernel|Bayesian|Spectral|DeepCPD)', response)
        if models:
            return list(set(models))
        
        return None

    @staticmethod
    def _extract_script(response: str) -> Optional[str]:
        """Extract Python script from LLM response."""
        import re
        
        # Try to find code blocks
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'```python\s*\n(.*?)\n```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest code block (most likely the complete script)
                return max(matches, key=len).strip()
        
        # Fallback: try to find detect_change_points function
        if "def detect_change_points" in response:
            # Find the start of the function
            start = response.find("def detect_change_points")
            # Find the end (next class/def at root level or end of string)
            end = len(response)
            lines = response[start:].split("\n")
            script_lines = []
            for line in lines:
                if line and not line[0].isspace() and line[0] not in "#\"'" and line != line.lstrip():
                    break
                script_lines.append(line)
            return "\n".join(script_lines).strip()
        
        return None

    @staticmethod
    def _extract_change_points(traj: ExperimentTrajectory) -> List[int]:
        """Extract change points from trajectory."""
        # Try to find change points in the script execution
        if hasattr(traj, '_change_points'):
            return traj._change_points
        return []
