"""Tests for EvoTS-Agent."""

import numpy as np
import pytest

from evots_agent import (
    EvoTSAgent,
    EvoTSConfig,
    compute_eda,
    validate_boundaries,
    ModelBank,
    TrajectoryPool,
    ExperimentTrajectory,
)


class TestEDA:
    """Tests for exploratory data analysis."""

    def test_basic_eda(self):
        np.random.seed(42)
        data = np.random.randn(500)
        profile = compute_eda(data)
        
        assert profile.sequence_length == 500
        assert profile.dimensionality == 1
        assert 0 <= profile.missing_ratio <= 1
        assert profile.local_mean_discrepancy >= 0
        assert profile.local_variance_discrepancy >= 0

    def test_multivariate_eda(self):
        np.random.seed(42)
        data = np.random.randn(500, 3)
        profile = compute_eda(data)
        
        assert profile.sequence_length == 500
        assert profile.dimensionality == 3

    def test_with_change_points(self):
        np.random.seed(42)
        # Create data with clear change point at 250
        data = np.concatenate([
            np.random.randn(250) * 1.0 + 0,
            np.random.randn(250) * 2.0 + 5,
        ])
        profile = compute_eda(data)
        
        # Should detect high mean discrepancy
        assert profile.local_mean_discrepancy > 0.5


class TestModelBank:
    """Tests for model bank."""

    def test_all_models_registered(self):
        bank = ModelBank()
        assert len(bank.get_all()) >= 7

    def test_get_script(self):
        bank = ModelBank()
        script = bank.get_script("PELT")
        assert "detect_change_points" in script

    def test_filter_by_strength(self):
        bank = ModelBank()
        mean_shift = bank.filter_by_strength(["mean_shift"])
        assert "PELT" in mean_shift or "BinSeg" in mean_shift


class TestValidation:
    """Tests for boundary validation."""

    def test_perfect_match(self):
        pred = [100, 200, 300]
        ref = [100, 200, 300]
        metrics = validate_boundaries(pred, ref)
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_partial_match(self):
        pred = [105, 205]
        ref = [100, 200]
        metrics = validate_boundaries(pred, ref, tolerance=0.05, sequence_length=500)
        assert metrics["f1"] > 0.5

    def test_empty_predictions(self):
        pred = []
        ref = [100, 200]
        metrics = validate_boundaries(pred, ref)
        assert metrics["recall"] == 0.0

    def test_hausdorff_distance(self):
        pred = [110]
        ref = [100]
        metrics = validate_boundaries(pred, ref, sequence_length=500)
        assert metrics["hausdorff"] == 10.0


class TestTrajectory:
    """Tests for trajectory recording."""

    def test_create_trajectory(self):
        traj = ExperimentTrajectory(
            step=0,
            model_name="PELT",
            validation_score=0.85,
            decision="Accept",
        )
        assert traj.step == 0
        assert traj.model_name == "PELT"
        assert traj.validation_score == 0.85

    def test_trajectory_pool(self):
        pool = TrajectoryPool()
        
        for i in range(5):
            traj = ExperimentTrajectory(
                step=i,
                model_name="PELT",
                validation_score=0.5 + i * 0.1,
                decision="Accept",
            )
            pool.add(traj)
        
        assert len(pool) == 5
        incumbent = pool.get_incumbent()
        assert incumbent.validation_score == 0.9

    def test_save_load(self, tmp_path):
        pool = TrajectoryPool()
        pool.add(ExperimentTrajectory(
            step=0, model_name="PELT", validation_score=0.8
        ))
        
        path = str(tmp_path / "pool.json")
        pool.save(path)
        
        new_pool = TrajectoryPool()
        new_pool.load(path)
        assert len(new_pool) == 1


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        np.random.seed(42)
        
        # Create synthetic data with change points
        n = 500
        true_bps = [150, 350]
        data = np.concatenate([
            np.random.randn(true_bps[0]) * 1,
            np.random.randn(true_bps[1] - true_bps[0]) * 2 + 3,
            np.random.randn(n - true_bps[1]) * 0.5 - 2,
        ])
        
        # Run agent with mock LLM
        def mock_llm(system, prompt):
            return '{"models": ["PELT", "BinSeg"]}', ""
        
        config = EvoTSConfig(budget=3, top_k=2)
        agent = EvoTSAgent(llm=mock_llm, config=config)
        
        result = agent.detect(data, validation_boundaries=true_bps)
        
        assert result.f1_score > 0
        assert len(result.change_points) > 0
