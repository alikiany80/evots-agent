"""Code execution and validation for change-point detection."""

from __future__ import annotations

import io
import json
import math
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


def validate_boundaries(
    predicted: List[int],
    reference: List[int],
    tolerance: float = 0.05,
    sequence_length: Optional[int] = None,
) -> Dict[str, float]:
    """
    Boundary-aware validation for change-point detection.
    
    Computes precision, recall, F1 with tolerance window around reference points,
    plus Hausdorff distance between predicted and reference boundaries.
    """
    if not predicted and not reference:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "hausdorff": 0.0}
    
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hausdorff": float(max(reference)) if reference else 0.0}
    
    if not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hausdorff": float(max(predicted)) if predicted else 0.0}
    
    n = sequence_length or max(max(predicted, default=0), max(reference, default=0)) + 1
    
    # Tolerance window
    tol = max(int(math.ceil(tolerance * n)), 5)
    
    # Match predictions to references
    ref_matched = set()
    pred_matched = set()
    
    for pred_idx in sorted(predicted):
        for ref_idx in sorted(reference):
            if ref_idx in ref_matched:
                continue
            if abs(pred_idx - ref_idx) <= tol:
                pred_matched.add(pred_idx)
                ref_matched.add(ref_idx)
                break
    
    tp = len(pred_matched)
    fp = len(predicted) - len(pred_matched)
    fn = len(reference) - len(ref_matched)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Hausdorff distance
    if not predicted:
        hausdorff = max(reference)
    elif not reference:
        hausdorff = max(predicted)
    else:
        hausdorff = _hausdorff_distance(predicted, reference)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hausdorff": hausdorff,
    }


def _hausdorff_distance(set_a: List[int], set_b: List[int]) -> float:
    """Compute Hausdorff distance between two sets of points."""
    if not set_a or not set_b:
        return 0.0
    
    max_min_dist = 0.0
    
    # Directed Hausdorff: A -> B
    for a in set_a:
        min_dist = min(abs(a - b) for b in set_b)
        max_min_dist = max(max_min_dist, min_dist)
    
    # Directed Hausdorff: B -> A
    for b in set_b:
        min_dist = min(abs(a - b) for a in set_a)
        max_min_dist = max(max_min_dist, min_dist)
    
    return max_min_dist


class CodeExecutor:
    """Safely execute change-point detection scripts."""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.execution_log = ""
    
    def execute(self, script: str, data: np.ndarray) -> Tuple[bool, List[int], str]:
        """
        Execute a detection script on data.
        
        Returns:
            (success, change_points, log)
        """
        import signal
        
        # Create execution namespace
        exec_namespace = {"np": np, "numpy": np}
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        change_points: List[int] = []
        success = False
        
        try:
            # Execute the script
            exec(script, exec_namespace)
            
            # Call detect_change_points
            if "detect_change_points" in exec_namespace:
                func = exec_namespace["detect_change_points"]
                result = func(data)
                
                if isinstance(result, (list, tuple)):
                    change_points = [int(x) for x in result]
                    success = True
                elif isinstance(result, np.ndarray):
                    change_points = [int(x) for x in result.tolist()]
                    success = True
                else:
                    self.execution_log = f"Unexpected result type: {type(result)}"
            else:
                self.execution_log = "No detect_change_points function found"
        
        except Exception as e:
            self.execution_log = traceback.format_exc()
        
        finally:
            stdout_val = sys.stdout.getvalue()
            stderr_val = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            if stdout_val:
                self.execution_log += f"\nSTDOUT:\n{stdout_val}"
            if stderr_val:
                self.execution_log += f"\nSTDERR:\n{stderr_val}"
        
        return success, change_points, self.execution_log
    
    def execute_and_validate(
        self,
        script: str,
        data: np.ndarray,
        validation_boundaries: List[int],
        tolerance: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Execute script and validate against reference boundaries.
        
        Returns dict with success, change_points, metrics, log.
        """
        import numpy as np
        
        success, change_points, log = self.execute(script, data)
        
        result = {
            "success": success,
            "change_points": change_points,
            "metrics": {},
            "log": log,
        }
        
        if success:
            metrics = validate_boundaries(
                change_points, validation_boundaries,
                tolerance=tolerance,
                sequence_length=len(data),
            )
            result["metrics"] = metrics
        
        return result
