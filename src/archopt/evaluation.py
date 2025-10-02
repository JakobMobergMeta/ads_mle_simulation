# src/archopt/evaluation.py
from __future__ import annotations
from typing import Callable, Dict, Any, Optional

from .config import EVAL_TRAINING_DAYS, QPS_MIN  # ensure QPS_MIN is in your config
from .networks import validate_arch_nested
from simenv.model_performance_api import ModelPerformanceAPI
import math

def _qps_penalty_soft(qps: float, qps_min: float, alpha: float = 2.0) -> float:
    """
    Smooth penalty in [0,1]. If qps >= qps_min => 1.0; else scales like (qps/qps_min)^alpha.
    Larger alpha penalizes low QPS more aggressively.
    """
    if qps_min <= 0:
        return 1.0
    x = max(0.0, min(1.0, qps / qps_min))
    return x ** alpha


def make_evaluator(
    model_object,
    qps_min: float = 3500.0,
    soft_qps_tau: Optional[float] = None,
) -> Callable[[Dict[str, Any]], float]:
    """
    Returns an evaluate(cfg) -> score function.

    - Hard gate (default): if qps < qps_min -> return a large negative sentinel.
    - Soft gate: if soft_qps_tau is set, multiply the model score by an exponential
      penalty when qps < qps_min:
          penalty = exp((qps - qps_min) / tau)
      which is 1 at qps=qps_min and decays smoothly as qps falls below the target.
    """

    def _score(ne: float, qps: float) -> float:
        base = float(model_object.get_score(ne))

        # Hard gate
        if soft_qps_tau is None:
            return base if qps >= qps_min else -100.0

        # Soft penalty
        if qps >= qps_min:
            return base
        tau = max(1e-9, float(soft_qps_tau))
        penalty = math.exp((qps - qps_min) / tau)
        return base * penalty

    def evaluate_cfg(cfg: Dict[str, Any]) -> float:
        arch = cfg["arch"]
        ne, qps, _ = model_object.train_model(
            arch=arch,
            training_days=EVAL_TRAINING_DAYS,
            ignore_budget=True,
        )
        return _score(ne, qps)

    return evaluate_cfg