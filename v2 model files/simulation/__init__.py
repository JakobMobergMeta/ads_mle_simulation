# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
MLEA Simulation Module

This module provides model performance simulation functionality for machine learning
experimentation and analysis. It includes APIs for simulating model training curves,
performance interpolation, and budget tracking.

Key Components:
- ModelPerformanceAPI: Core API for simulating model training performance
- SimplePerformanceInterpolator: Interpolation between reference performance curves
- Complexity visualization and heatmap generation tools
"""

from mlea.simulation.model_performance_api import (
    BudgetExceededError,
    CompositeKeyCache,
    ModelPerformanceAPI,
)
from mlea.simulation.simple_interpolation import (
    create_simple_interpolator,
    SimplePerformanceInterpolator,
)

__all__ = [
    "ModelPerformanceAPI",
    "BudgetExceededError",
    "CompositeKeyCache",
    "SimplePerformanceInterpolator",
    "create_simple_interpolator",
]
