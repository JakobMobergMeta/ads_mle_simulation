# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Simple interpolation for creating smooth surfaces from 3 reference curves with plateau.
Clean, straightforward approach using basic interpolation techniques.
"""

from typing import Callable

import numpy as np


class SimplePerformanceInterpolator:
    """
    Simple interpolation between 3 reference curves with proper plateau handling.
    """

    def __init__(
        self,
        small_curve: Callable[[float], float],
        medium_curve: Callable[[float], float],
        large_curve: Callable[[float], float],
        small_complexity: float,
        medium_complexity: float,
        large_complexity: float,
    ) -> None:
        """
        Initialize with 3 reference curves.

        Args:
            small_curve, medium_curve, large_curve: Performance curves f(training_sufficiency) -> performance
            small_params, medium_params, large_params: Parameter values for each curve
            saturation_point: Where performance plateaus
        """
        self.small_curve = small_curve
        self.medium_curve = medium_curve
        self.large_curve = large_curve
        self.small_complexity = small_complexity
        self.medium_complexity = medium_complexity
        self.large_complexity = large_complexity

    def interpolate_1d(self, complexity: float, training_sufficiency: float) -> float:
        """
        Simple 1D interpolation between the 3 curves based on parameter count.

        Args:
            params_millions: Model parameter count in millions
            training_sufficiency: Training sufficiency (0.0 to 1.0)

        Returns:
            Interpolated performance value
        """

        # Get performance values from all 3 curves at this training level
        small_perf = self.small_curve(training_sufficiency)
        medium_perf = self.medium_curve(training_sufficiency)
        large_perf = self.large_curve(training_sufficiency)

        # Reference parameter points and their performance values
        complexity_points = np.array(
            [self.small_complexity, self.medium_complexity, self.large_complexity]
        )
        perf_values = np.array([small_perf, medium_perf, large_perf])

        # Simple linear interpolation between curves
        if complexity <= self.small_complexity:
            # Below small model - use small curve
            return small_perf
        elif complexity >= self.large_complexity:
            # At or above large model (but below saturation) - use large curve
            return large_perf
        else:
            # Interpolate between the curves
            return float(np.interp(complexity, complexity_points, perf_values))


def create_simple_interpolator(
    small_complexity: float,
    medium_complexity: float,
    large_complexity: float,
) -> SimplePerformanceInterpolator:
    """
    Create interpolator using the exact curves from performance.py.

    Returns:
        Configured SimplePerformanceInterpolator
    """

    def small_curve(training_sufficiency: float) -> float:
        small_plateau_start = 0.6
        small_start_y = 0.62
        if training_sufficiency > small_plateau_start:
            training_sufficiency = small_plateau_start
        return small_start_y - 0.04 / (1.0 + np.exp(-10 * (training_sufficiency - 0.2)))

    def medium_curve(training_sufficiency: float) -> float:
        medium_plateau_start = 0.8
        medium_start_y = 0.65
        # Adjusted: Start slightly higher by increasing the base value from 0.66 to 0.68
        if training_sufficiency > medium_plateau_start:
            training_sufficiency = medium_plateau_start
        return medium_start_y - 0.16 / (1.0 + np.exp(-8 * (training_sufficiency - 0.3)))

    def large_curve(training_sufficiency: float) -> float:
        # Adjusted: End slightly lower by reducing the final performance range from 0.18 to 0.15
        return 0.68 - 0.20 / (1.0 + np.exp(-8 * (training_sufficiency - 0.3)))

    return SimplePerformanceInterpolator(
        small_curve=small_curve,
        medium_curve=medium_curve,
        large_curve=large_curve,
        small_complexity=small_complexity,
        medium_complexity=medium_complexity,
        large_complexity=large_complexity,
    )
