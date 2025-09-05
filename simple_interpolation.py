# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Simple interpolation for creating smooth surfaces from 3 reference curves with plateau.
Clean, straightforward approach using basic interpolation techniques.
"""

from typing import Any, Callable, Tuple

import numpy as np
from scipy.interpolate import griddata, interp1d


class SimplePerformanceInterpolator:
    """
    Simple interpolation between 3 reference curves with proper plateau handling.
    """

    def __init__(
        self,
        small_curve: Callable[[float], float],
        medium_curve: Callable[[float], float],
        large_curve: Callable[[float], float],
        small_params: float,
        medium_params: float,
        large_params: float,
        saturation_point: float,
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
        self.small_params = small_params
        self.medium_params = medium_params
        self.large_params = large_params
        self.saturation_point = saturation_point

    def interpolate_1d(
        self, params_millions: float, training_sufficiency: float
    ) -> float:
        """
        Simple 1D interpolation between the 3 curves based on parameter count.

        Args:
            params_millions: Model parameter count in millions
            training_sufficiency: Training sufficiency (0.0 to 1.0)

        Returns:
            Interpolated performance value
        """
        # Handle plateau region - performance stays at large model level
        if params_millions >= self.saturation_point:
            return self.large_curve(training_sufficiency)

        # Get performance values from all 3 curves at this training level
        small_perf = self.small_curve(training_sufficiency)
        medium_perf = self.medium_curve(training_sufficiency)
        large_perf = self.large_curve(training_sufficiency)

        # Reference parameter points and their performance values
        param_points = np.array(
            [self.small_params, self.medium_params, self.large_params]
        )
        perf_values = np.array([small_perf, medium_perf, large_perf])

        # Simple linear interpolation between curves
        if params_millions <= self.small_params:
            # Below small model - use small curve
            return small_perf
        elif params_millions >= self.large_params:
            # At or above large model (but below saturation) - use large curve
            return large_perf
        else:
            # Interpolate between the curves
            return float(np.interp(params_millions, param_points, perf_values))

    def interpolate_cubic(
        self, params_millions: float, training_sufficiency: float
    ) -> float:
        """
        Cubic interpolation for smoother results.

        Args:
            params_millions: Model parameter count in millions
            training_sufficiency: Training sufficiency (0.0 to 1.0)

        Returns:
            Interpolated performance value with cubic smoothness
        """
        # Handle plateau region - performance stays at large model level
        if params_millions >= self.saturation_point:
            return self.large_curve(training_sufficiency)

        # Get performance values from all 3 curves
        small_perf = self.small_curve(training_sufficiency)
        medium_perf = self.medium_curve(training_sufficiency)
        large_perf = self.large_curve(training_sufficiency)

        # Reference points
        param_points = np.array(
            [self.small_params, self.medium_params, self.large_params]
        )
        perf_values = np.array([small_perf, medium_perf, large_perf])

        # Extend with boundary conditions for cubic spline
        if params_millions <= self.small_params:
            return small_perf
        elif params_millions >= self.large_params:
            return large_perf
        else:
            # Create cubic interpolator
            cubic_interp = interp1d(
                param_points,
                perf_values,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            result = cubic_interp(params_millions)
            return float(result)

    def interpolate_2d_grid(
        self,
        training_grid: Any,
        param_range: Tuple[float, float] = (0.05, 8.0),
        n_param_points: int = 30,
        method: str = "cubic",
    ) -> Callable[[Any, Any], Any]:
        """
        Create 2D grid interpolation for the entire surface.

        Args:
            training_grid: Training sufficiency values to sample
            param_range: Parameter range to cover
            n_param_points: Number of parameter points
            method: Interpolation method ('linear', 'cubic', 'nearest')

        Returns:
            2D interpolation function
        """
        # Create parameter grid
        param_grid = np.linspace(param_range[0], param_range[1], n_param_points)

        # Generate grid data
        param_coords = []
        training_coords = []
        performance_values = []

        for p in param_grid:
            for t in training_grid:
                param_coords.append(p)
                training_coords.append(t)
                # Use 1D interpolation to get the value
                performance_values.append(self.interpolate_1d(p, t))

        # Convert to arrays
        points: Any = np.column_stack([param_coords, training_coords])
        values: Any = np.array(performance_values)

        def grid_interpolator(params: Any, training: Any) -> Any:
            """2D grid interpolation function."""
            if np.isscalar(params):
                params = np.array([params])
            if np.isscalar(training):
                training = np.array([training])

            query_points = np.column_stack([params.ravel(), training.ravel()])
            result = griddata(
                points, values, query_points, method=method, fill_value=0.5
            )  # Default fallback

            return result.reshape(params.shape) if result.size > 1 else float(result)

        return grid_interpolator

    def interpolate_smooth_blend(
        self,
        params_millions: float,
        training_sufficiency: float,
        blend_width: float = 0.5,
    ) -> float:
        """
        Smooth blending between curves with adjustable transition width.

        Args:
            params_millions: Model parameter count in millions
            training_sufficiency: Training sufficiency (0.0 to 1.0)
            blend_width: Width of blending transitions (larger = smoother)

        Returns:
            Smoothly blended performance value
        """
        # Handle plateau region - performance stays at large model level
        if params_millions >= self.saturation_point:
            return self.large_curve(training_sufficiency)

        # Get curve values
        small_perf = self.small_curve(training_sufficiency)
        medium_perf = self.medium_curve(training_sufficiency)
        large_perf = self.large_curve(training_sufficiency)

        # Calculate smooth weights using Gaussian-like functions
        small_weight = np.exp(
            -(((params_millions - self.small_params) / blend_width) ** 2)
        )
        medium_weight = np.exp(
            -(((params_millions - self.medium_params) / blend_width) ** 2)
        )
        large_weight = np.exp(
            -(((params_millions - self.large_params) / blend_width) ** 2)
        )

        # Normalize weights
        total_weight = small_weight + medium_weight + large_weight
        small_weight /= total_weight
        medium_weight /= total_weight
        large_weight /= total_weight

        # Weighted combination
        return (
            small_weight * small_perf
            + medium_weight * medium_perf
            + large_weight * large_perf
        )


def create_simple_interpolator(
    small_params: float,
    medium_params: float,
    large_params: float,
    saturation_point: float,
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
        small_params=small_params,
        medium_params=medium_params,
        large_params=large_params,
        saturation_point=saturation_point,
    )
