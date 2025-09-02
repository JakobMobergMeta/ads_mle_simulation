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
        small_params: float = 0.089,
        medium_params: float = 1.256,
        large_params: float = 4.41,
        saturation_point: float = 4.41,
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


def create_simple_interpolator() -> SimplePerformanceInterpolator:
    """
    Create interpolator using the exact curves from performance.py.

    Returns:
        Configured SimplePerformanceInterpolator
    """

    def small_curve(training_sufficiency: float) -> float:
        small_plateau_start = 0.6
        small_plateau_value = 0.62 - 0.05 / (
            1.0 + np.exp(-10 * (small_plateau_start - 0.2))
        )

        if training_sufficiency <= small_plateau_start:
            return 0.62 - 0.05 / (1.0 + np.exp(-10 * (training_sufficiency - 0.2)))
        else:
            return small_plateau_value

    def medium_curve(training_sufficiency: float) -> float:
        medium_plateau_start = 0.8
        medium_plateau_value = 0.66 - 0.11 / (
            1.0 + np.exp(-8 * (medium_plateau_start - 0.3))
        )

        if training_sufficiency <= medium_plateau_start:
            return 0.66 - 0.11 / (1.0 + np.exp(-8 * (training_sufficiency - 0.3)))
        else:
            return medium_plateau_value

    def large_curve(training_sufficiency: float) -> float:
        return 0.68 - 0.18 / (1.0 + np.exp(-6 * (training_sufficiency - 0.43)))

    return SimplePerformanceInterpolator(
        small_curve=small_curve,
        medium_curve=medium_curve,
        large_curve=large_curve,
        small_params=0.089,
        medium_params=1.256,
        large_params=4.41,
        saturation_point=4.41,
    )


# Example usage
if __name__ == "__main__":
    # Create simple interpolator
    interpolator: SimplePerformanceInterpolator = create_simple_interpolator()

    print("Simple Interpolation Test:")
    print("=" * 50)

    # Test different interpolation methods
    test_points = [
        (0.089, 0.5),  # At small model
        (0.5, 0.5),  # Between small and medium
        (1.256, 0.5),  # At medium model
        (2.0, 0.5),  # Between medium and large
        (4.41, 0.5),  # At large model (saturation point)
        (6.0, 0.5),  # Above saturation (should plateau)
        (8.0, 0.5),  # Well above saturation (should plateau)
    ]

    print("\nLinear Interpolation:")
    print("-" * 30)
    for params, training in test_points:
        result = interpolator.interpolate_1d(params, training)
        print(f"Params: {params:5.3f}M -> Performance: {result:.4f}")

    print("\nCubic Interpolation:")
    print("-" * 30)
    for params, training in test_points:
        result = interpolator.interpolate_cubic(params, training)
        print(f"Params: {params:5.3f}M -> Performance: {result:.4f}")

    print("\nSmooth Blend Interpolation:")
    print("-" * 30)
    for params, training in test_points:
        result = interpolator.interpolate_smooth_blend(
            params, training, blend_width=0.8
        )
        print(f"Params: {params:5.3f}M -> Performance: {result:.4f}")

    print("\nTEST: Large model should vary with training:")
    print("-" * 40)
    training_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for t in training_vals:
        result = interpolator.large_curve(t)
        print(f"Large model at training {t:.1f}: {result:.4f}")

    print("\nTEST: Plateau should follow large model curve:")
    print("-" * 45)
    for t in training_vals:
        large_val = interpolator.large_curve(t)
        plateau_val = interpolator.interpolate_1d(6.0, t)  # 6M params (saturated)
        print(
            f"Training {t:.1f}: Large={large_val:.4f}, Plateau={plateau_val:.4f}, Match={abs(large_val - plateau_val) < 1e-10}"
        )
