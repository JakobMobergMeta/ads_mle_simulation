# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

#!/usr/bin/env python3
"""
Bento cell for visualizing eval_ne as a function of training_sufficiency and complexity.
Creates a 3D noisy surface visualization with smooth interpolation.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import the model performance API for performance calculations
from admarket.ads_copilot.common.training_simulation.model_performance_api import (
    ModelPerformanceAPI,
)

# Constants
MAX_TRAINING_DAYS: int = 60
PLOT_NOISE_SCALE: Optional[float] = None

# Model parameter definitions
SMALL_PARAMS: float = 0.089
MEDIUM_PARAMS: float = 1.256
LARGE_PARAMS: float = 4.41


@dataclass
class PlotConfig:
    """Configuration class to reduce parameter passing complexity."""

    training_days_range: Tuple[int, int] = (6, 60)
    model_params_range: Tuple[float, float] = (0.1, 6.0)
    training_steps: int = 15
    model_params_steps: int = 25
    figsize: Tuple[int, int] = (14, 10)
    metric: str = "eval_ne"
    noise_scale: Optional[float] = None


@dataclass
class MetricStyle:
    """Style configuration for different metrics."""

    colormap: str
    z_label: str
    title: str
    colorbar_label: str


def _get_metric_style(metric: str) -> MetricStyle:
    """Get style configuration for a given metric."""
    if metric == "qps":
        return MetricStyle(
            colormap="viridis",
            z_label="QPS (Higher = Better)",
            title="3D NOISY Surface: QPS vs Training & Model Params",
            colorbar_label="QPS",
        )
    else:
        return MetricStyle(
            colormap="RdYlGn_r",
            z_label=f"{metric.replace('_', ' ').title()} (Lower = Better)",
            title=f"3D NOISY Surface: {metric.replace('_', ' ').title()} vs Training & Model Params",
            colorbar_label=metric.replace("_", " ").title(),
        )


def _generate_surface_data(
    T: Any, P: Any, metric: str, noise_scale: Optional[float]
) -> Tuple[Any, Any]:
    """Generate surface data by calling simulation API for each point."""
    Z_smooth = np.zeros_like(T)
    Z_noisy = np.zeros_like(T)

    print(f"Calculating 3D surface for {metric} using API function...")
    print(f"🎛️  Noise parameter: {noise_scale}")

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            # Critical: Keep these API calls as they simulate the data points
            metrics = ModelPerformanceAPI.get_model_performance(
                training_days=T[i, j],
                params_millions=P[i, j],
                override_noise=noise_scale,
                ignore_budget=True,
            )
            base_value = metrics[metric]
            Z_smooth[i, j] = base_value
            Z_noisy[i, j] = base_value

    return Z_smooth, Z_noisy


def _create_3d_plot(
    T: Any,
    P: Any,
    Z_smooth: Any,
    Z_noisy: Any,
    style: MetricStyle,
    figsize: Tuple[int, int],
) -> plt.Figure:
    """Create and configure the 3D plot."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create surface plot with noise
    surface = ax.plot_surface(
        T, P, Z_noisy, cmap=style.colormap, alpha=0.8, edgecolor="none"
    )

    # Overlay smooth surface as wireframe
    ax.plot_wireframe(
        T,
        P,
        Z_smooth,
        colors="black",
        alpha=0.3,
        linewidth=0.5,
        label="Smooth Base Surface",
    )

    # Customize the plot
    ax.set_xlabel("Training Days", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model Params in M", fontsize=12, fontweight="bold")
    ax.set_zlabel(style.z_label, fontsize=12, fontweight="bold")
    ax.set_title(style.title, fontsize=14, fontweight="bold", pad=20)

    # Add colorbar and annotations
    fig.colorbar(surface, shrink=0.5, aspect=20, label=style.colorbar_label)
    ax.text2D(
        0.02,
        0.98,
        "More noise at low training sufficiency",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax.view_init(elev=25, azim=45)

    return fig


def plot_3d_surface(config: PlotConfig) -> Tuple[plt.Figure, Tuple[Any, Any, Any, Any]]:
    """Create 3D surface plot using the unified performance function."""
    # Create parameter grids
    training_days_values = np.linspace(
        config.training_days_range[0],
        config.training_days_range[1],
        config.training_steps,
    )
    model_params_values = np.linspace(
        config.model_params_range[0],
        config.model_params_range[1],
        config.model_params_steps,
    )
    T, P = np.meshgrid(training_days_values, model_params_values)

    # Generate surface data through simulation (keeps API calls)
    Z_smooth, Z_noisy = _generate_surface_data(T, P, config.metric, config.noise_scale)

    # Get style configuration and create plot
    style = _get_metric_style(config.metric)
    fig = _create_3d_plot(T, P, Z_smooth, Z_noisy, style, config.figsize)

    return fig, (T, P, Z_noisy, Z_smooth)


def _calculate_reference_performance(
    training_days_values: Any, metric: str, noise_scale: Optional[float]
) -> Tuple[List[float], List[float], List[float]]:
    """Calculate performance for all three reference models through simulation."""
    small_performance = []
    medium_performance = []
    large_performance = []

    _log_noise_status(metric, noise_scale)

    for i, training_days in enumerate(training_days_values):
        # Critical: Keep these API calls as they simulate the data points
        small_metrics = ModelPerformanceAPI.get_model_performance(
            training_days=training_days,
            params_millions=SMALL_PARAMS,
            override_noise=noise_scale,
            ignore_budget=True,
        )
        medium_metrics = ModelPerformanceAPI.get_model_performance(
            training_days=training_days,
            params_millions=MEDIUM_PARAMS,
            override_noise=noise_scale,
            ignore_budget=True,
        )
        large_metrics = ModelPerformanceAPI.get_model_performance(
            training_days=training_days,
            params_millions=LARGE_PARAMS,
            override_noise=noise_scale,
            ignore_budget=True,
        )

        small_performance.append(small_metrics[metric])
        medium_performance.append(medium_metrics[metric])
        large_performance.append(large_metrics[metric])

        # Debug print for first few points
        if i < 5:
            print(
                f"Day {training_days:.1f}: Small={small_performance[-1]:.4f}, Medium={medium_performance[-1]:.4f}, Large={large_performance[-1]:.4f}"
            )

    return small_performance, medium_performance, large_performance


def _log_noise_status(metric: str, noise_scale: Optional[float]) -> None:
    """Log noise parameter status for debugging."""
    print(f"Calculating reference curves for {metric} using API function...")
    if noise_scale is None:
        noise_status = "USING API DEFAULT"
    elif noise_scale == 0:
        noise_status = "SMOOTH (noise_scale=0)"
    else:
        noise_status = f"OVERRIDE (noise_scale={noise_scale})"
    print(f"🎛️  Noise parameter: {noise_status}")


def _plot_reference_curves(
    training_days_values: Any,
    small_performance: List[float],
    medium_performance: List[float],
    large_performance: List[float],
    figsize: Tuple[int, int],
    metric: str,
) -> plt.Figure:
    """Create and style the reference curves plot."""
    plt.figure(figsize=figsize)

    # Plot curves with different styles
    curve_styles = [
        (small_performance, "#FF6B6B", "o", f"Small Model ({SMALL_PARAMS}M params)"),
        (medium_performance, "#4ECDC4", "s", f"Medium Model ({MEDIUM_PARAMS}M params)"),
        (large_performance, "#45B7D1", "^", f"Large Model ({LARGE_PARAMS}M params)"),
    ]

    for performance, color, marker, label in curve_styles:
        plt.plot(
            training_days_values,
            performance,
            linewidth=4,
            color=color,
            marker=marker,
            markersize=6,
            alpha=0.8,
            markevery=10,
            label=label,
        )

    _add_plateau_lines()
    _style_plot(metric)
    _add_interpretation_text()

    return plt.gcf()


def _add_plateau_lines() -> None:
    """Add vertical lines showing model plateau points."""
    plateaus = [
        (0.6 * MAX_TRAINING_DAYS, "#FF6B6B", "Small Model Plateau"),
        (0.8 * MAX_TRAINING_DAYS, "#4ECDC4", "Medium Model Plateau"),
    ]

    for x_pos, color, label in plateaus:
        plt.axvline(
            x=x_pos,
            color=color,
            linestyle="--",
            alpha=0.5,
            linewidth=2,
            label=f"{label} ({x_pos:.0f} days)",
        )


def _style_plot(metric: str) -> None:
    """Apply styling to the reference curves plot."""
    plt.xlabel("Training Days", fontsize=14, fontweight="bold")
    plt.ylabel(
        f"{metric.replace('_', ' ').title()} (Lower = Better)",
        fontsize=14,
        fontweight="bold",
    )
    plt.title(
        "Reference Model Performance Curves\n3 Base Models Used for Surface Interpolation",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.legend(
        fontsize=11,
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
    )
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)


def _add_interpretation_text() -> None:
    """Add interpretation text to the plot."""
    interpretation = (
        "• Small model: Fast plateau at 60% training, limited final performance\n"
        "• Medium model: Plateau at 80% training, moderate final performance\n"
        "• Large model: No plateau, continues improving, best final performance\n"
        "• These 3 curves are interpolated to create smooth surfaces for any parameter count"
    )

    plt.figtext(
        0.02,
        0.02,
        interpretation,
        fontsize=9,
        style="italic",
        wrap=True,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, left=0.12, right=0.95, top=0.90)


def plot_reference_curves_lines(
    training_days_range: Tuple[int, int] = (6, 60),
    training_steps: int = 100,
    figsize: Tuple[int, int] = (12, 8),
    metric: str = "training_ne",
    noise_scale: Optional[float] = None,
) -> Tuple[plt.Figure, Tuple[Any, List[float], List[float], List[float]]]:
    """Plot the 3 reference model curves using the unified performance function."""
    training_days_values = np.linspace(
        training_days_range[0], training_days_range[1], training_steps
    )

    # Generate performance data through simulation (keeps API calls)
    small_performance, medium_performance, large_performance = (
        _calculate_reference_performance(training_days_values, metric, noise_scale)
    )

    # Create and style the plot
    fig = _plot_reference_curves(
        training_days_values,
        small_performance,
        medium_performance,
        large_performance,
        figsize,
        metric,
    )

    return fig, (
        training_days_values,
        small_performance,
        medium_performance,
        large_performance,
    )


def _create_reference_curves_plot() -> plt.Figure:
    """Create reference curves plot with logging."""
    print("\n📈 Creating Reference Model Curves (Small/Medium/Large)...")
    print("📊 Graph 1 - Reference Curves: Using noise_scale=API_DEFAULT")
    fig, _ = plot_reference_curves_lines(
        training_days_range=(6, 60),
        training_steps=100,
        metric="eval_ne",
        noise_scale=PLOT_NOISE_SCALE,
    )
    plt.show()
    return fig


def _create_eval_ne_surface_plot() -> plt.Figure:
    """Create 3D surface plot for eval_ne."""
    print("\n🎲 Creating 3D Surface Plot for Eval NE...")
    print("📊 Graph 2 - 3D Eval NE Surface: Using noise_scale=API_DEFAULT")
    config = PlotConfig(
        training_days_range=(6, 60),
        model_params_range=(0.1, 6.0),
        training_steps=20,
        model_params_steps=25,
        metric="eval_ne",
        noise_scale=PLOT_NOISE_SCALE,
    )
    fig, _ = plot_3d_surface(config)
    plt.show()
    return fig


def _create_qps_surface_plot() -> plt.Figure:
    """Create 3D surface plot for QPS."""
    print("\n⚡ Creating 3D Surface Plot for QPS...")
    print("📊 Graph 3 - 3D QPS Surface: Using noise_scale=API_DEFAULT")
    config = PlotConfig(
        training_days_range=(6, 60),
        model_params_range=(0.1, 6.0),
        training_steps=20,
        model_params_steps=25,
        metric="qps",
        noise_scale=PLOT_NOISE_SCALE,
    )
    fig, _ = plot_3d_surface(config)
    plt.show()
    return fig


def _log_analysis_results() -> None:
    """Log key observations from the analysis."""
    observations = {
        "Reference Curves": [
            "Small model plateaus early but has limited performance",
            "Medium model plateaus later with moderate performance",
            "Large model continues improving with best final performance",
            "These 3 curves are interpolated for any parameter count",
        ],
        "Eval NE Surface": [
            "Smooth base surface (black wireframe) shows interpolated performance",
            "Colored surface adds realistic noise on top of smooth base",
            "Noise is more pronounced at low training sufficiency",
            "Lower values (green) are better performance",
        ],
        "QPS Surface": [
            "Shows queries per second capability across model sizes",
            "Smaller models have higher QPS (less computational cost)",
            "Larger models have lower QPS but potentially better quality",
            "Higher values (yellow) are better throughput",
        ],
    }

    print("\n✅ All visualizations complete!")
    print("\n🔍 Key Observations:")
    for section, points in observations.items():
        print(f"• {section}:")
        for point in points:
            print(f"  - {point}")


def main() -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Main function to call from Bento notebook. Creates visualizations including reference curves and 3D surfaces."""
    print("🔥 Generating Performance Analysis...")
    print("=" * 60)

    # Create visualizations
    fig1 = _create_reference_curves_plot()
    fig2 = _create_eval_ne_surface_plot()
    fig3 = _create_qps_surface_plot()

    # Log results
    _log_analysis_results()

    return fig1, fig2, fig3


# For direct Bento execution
if __name__ == "__main__":
    main()
