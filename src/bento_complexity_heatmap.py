# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
Bento cell for visualizing training_ne as a function of training_sufficiency and architecture characteristics.
Creates various plots showcasing different aspects of model architecture performance:
- Network depth analysis
- Layer width analysis
- Multi-subarchitecture configurations
- Architecture complexity heatmaps
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


# Get model parameter definitions from API configuration
def get_model_complexities() -> (
    Tuple[float, float, float, List[List[int]], List[List[int]], List[List[int]]]
):
    """Get model complexities from the API configuration."""
    api = ModelPerformanceAPI()
    config = api.model_config_dict

    small_arch = config["SMALL_NETWORK"]
    medium_arch = config["MEDIUM_NETWORK"]
    large_arch = config["LARGE_NETWORK"]

    # Use the actual complexity_ne values directly
    small_complexity_ne, _ = api.arch_to_scaled_complexities(small_arch)
    medium_complexity_ne, _ = api.arch_to_scaled_complexities(medium_arch)
    large_complexity_ne, _ = api.arch_to_scaled_complexities(large_arch)

    return (
        small_complexity_ne,
        medium_complexity_ne,
        large_complexity_ne,
        small_arch,
        medium_arch,
        large_arch,
    )


# Get actual complexity values from API
SMALL_COMPLEXITY_NE: float
MEDIUM_COMPLEXITY_NE: float
LARGE_COMPLEXITY_NE: float
SMALL_ARCH: List[List[int]]
MEDIUM_ARCH: List[List[int]]
LARGE_ARCH: List[List[int]]

(
    SMALL_COMPLEXITY_NE,
    MEDIUM_COMPLEXITY_NE,
    LARGE_COMPLEXITY_NE,
    SMALL_ARCH,
    MEDIUM_ARCH,
    LARGE_ARCH,
) = get_model_complexities()

# Print the actual values being used
print("📊 Model Complexities from API Configuration:")
print(f"   Small Network: {SMALL_ARCH} → {SMALL_COMPLEXITY_NE:.3f} complexity_ne")
print(f"   Medium Network: {MEDIUM_ARCH} → {MEDIUM_COMPLEXITY_NE:.3f} complexity_ne")
print(f"   Large Network: {LARGE_ARCH} → {LARGE_COMPLEXITY_NE:.3f} complexity_ne")


@dataclass
class PlotConfig:
    """Configuration class to reduce parameter passing complexity."""

    training_days_range: Tuple[int, int] = (1, 60)
    complexity_range: Tuple[float, float] = (0.0, 1.0)
    figsize: Tuple[int, int] = (14, 10)
    metric: str = "training_ne"
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
    T: Any, C: Any, metric: str, noise_scale: Optional[float]
) -> Tuple[Any, Any]:
    """Generate surface data using scaled complexities directly."""
    Z_smooth = np.zeros_like(T)
    Z_noisy = np.zeros_like(T)

    print(f"Calculating 3D surface for {metric} using scaled complexities...")
    print(f"🎛️  Noise parameter: {noise_scale}")

    # Create fresh API instance for this surface generation
    api_default_noise = ModelPerformanceAPI()

    dict_config_no_noise = api_default_noise.get_default_model_config_dict()
    dict_config_noise = api_default_noise.get_default_model_config_dict()
    dict_config_no_noise["GLOBAL_NOISE_SCALE"] = 0

    if noise_scale is not None:
        print("🎛️  Overriding API default noise scale with:", noise_scale)
        dict_config_noise["GLOBAL_NOISE_SCALE"] = noise_scale
    else:
        print(
            "🎛️  Using API default noise scale:", dict_config_noise["GLOBAL_NOISE_SCALE"]
        )

    api_no_noise = ModelPerformanceAPI(dict_config_no_noise)
    api_noise = ModelPerformanceAPI(dict_config_noise)

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            training_days = int(T[i, j])
            complexity_scaled_ne = C[i, j]
            complexity_scaled_qps = C[i, j]  # Use same complexity for both metrics

            # Calculate training sufficiency
            training_sufficiency = np.clip(
                training_days / api_no_noise.model_config_dict["MAX_TRAINING_DAYS"],
                0.0,
                1.0,
            )

            # Calculate performance directly using API's internal methods
            # No noise version
            training_ne_smooth = api_no_noise._calculate_base_performance(
                complexity_scaled_ne, training_sufficiency
            )
            qps_smooth = api_no_noise._calculate_qps(complexity_scaled_qps)

            # Noisy version
            training_ne_noisy = api_noise._calculate_base_performance(
                complexity_scaled_ne, training_sufficiency
            )
            qps_noisy = api_noise._calculate_qps(complexity_scaled_qps)

            # Apply noise if enabled
            if api_noise.model_config_dict["GLOBAL_NOISE_SCALE"] > 0.0:
                training_ne_noisy, qps_noisy = api_noise._apply_noise_to_metrics(
                    training_ne_noisy,
                    qps_noisy,
                    training_sufficiency,
                    api_noise.model_config_dict["GLOBAL_NOISE_SCALE"],
                )

            # Extract the requested metric
            if metric == "training_ne":
                Z_smooth[i, j] = training_ne_smooth
                Z_noisy[i, j] = training_ne_noisy
            else:
                Z_smooth[i, j] = qps_smooth
                Z_noisy[i, j] = qps_noisy

    return Z_smooth, Z_noisy


def _create_3d_plot_complexity(
    T: Any,
    C: Any,
    Z_smooth: Any,
    Z_noisy: Any,
    style: MetricStyle,
    figsize: Tuple[int, int],
) -> plt.Figure:
    """Create and configure the 3D plot with complexity axis."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create surface plot with noise
    surface = ax.plot_surface(
        T, C, Z_noisy, cmap=style.colormap, alpha=0.8, edgecolor="none"
    )

    # Overlay smooth surface as wireframe
    ax.plot_wireframe(
        T,
        C,
        Z_smooth,
        colors="black",
        alpha=0.3,
        linewidth=0.5,
        label="Smooth Base Surface",
    )

    # Customize the plot
    ax.set_xlabel("Training Days", fontsize=12, fontweight="bold")
    ax.set_ylabel("Complexity Scaled [0,1]", fontsize=12, fontweight="bold")
    ax.set_zlabel(style.z_label, fontsize=12, fontweight="bold")
    ax.set_title(
        style.title.replace("Model Params", "Complexity"),
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

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
    """Create 3D surface plot using scaled complexity directly."""
    # Create grids with 1 day step size for training and 0.05 increments for complexity
    training_steps = config.training_days_range[1] - config.training_days_range[0] + 1
    training_days_values = np.linspace(
        config.training_days_range[0],
        config.training_days_range[1],
        training_steps,
    )
    # Use 0.05 increments for complexity to get good resolution
    complexity_steps = (
        int((config.complexity_range[1] - config.complexity_range[0]) / 0.05) + 1
    )
    complexity_values = np.linspace(
        config.complexity_range[0],
        config.complexity_range[1],
        complexity_steps,
    )
    T, C = np.meshgrid(training_days_values, complexity_values)

    # Generate surface data through direct complexity calculation
    Z_smooth, Z_noisy = _generate_surface_data(T, C, config.metric, config.noise_scale)

    # Get style configuration and create plot with updated labels
    style = _get_metric_style(config.metric)
    fig = _create_3d_plot_complexity(T, C, Z_smooth, Z_noisy, style, config.figsize)

    return fig, (T, C, Z_noisy, Z_smooth)


def _calculate_reference_performance(
    training_days_values: Any, metric: str, noise_scale: Optional[float]
) -> Tuple[List[float], List[float], List[float]]:
    """Calculate performance for all three reference models through simulation."""
    small_performance = []
    medium_performance = []
    large_performance = []

    _log_noise_status(metric, noise_scale)

    # Create fresh API instance for reference performance calculation

    api = ModelPerformanceAPI()

    if noise_scale is not None:
        print("🎛️  Overriding API default noise scale with:", noise_scale)
        config_dict = api.model_config_dict.copy()
        config_dict["GLOBAL_NOISE_SCALE"] = noise_scale
        api = ModelPerformanceAPI(config_dict)

    for i, training_days in enumerate(training_days_values):
        training_days_int = int(training_days)

        # Use train_model to get final performance for each model size
        small_final_ne, small_final_qps, _ = api.train_model(
            training_days=training_days_int,
            arch=SMALL_ARCH,
            ignore_budget=True,
        )
        medium_final_ne, medium_final_qps, _ = api.train_model(
            training_days=training_days_int,
            arch=MEDIUM_ARCH,
            ignore_budget=True,
        )
        large_final_ne, large_final_qps, _ = api.train_model(
            training_days=training_days_int,
            arch=LARGE_ARCH,
            ignore_budget=True,
        )

        # Extract the requested metric
        if metric == "training_ne":
            small_performance.append(small_final_ne)
            medium_performance.append(medium_final_ne)
            large_performance.append(large_final_ne)
        elif metric == "qps":
            small_performance.append(small_final_qps)
            medium_performance.append(medium_final_qps)
            large_performance.append(large_final_qps)
        else:
            # For unsupported metrics, default to training_ne values
            small_final_ne, _, _ = api.train_model(
                training_days=training_days_int,
                arch=SMALL_ARCH,
                ignore_budget=True,
            )
            medium_final_ne, _, _ = api.train_model(
                training_days=training_days_int,
                arch=MEDIUM_ARCH,
                ignore_budget=True,
            )
            large_final_ne, _, _ = api.train_model(
                training_days=training_days_int,
                arch=LARGE_ARCH,
                ignore_budget=True,
            )
            small_performance.append(small_final_ne)
            medium_performance.append(medium_final_ne)
            large_performance.append(large_final_ne)

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

    # Plot curves with different styles using complexity_ne values
    # Use shorter labels to avoid text overlap
    curve_styles = [
        (
            small_performance,
            "#FF6B6B",
            "o",
            f"Small (complexity: {SMALL_COMPLEXITY_NE:.3f})",
        ),
        (
            medium_performance,
            "#4ECDC4",
            "s",
            f"Medium (complexity: {MEDIUM_COMPLEXITY_NE:.3f})",
        ),
        (
            large_performance,
            "#45B7D1",
            "^",
            f"Large (complexity: {LARGE_COMPLEXITY_NE:.3f})",
        ),
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
    # Improved legend positioning to avoid text overlap
    plt.legend(
        fontsize=10,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        ncol=1,
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
    figsize: Tuple[int, int] = (12, 8),
    metric: str = "training_ne",
    noise_scale: Optional[float] = None,
) -> Tuple[plt.Figure, Tuple[Any, List[float], List[float], List[float]]]:
    """Plot the 3 reference model curves using the unified performance function."""
    # Calculate training steps to always use 1 day step size
    training_steps = training_days_range[1] - training_days_range[0] + 1
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
        metric="training_ne",
        noise_scale=PLOT_NOISE_SCALE,
    )
    plt.show()
    return fig


def _create_training_ne_surface_plot() -> plt.Figure:
    """Create 3D surface plot for training_ne."""
    print("\n🎲 Creating 3D Surface Plot for Training NE...")
    print("📊 Graph 2 - 3D Training NE Surface: Using noise_scale=API_DEFAULT")
    config = PlotConfig(
        training_days_range=(6, 60),
        complexity_range=(0.0, 1.0),
        metric="training_ne",
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
        complexity_range=(0.0, 1.0),
        metric="qps",
        noise_scale=PLOT_NOISE_SCALE,
    )
    fig, _ = plot_3d_surface(config)
    plt.show()
    return fig


def plot_3d_surface_no_noise(
    training_days_range: Tuple[int, int] = (6, 60),
    complexity_range: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[int, int] = (14, 10),
    metric: str = "training_ne",
) -> Tuple[plt.Figure, Tuple[Any, Any, Any, Any]]:
    """
    Create 3D surface plot with NO noise for clean mathematical visualization.

    Args:
        training_days_range: Range of training days to visualize
        complexity_range: Range of complexity values [0,1]
        figsize: Figure size
        metric: Metric to visualize (default: training_ne)

    Returns:
        Figure and data tuple (T, C, Z_noisy, Z_smooth)
    """
    print(f"\n🎯 Creating Clean 3D Surface Plot for {metric.upper()} (NO NOISE)...")
    print("📊 Clean 3D Surface: Using noise_scale=0.0 (PURE MATHEMATICAL SURFACE)")

    config = PlotConfig(
        training_days_range=training_days_range,
        complexity_range=complexity_range,
        figsize=figsize,
        metric=metric,
        noise_scale=0.0,  # NO NOISE
    )

    # Calculate training steps to always use 1 day step size
    training_steps = config.training_days_range[1] - config.training_days_range[0] + 1
    # Calculate complexity steps to use 0.05 increments
    complexity_steps = (
        int((config.complexity_range[1] - config.complexity_range[0]) / 0.05) + 1
    )

    # Create parameter grids
    training_days_values = np.linspace(
        config.training_days_range[0],
        config.training_days_range[1],
        training_steps,
    )
    complexity_values = np.linspace(
        config.complexity_range[0],
        config.complexity_range[1],
        complexity_steps,
    )
    T, C = np.meshgrid(training_days_values, complexity_values)

    # Generate surface data with NO noise (clean mathematical surface)
    Z_smooth, Z_noisy = _generate_surface_data(T, C, config.metric, config.noise_scale)

    # Since noise_scale=0.0, Z_smooth and Z_noisy should be identical
    # Create a clean 3D plot without noise artifacts
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Get style configuration
    style = _get_metric_style(config.metric)

    # Update style for clean visualization
    clean_title = style.title.replace("NOISY", "CLEAN MATHEMATICAL").replace(
        "Model Params", "Complexity"
    )

    # Create clean surface plot (no noise, so only one surface needed)
    surface = ax.plot_surface(
        T, C, Z_smooth, cmap=style.colormap, alpha=0.9, edgecolor="none"
    )

    # Add contour lines for better readability
    ax.contour(T, C, Z_smooth, levels=10, colors="black", alpha=0.4, linewidths=0.5)

    # Customize the plot
    ax.set_xlabel("Training Days", fontsize=12, fontweight="bold")
    ax.set_ylabel("Complexity Scaled [0,1]", fontsize=12, fontweight="bold")
    ax.set_zlabel(style.z_label, fontsize=12, fontweight="bold")
    ax.set_title(clean_title, fontsize=14, fontweight="bold", pad=20)

    # Add colorbar and annotations
    fig.colorbar(surface, shrink=0.5, aspect=20, label=style.colorbar_label)
    ax.text2D(
        0.02,
        0.98,
        "🎯 CLEAN MATHEMATICAL SURFACE: No noise applied",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightblue", "alpha": 0.9},
    )
    ax.view_init(elev=25, azim=45)

    plt.show()
    return fig, (T, C, Z_noisy, Z_smooth)


def plot_3d_surface_training_ne_no_noise(
    training_days_range: Tuple[int, int] = (6, 60),
    complexity_range: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[int, int] = (14, 10),
) -> Tuple[plt.Figure, Tuple[Any, Any, Any, Any]]:
    """
    Create 3D surface plot for TRAINING_NE with NO noise for clean mathematical visualization.

    Args:
        training_days_range: Range of training days to visualize
        complexity_range: Range of complexity values [0,1]
        figsize: Figure size

    Returns:
        Figure and data tuple (T, C, Z_noisy, Z_smooth)
    """
    print("\n🎯 Creating Clean 3D Surface Plot for TRAINING_NE (NO NOISE)...")
    print(
        "📊 Clean Training NE Surface: Using noise_scale=0.0 (PURE MATHEMATICAL SURFACE)"
    )

    # Use the general plot_3d_surface_no_noise function
    return plot_3d_surface_no_noise(
        training_days_range=training_days_range,
        complexity_range=complexity_range,
        figsize=figsize,
        metric="training_ne",
    )


def _create_clean_training_ne_3d_surface_plot() -> plt.Figure:
    """Create clean 3D surface plot with NO noise for training_ne."""
    fig, _ = plot_3d_surface_training_ne_no_noise(
        training_days_range=(6, 60),
        complexity_range=(0.0, 1.0),
    )
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


def plot_clean_reference_curves(
    training_days_range: Tuple[int, int] = (6, 60),
    figsize: Tuple[int, int] = (12, 8),
    metric: str = "training_ne",
) -> Tuple[plt.Figure, Tuple[Any, List[float], List[float], List[float]]]:
    """
    Plot the 3 reference model curves with NO noise for clean visualization.

    Args:
        training_days_range: Range of training days to plot
        figsize: Figure size
        metric: Metric to plot (default: training_ne)

    Returns:
        Figure and data tuple
    """
    print("\n📐 Creating Clean Reference Model Curves (NO NOISE)...")
    print(
        "📊 Graph - Clean Curves: Using noise_scale=0.0 (PURE MATHEMATICAL FUNCTIONS)"
    )

    # Calculate training steps to always use 1 day step size
    training_steps = training_days_range[1] - training_days_range[0] + 1
    training_days_values = np.linspace(
        training_days_range[0], training_days_range[1], training_steps
    )

    # Generate performance data with NO noise (noise_scale=0.0)
    small_performance, medium_performance, large_performance = (
        _calculate_reference_performance(training_days_values, metric, noise_scale=0.0)
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

    # Add clean curves annotation
    plt.figtext(
        0.02,
        0.95,
        "⚡ CLEAN MATHEMATICAL CURVES: No noise applied - pure underlying functions",
        fontsize=12,
        fontweight="bold",
        color="darkblue",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightblue", "alpha": 0.9},
    )

    # Update title to reflect no noise
    plt.suptitle(
        "Clean Reference Model Performance Curves (NO NOISE)\n3 Base Mathematical Functions Used for Surface Interpolation",
        fontsize=16,
        fontweight="bold",
        y=0.92,
    )

    plt.show()
    return fig, (
        training_days_values,
        small_performance,
        medium_performance,
        large_performance,
    )


def _create_clean_curves_plot() -> plt.Figure:
    """Create reference curves plot with NO noise for clean mathematical visualization."""
    fig, _ = plot_clean_reference_curves(
        training_days_range=(6, 60),
        metric="training_ne",
    )
    return fig


def main() -> None:
    """Main function to create all visualizations in the bento cell."""
    print("🚀 Starting Bento Complexity Heatmap Visualization Suite")
    print("=" * 60)

    try:
        # Create all visualizations
        _create_clean_curves_plot()
        _create_reference_curves_plot()
        _create_training_ne_surface_plot()
        _create_qps_surface_plot()
        _create_clean_training_ne_3d_surface_plot()

        # Log analysis results
        _log_analysis_results()

        print("\n🎉 All visualizations completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during visualization creation: {str(e)}")
        raise


# =======================================================================


# For direct Bento execution
if __name__ == "__main__":
    main()
