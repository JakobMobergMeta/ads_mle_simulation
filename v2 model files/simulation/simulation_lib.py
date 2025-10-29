# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
Simulation tools for MLEA.

WARNING: This file is used by ML Exploration Agent with special settings,
please DO NOT import modules in this file unless necessary.

These tools provide a simplified interface for agents to interact with the MLEA
simulation system for model training, performance analysis, and budget management.
"""

from typing import Dict, List, Optional, Union

from mlea.simulation import BudgetExceededError, ModelPerformanceAPI
from mlea.tools.simulation.simulation_enums import SimulationConfigs


# Global API instance to maintain state across calls
_simulation_api: Optional[ModelPerformanceAPI] = None


def _get_api() -> ModelPerformanceAPI:
    """Get or create the global simulation API instance."""
    global _simulation_api
    if _simulation_api is None:
        _simulation_api = ModelPerformanceAPI()
    return _simulation_api


def simulation_train_model(
    training_days: int,
    arch: List[List[int]],
    ignore_budget: bool = False,
) -> Dict[str, Union[float, Dict[int, Dict[str, float]]]]:
    """Train a model using simulation and return performance metrics.

    This tool provides access to the MLEA model performance simulation system.
    It simulates training a machine learning model with the specified architecture
    for the given number of days and returns performance metrics including
    normalized error, queries per second, and learning curves.

    Args:
        training_days: Number of training days (1-60)
        arch: Model architecture as nested list of sub-architectures, where each
              sub-architecture is a list of layer sizes (e.g., [[512, 512]])
        ignore_budget: Whether to ignore budget constraints for training

    Returns:
        Dictionary containing:
        - training_ne: Final normalized error (0.0-1.0, lower is better)
        - qps: Queries per second (performance metric)
        - learning_curve: Training progress by day with nested dict format:
          {day: {'ne': error_value, 'qps': qps_value}}

    Raises:
        BudgetExceededError: If training exceeds available budget
        ValueError: If architecture is invalid or training_days out of range

    Example:
        >>> result = simulation_train_model(training_days=20, arch=[[512, 512]])
        >>> print(f"Final NE: {result['training_ne']:.4f}")
        >>> print(f"QPS: {result['qps']:.0f}")
        >>> # Check learning curve at day 10
        >>> day_10_ne = result['learning_curve'][10]['ne']
        >>> print(f"Day 10 NE: {day_10_ne:.4f}")
    """
    api = _get_api()

    # Validate training days
    if not (1 <= training_days <= 60):
        raise ValueError("Training days must be between 1 and 60")

    # Validate architecture format - must be nested list
    if not isinstance(arch, list) or len(arch) == 0:
        raise ValueError("Architecture must be a non-empty list of sub-architectures")

    for i, sub_arch in enumerate(arch):
        if not isinstance(sub_arch, list) or len(sub_arch) == 0:
            raise ValueError(
                f"Sub-architecture {i} must be a non-empty list of layer sizes, "
                f"got {type(sub_arch)}: {sub_arch}"
            )

        for j, layer_size in enumerate(sub_arch):
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise ValueError(
                    f"Layer size at sub-architecture {i}, position {j} must be a positive integer, "
                    f"got {type(layer_size)}: {layer_size}"
                )

    try:
        training_ne, qps, learning_curve = api.train_model(
            training_days=training_days,
            arch=arch,
            ignore_budget=ignore_budget,
        )

        return {
            "training_ne": training_ne,
            "qps": qps,
            "learning_curve": learning_curve,
        }
    except BudgetExceededError as e:
        # Re-raise with more context for agents
        raise BudgetExceededError(
            f"Training failed due to budget constraints: {e}. "
            f"Consider using ignore_budget=True or check remaining budget first."
        )


def simulation_get_budget_status() -> Dict[str, Union[float, int]]:
    """Get current simulation budget information.

    Returns:
        Dictionary containing:
        - remaining_budget: GPU-days remaining
        - total_budget: Total budget in GPU-days
        - budget_used: GPU-days already used
        - budget_percentage_used: Percentage of budget used

    Example:
        >>> status = simulation_get_budget_status()
        >>> print(f"Budget remaining: {status['remaining_budget']} GPU-days")
        >>> print(f"Budget used: {status['budget_percentage_used']:.1f}%")
    """
    api = _get_api()

    remaining = api.get_remaining_budget()
    config = api.get_default_model_config_dict()
    total = config.get("TOTAL_BUDGET_GPU_DAYS", 8000)
    used = total - remaining
    percentage_used = (used / total) * 100 if total > 0 else 0

    return {
        "remaining_budget": remaining,
        "total_budget": total,
        "budget_used": used,
        "budget_percentage_used": percentage_used,
    }


def simulation_reset_budget() -> Dict[str, str]:
    """Reset the simulation budget to full amount.

    Returns:
        Dictionary with status message

    Example:
        >>> result = simulation_reset_budget()
        >>> print(result['message'])
    """
    api = _get_api()
    api.reset_budget()

    return {"message": "Simulation budget has been reset to full amount"}


def simulation_calculate_cost(
    training_days: int, gpus_per_day: Optional[int] = None
) -> Dict[str, Union[float, int, bool]]:
    """Calculate the cost of training a model without actually training it.

    Args:
        training_days: Number of training days
        gpus_per_day: GPUs per day (uses default if not provided)

    Returns:
        Dictionary containing:
        - total_cost: Total cost in GPU-days
        - daily_cost: Cost per day in GPU-days
        - gpus_per_day: Number of GPUs used per day
        - can_afford: Whether current budget can afford this training

    Example:
        >>> cost = simulation_calculate_cost(training_days=10)
        >>> print(f"Training will cost {cost['total_cost']} GPU-days")
        >>> print(f"Can afford: {cost['can_afford']}")
    """
    api = _get_api()

    # Get default GPUs per day if not provided
    if gpus_per_day is None:
        config = api.get_default_model_config_dict()
        gpus_per_day = config.get("GPUS_PER_DAY", 8)

    total_cost = api.calculate_cost(
        training_days=training_days, gpus_per_day=gpus_per_day
    )
    daily_cost = total_cost / training_days if training_days > 0 else 0
    remaining_budget = api.get_remaining_budget()
    can_afford = remaining_budget >= total_cost

    return {
        "total_cost": total_cost,
        "daily_cost": daily_cost,
        "gpus_per_day": gpus_per_day,
        "can_afford": can_afford,
    }


def simulation_use_config(
    network_config_name: str,
) -> Dict[str, str]:
    global _simulation_api

    # If network_config_name is provided, look up the configuration
    if not hasattr(SimulationConfigs, network_config_name):
        available = [
            attr for attr in dir(SimulationConfigs) if not attr.startswith("_")
        ]
        raise ValueError(
            f"Network configuration '{network_config_name}' not found. "
            f"Available configurations: {', '.join(available)}"
        )
    config_updates = getattr(SimulationConfigs, network_config_name)

    # Get current config and update it
    api = _get_api()
    current_config = api.get_default_model_config_dict()
    current_config.update(config_updates)

    # Create new API instance with updated config
    _simulation_api = ModelPerformanceAPI(model_config_dict=current_config)

    updated_keys = list(config_updates.keys())
    config_source = (
        f"network config '{network_config_name}'"
        if network_config_name
        else "custom config"
    )
    return {
        "message": f"Updated simulation config using {config_source} for keys: {', '.join(updated_keys)}"
    }
